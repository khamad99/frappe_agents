"""Totals and grouping over one doctype.

`aggregate_documents` answers "how many" and "how much" with numbers the
database added up, not numbers the model added up from a page of rows. It goes
through the same door as `search_documents` — the rule, then the user's read
permission, then `frappe.get_list` — and adds Frappe's own Report View gate,
`report` permission, because grouped totals are what Report View is for.

A total cannot carry a `<restricted>` mark the way a list cell can, so a field
the user may not read, or sees masked, is refused outright wherever it appears:
in a metric, in a group, or in a filter.

`get_list` applies query conditions but never the per-row `has_permission`
hooks. On a doctype that registers one, an SQL `GROUP BY` would count rows the
user cannot open, so the rows are listed and checked one by one instead, up to
`ROW_CHECK_CAP`. Past the cap the call is refused: a total over part of the data
is a wrong total with nothing on it to say so.

Dates are rolled up into days, months, quarters and years here rather than in
SQL, so the answer is the same on every v16 query engine.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import frappe
from frappe.model import no_value_fields
from frappe.utils import cint, flt, getdate

from frappe_agents.context.slices import (
	has_row_permission_hook,
	masked_fieldnames,
	may_read,
	status_word,
)
from frappe_agents.tools.base import CAPABILITY_READ, ToolDenied, row_cap
from frappe_agents.tools.charts import CHART_TYPES, MAX_TITLE, build_chart, resolve_chart
from frappe_agents.tools.read_tools import (
	DEFAULT_LIMIT,
	_as_bool,
	_require_read,
	_require_str,
	_resolve_filters,
	_resolve_limit,
)

MAX_GROUPS = 100
MAX_METRICS = 5
MAX_GROUP_FIELDS = 2
# Rows checked one by one on a doctype with a per-row permission hook.
ROW_CHECK_CAP = 5000
# Raw groups SQL may hand back before the roll-up; more means the grouping is too fine.
RAW_GROUP_CAP = 5000

FUNCTIONS = ("count", "sum", "avg", "min", "max")
PERIODS = ("day", "month", "quarter", "year")

NUMERIC_TYPES = frozenset({"Currency", "Float", "Int", "Percent"})
DATE_TYPES = frozenset({"Date", "Datetime"})
ORDERED_TYPES = NUMERIC_TYPES | DATE_TYPES
GROUP_TYPES = frozenset({"Link", "Select", "Data", "Check", "Int"}) | DATE_TYPES

# Standard columns a total may use, typed the way a DocField would be.
STANDARD_TYPES = {
	"owner": "Link",
	"modified_by": "Link",
	"creation": "Datetime",
	"modified": "Datetime",
	"docstatus": "Int",
}

COUNT_ALIAS = "_rows"

# How a metric is named on a chart: "Sum of Paid Amount", not "sum_paid_amount".
METRIC_WORDS = {
	"count": "Count of",
	"sum": "Sum of",
	"avg": "Average of",
	"min": "Minimum of",
	"max": "Maximum of",
}


def aggregate_documents(payload: dict) -> dict:
	"""Count, sum, average, minimum or maximum over one doctype, grouped or not."""
	doctype = _require_str(payload, "doctype")
	_require_read(doctype)
	if not frappe.has_permission(doctype, "report"):
		raise ToolDenied(f"You are not allowed to report on {doctype}.")

	meta = frappe.get_meta(doctype)
	if cint(meta.issingle):
		raise ValueError(f"{doctype} is a single record and has nothing to add up.")

	fields = _FieldCheck(meta)
	metrics = _resolve_metrics(payload.get("metrics"), fields)
	groups = _resolve_groups(payload.get("group_by"), fields)
	filters = _resolve_filters(meta, payload.get("filters"), fields.allowed_levels)
	for fieldname in filters:
		fields.require_unmasked(fieldname, "filter")
	limit = _resolve_limit(payload.get("limit"), row_cap(doctype, MAX_GROUPS))
	include_cancelled = _as_bool(payload.get("include_cancelled"))
	chart_spec = resolve_chart(payload.get("chart"))

	if cint(meta.is_submittable) and not include_cancelled and "docstatus" not in filters:
		filters = dict(filters, docstatus=("!=", 2))

	currency_fields = _currency_fields(meta, metrics, groups, fields)
	key_fields = _unique([group["field"] for group in groups] + currency_fields)

	checked_row_by_row = has_row_permission_hook(doctype)
	if checked_row_by_row:
		raw, filtered = _checked_rows(doctype, filters, key_fields, metrics)
	else:
		raw, filtered = _grouped_rows(doctype, filters, key_fields, metrics), 0

	split, notes = _currency_split(raw, currency_fields, groups)
	totals = _roll_up(raw, groups, split, metrics)
	# The grand total splits by currency too: one sum over two currencies is not a number.
	overall = _roll_up(raw, [], split, metrics)

	rows = [_group_row(key, bucket, groups, split, metrics) for key, bucket in totals.items()]
	rows = _order(rows, payload.get("order_by"), groups, split, metrics)
	rows_counted = sum(bucket.rows for bucket in overall.values())
	chart = _chart(chart_spec, rows, limit, [group["key"] for group in groups] + split, metrics, notes)

	return {
		"doctype": doctype,
		"metrics": [metric["alias"] for metric in metrics],
		"group_by": [group["key"] for group in groups] + split,
		"groups": rows[:limit],
		"group_count": len(rows),
		"truncated": len(rows) > limit,
		"limit": limit,
		"overall": [_group_row(key, bucket, [], split, metrics) for key, bucket in overall.items()],
		"rows_counted": rows_counted,
		"include_cancelled": include_cancelled,
		"checked_row_by_row": checked_row_by_row,
		"filtered_by_permission": filtered,
		"notes": notes,
		"chart": chart,
		# An aggregate shows no one document, so the audit row names none.
		"_docs_touched": f"{doctype}: aggregate of {rows_counted} row(s)",
		"_result_summary": f"{min(len(rows), limit)} group(s) over {rows_counted} row(s)",
	}


class _FieldCheck:
	"""Which fields of one doctype the current user may total, group or filter on."""

	def __init__(self, meta: Any):
		self.meta = meta
		self.allowed_levels = meta.get_permlevel_access("read")
		self.masked = masked_fieldnames(meta)

	def fieldtype(self, fieldname: Any, use: str) -> str:
		"""The field's type, after the same permission checks a filter gets — and masking."""
		if not isinstance(fieldname, str) or not fieldname.strip():
			raise ValueError(f"Every {use} needs a fieldname.")
		fieldname = fieldname.strip()
		if fieldname in STANDARD_TYPES:
			return STANDARD_TYPES[fieldname]

		df = self.meta.get_field(fieldname)
		if not df or df.fieldtype in no_value_fields:
			raise ValueError(f"Unknown field {fieldname} on {self.meta.name}.")
		if cint(df.permlevel) not in self.allowed_levels:
			raise ToolDenied(f"You are not allowed to {use} {self.meta.name} on {fieldname}.")
		self.require_unmasked(fieldname, use)
		return df.fieldtype

	def require_unmasked(self, fieldname: str, use: str) -> None:
		if fieldname in self.masked:
			raise ToolDenied(f"{fieldname} on {self.meta.name} is masked for you, so you cannot {use} on it.")

	def readable(self, fieldname: str) -> bool:
		df = self.meta.get_field(fieldname)
		return bool(df) and cint(df.permlevel) in self.allowed_levels and fieldname not in self.masked


def _resolve_metrics(requested: Any, fields: _FieldCheck) -> list[dict]:
	if not requested:
		requested = [{"fn": "count"}]
	if isinstance(requested, dict):
		requested = [requested]
	if not isinstance(requested, list) or len(requested) > MAX_METRICS:
		raise ValueError(
			f"metrics must be a list of 1 to {MAX_METRICS} objects like {{'fn': 'sum', 'field': 'amount'}}."
		)

	metrics: list[dict] = []
	for raw in requested:
		if not isinstance(raw, dict):
			raise ValueError("Each metric must be an object like {'fn': 'sum', 'field': 'amount'}.")
		fn = str(raw.get("fn") or "").strip().lower()
		if fn not in FUNCTIONS:
			raise ValueError(f"Metric fn must be one of {', '.join(FUNCTIONS)}.")
		field = raw.get("field")
		metric = {"fn": fn, "field": None, "fieldtype": None, "alias": fn, "label": "Count"}
		if fn != "count" or field:
			fieldtype = fields.fieldtype(field, fn)
			_require_type(fn, field, fieldtype)
			field = field.strip()
			metric.update(
				field=field,
				fieldtype=fieldtype,
				alias=f"{fn}_{field}",
				label=f"{METRIC_WORDS[fn]} {fields.meta.get_label(field)}",
			)
		metric["numeric"] = fn in ("count", "sum", "avg") or metric["fieldtype"] in NUMERIC_TYPES
		if metric["alias"] not in {seen["alias"] for seen in metrics}:
			metrics.append(metric)
	return metrics


def _require_type(fn: str, field: str, fieldtype: str) -> None:
	if fn in ("sum", "avg") and fieldtype not in NUMERIC_TYPES:
		raise ValueError(f"Cannot {fn} {field}: it is a {fieldtype} field, not a number.")
	if fn in ("min", "max") and fieldtype not in ORDERED_TYPES:
		raise ValueError(f"Cannot take the {fn} of {field}: it is a {fieldtype} field.")


def _resolve_groups(requested: Any, fields: _FieldCheck) -> list[dict]:
	if not requested:
		return []
	if isinstance(requested, str):
		requested = [requested]
	if not isinstance(requested, list) or len(requested) > MAX_GROUP_FIELDS:
		raise ValueError(f"group_by must be a list of up to {MAX_GROUP_FIELDS} fieldnames.")

	groups: list[dict] = []
	for raw in requested:
		if not isinstance(raw, str):
			raise ValueError("group_by entries are fieldnames, like 'project' or 'posting_date:month'.")
		field, _, period = raw.strip().partition(":")
		period = period.strip().lower() or None
		fieldtype = fields.fieldtype(field, "group")
		if fieldtype not in GROUP_TYPES:
			raise ValueError(f"Cannot group by {field}: it is a {fieldtype} field.")
		if period and (period not in PERIODS or fieldtype not in DATE_TYPES):
			raise ValueError(f"A period ({', '.join(PERIODS)}) only goes on a Date or Datetime field.")
		if fieldtype == "Datetime" and not period:
			period = "day"
		key = f"{field.strip()}:{period}" if period else field.strip()
		if key not in {group["key"] for group in groups}:
			groups.append({"field": field.strip(), "period": period, "key": key})
	return groups


def _currency_fields(meta: Any, metrics: list[dict], groups: list[dict], fields: _FieldCheck) -> list[str]:
	"""The currency columns a money total has to be split by, when they differ."""
	grouped = {group["field"] for group in groups if not group["period"]}
	found: list[str] = []
	for metric in metrics:
		if metric["fieldtype"] != "Currency":
			continue
		options = meta.get_field(metric["field"]).options
		if options and fields.readable(options) and options not in grouped and options not in found:
			found.append(options)
	return found


def _grouped_rows(doctype: str, filters: dict, key_fields: list[str], metrics: list[dict]) -> list[dict]:
	"""Let SQL do the grouping, one raw group per distinct key."""
	query_fields: list[Any] = [*key_fields, {"COUNT": "*", "as": COUNT_ALIAS}]
	for part, column in _sql_parts(metrics).items():
		fn, field = part
		query_fields.append({fn.upper(): field, "as": column})

	rows = frappe.get_list(
		doctype,
		filters=filters,
		fields=query_fields,
		group_by=", ".join(key_fields) or None,
		order_by=", ".join(f"{field} asc" for field in key_fields) or None,
		limit=RAW_GROUP_CAP + 1,
	)
	if len(rows) > RAW_GROUP_CAP:
		raise ValueError(
			f"Grouping {doctype} that way gives more than {RAW_GROUP_CAP} groups. "
			"Group by a period (like 'posting_date:month') or add filters."
		)
	parts = _sql_parts(metrics)
	return [_raw_from_sql(row, key_fields, parts) for row in rows]


def _sql_parts(metrics: list[dict]) -> dict[tuple[str, str], str]:
	"""The SQL aggregates the metrics need. An average is a sum and a count, so it rolls up."""
	parts: dict[tuple[str, str], str] = {}
	for metric in metrics:
		if not metric["field"]:
			continue
		needs = (
			[("sum", metric["field"]), ("count", metric["field"])]
			if metric["fn"] == "avg"
			else [(metric["fn"], metric["field"])]
		)
		for part in needs:
			parts.setdefault(part, f"_{part[0]}_{part[1]}")
	return parts


def _raw_from_sql(row: dict, key_fields: list[str], parts: dict[tuple[str, str], str]) -> dict:
	return {
		"key": {field: row.get(field) for field in key_fields},
		"rows": cint(row.get(COUNT_ALIAS)),
		"parts": {part: _number(row.get(column)) for part, column in parts.items()},
	}


def _number(value: Any) -> Any:
	"""SQL hands a SUM back as Decimal; the model gets a plain int or float."""
	if isinstance(value, Decimal):
		return int(value) if value == value.to_integral_value() else float(value)
	return value


def _checked_rows(
	doctype: str, filters: dict, key_fields: list[str], metrics: list[dict]
) -> tuple[list[dict], int]:
	"""List the rows, keep the ones this user may read, one raw group per row."""
	value_fields = _unique(key_fields + [metric["field"] for metric in metrics if metric["field"]])
	rows = frappe.get_list(
		doctype,
		filters=filters,
		fields=_unique(["name", *value_fields]),
		order_by="name asc",
		limit=ROW_CHECK_CAP + 1,
	)
	if len(rows) > ROW_CHECK_CAP:
		raise ValueError(
			f"More than {ROW_CHECK_CAP} {doctype} rows match, and each one has to be checked "
			"against your permissions before it counts. Add filters, like a date range, and ask again."
		)

	permitted = [row for row in rows if may_read(doctype, row.get("name"))]
	parts = _sql_parts(metrics)
	raw = []
	for row in permitted:
		values = {}
		for fn, field in parts:
			value = _number(row.get(field))
			values[(fn, field)] = (0 if value is None else 1) if fn == "count" else value
		raw.append({"key": {field: row.get(field) for field in key_fields}, "rows": 1, "parts": values})
	return raw, len(rows) - len(permitted)


def _currency_split(
	raw: list[dict], currency_fields: list[str], groups: list[dict]
) -> tuple[list[str], list[str]]:
	"""Keep a currency column in the key only when the rows really mix currencies."""
	split: list[str] = []
	notes: list[str] = []
	for field in currency_fields:
		seen = {entry["key"].get(field) for entry in raw}
		if len(seen) > 1:
			split.append(field)
			notes.append(
				f"The rows are in more than one currency, so money totals are split by {field}. "
				"For one total in the company currency, use the matching base_ field if the doctype has one."
			)
	return split, notes


class _Bucket:
	"""The running totals for one group."""

	def __init__(self):
		self.rows = 0
		self.parts: dict[tuple[str, str], Any] = {}

	def add(self, entry: dict) -> None:
		self.rows += entry["rows"]
		for (fn, field), value in entry["parts"].items():
			if value is None:
				continue
			current = self.parts.get((fn, field))
			if current is None:
				self.parts[(fn, field)] = value
			elif fn in ("sum", "count"):
				self.parts[(fn, field)] = current + value
			elif fn == "min":
				self.parts[(fn, field)] = min(current, value)
			else:
				self.parts[(fn, field)] = max(current, value)

	def values(self, metrics: list[dict]) -> dict:
		out: dict[str, Any] = {}
		for metric in metrics:
			fn, field = metric["fn"], metric["field"]
			if not field:
				out[metric["alias"]] = self.rows
			elif fn == "avg":
				total, count = self.parts.get(("sum", field)), cint(self.parts.get(("count", field)))
				out[metric["alias"]] = flt(flt(total) / count, 6) if count else None
			elif fn in ("sum", "count"):
				out[metric["alias"]] = self.parts.get((fn, field)) or 0
			else:
				out[metric["alias"]] = self.parts.get((fn, field))
		return out


def _roll_up(
	raw: list[dict], groups: list[dict], split: list[str], metrics: list[dict]
) -> dict[tuple, _Bucket]:
	buckets: dict[tuple, _Bucket] = {}
	for entry in raw:
		key = tuple(_group_value(entry["key"].get(group["field"]), group) for group in groups)
		key += tuple(entry["key"].get(field) for field in split)
		buckets.setdefault(key, _Bucket()).add(entry)
	return buckets


def _group_value(value: Any, group: dict) -> Any:
	if group["field"] == "docstatus":
		return status_word(value)
	period = group["period"]
	if not period or value in (None, ""):
		return value
	day = getdate(value) if isinstance(value, (str, datetime, date)) else value
	if period == "day":
		return day.isoformat()
	if period == "month":
		return f"{day.year}-{day.month:02d}"
	if period == "quarter":
		return f"{day.year}-Q{(day.month - 1) // 3 + 1}"
	return str(day.year)


def _chart(
	spec: dict | None, rows: list[dict], limit: int, keys: list[str], metrics: list[dict], notes: list[str]
) -> dict | None:
	"""The chart asked for, drawn from the groups returned. A reason in notes when it cannot be."""
	if not spec:
		return None
	chart, reason = build_chart(spec, rows[:limit], keys, metrics)
	if not chart:
		notes.append(reason)
		return None
	if len(rows) > limit:
		chart["note"] = f"The chart shows the first {limit} of {len(rows)} groups."
	return chart


def _group_row(
	key: tuple, bucket: _Bucket, groups: list[dict], split: list[str], metrics: list[dict]
) -> dict:
	names = [group["key"] for group in groups] + split
	return dict(zip(names, key, strict=True)) | bucket.values(metrics)


def _order(
	rows: list[dict], order_by: Any, groups: list[dict], split: list[str], metrics: list[dict]
) -> list[dict]:
	"""Sort the groups. Empty values go last either way, so the top N is never a run of blanks.

	By default a period group reads in date order and anything else reads largest first.
	"""
	columns = [metric["alias"] for metric in metrics] + [group["key"] for group in groups] + split
	periods = [group["key"] for group in groups if group["period"]]
	if not order_by:
		column, descending = (periods[0], False) if periods else (columns[0], True)
	else:
		if not isinstance(order_by, str):
			raise ValueError("order_by must be a string like 'sum_amount desc'.")
		parts = order_by.split()
		column = parts[0] if parts else ""
		direction = parts[1].lower() if len(parts) > 1 else "desc"
		if column not in columns or direction not in ("asc", "desc") or len(parts) > 2:
			raise ValueError(f"order_by must be one of {', '.join(columns)}, then asc or desc.")
		descending = direction == "desc"

	present = [row for row in rows if row.get(column) is not None]
	empty = [row for row in rows if row.get(column) is None]
	present.sort(key=lambda row: _sort_key(row[column]), reverse=descending)
	return present + empty


def _sort_key(value: Any) -> tuple:
	if isinstance(value, (int, float)) and not isinstance(value, bool):
		return (0, value, "")
	return (1, 0, str(value))


def _unique(values: list) -> list:
	return list(dict.fromkeys(value for value in values if value))


TOOLS = [
	{
		"tool_name": "aggregate_documents",
		"handler_path": "frappe_agents.tools.aggregate_tools.aggregate_documents",
		"capability": CAPABILITY_READ,
		"description": (
			"Count, sum, average, minimum or maximum over one doctype, optionally grouped by up to "
			"two fields. Use this for every 'how many' and 'how much' question instead of adding up "
			"rows from search_documents: a search returns at most one page, so its sum is wrong "
			"whenever there are more rows. Group a date by period with 'posting_date:month' "
			"(day, month, quarter, year). Only rows the user may read are counted. Fields the user "
			"may not read, or sees masked, are refused. Cancelled documents are left out unless you "
			"pass include_cancelled. Money in more than one currency comes back split by currency; "
			"read 'notes'. 'overall' is the total across every group, not only the ones returned. "
			"Pass 'chart' to show the groups as a chart under your answer; the chart is drawn from "
			"these totals, so do not repeat every number of it in your text."
		),
		"args_schema": {
			"type": "object",
			"properties": {
				"doctype": {"type": "string", "description": "DocType to add up, e.g. 'Sales Invoice'."},
				"metrics": {
					"type": "array",
					"items": {
						"type": "object",
						"properties": {
							"fn": {"type": "string", "enum": list(FUNCTIONS)},
							"field": {
								"type": "string",
								"description": "Field to add up. Leave out for a count of documents.",
							},
						},
						"required": ["fn"],
					},
					"description": (
						f"Up to {MAX_METRICS}, e.g. [{{'fn': 'sum', 'field': 'grand_total'}}, {{'fn': 'count'}}]. "
						"Default: a count of documents."
					),
				},
				"group_by": {
					"type": "array",
					"items": {"type": "string"},
					"description": (
						f"Up to {MAX_GROUP_FIELDS} fieldnames, e.g. ['customer'] or ['posting_date:month']. "
						"Leave out for one total."
					),
				},
				"filters": {
					"type": "object",
					"description": (
						'Field filters, e.g. {"status": "Paid"} or {"posting_date": ["between", ["2026-01-01", "2026-03-31"]]}.'
					),
				},
				"order_by": {
					"type": "string",
					"description": "A metric or group name and a direction, e.g. 'sum_grand_total desc'. Default: the first metric, highest first.",
				},
				"limit": {
					"type": "integer",
					"description": f"Groups to return, 1 to {MAX_GROUPS}. Default {DEFAULT_LIMIT}.",
				},
				"include_cancelled": {
					"type": "boolean",
					"description": "Include cancelled documents. Default false.",
				},
				"chart": {
					"type": "object",
					"properties": {
						"type": {"type": "string", "enum": list(CHART_TYPES)},
						"title": {
							"type": "string",
							"description": f"Plain text, at most {MAX_TITLE} characters.",
						},
					},
					"required": ["type"],
					"description": (
						"Show the groups as a chart. Needs group_by. 'line' suits a date period, 'bar' a "
						"comparison, 'pie' or 'donut' the parts of one total (one grouping, one metric)."
					),
				},
			},
			"required": ["doctype"],
		},
	},
]
