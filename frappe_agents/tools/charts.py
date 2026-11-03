"""Charts drawn from totals the database counted.

A chart shows numbers, and a number in a chart reads as a fact. So the model
never types one: it asks `aggregate_documents` for a chart, and the chart is
built here from the groups that call has just counted. There is no tool that
draws a chart from numbers the model supplies.

What comes back is data, not markup — labels, series and values. The chat
draws it with Frappe's own chart library, and escapes every label first,
because a label is whatever somebody typed into a customer or a project name.

A chart that cannot be drawn from the groups is not an error. The totals are
still right, so the call still succeeds, and the reason goes in `notes`.
"""

from typing import Any

from frappe.utils import cstr

CHART_TYPES = ("bar", "line", "pie", "donut")
ROUND_TYPES = ("pie", "donut")
MAX_TITLE = 80
MAX_SERIES = 6
NONE_LABEL = "(none)"


def resolve_chart(spec: Any) -> dict | None:
	"""Check the `chart` argument. None means no chart was asked for."""
	if not spec:
		return None
	if isinstance(spec, str):
		spec = {"type": spec}
	if not isinstance(spec, dict):
		raise ValueError("chart must be an object like {'type': 'bar', 'title': 'Sales by month'}.")

	chart_type = cstr(spec.get("type")).strip().lower()
	if chart_type not in CHART_TYPES:
		raise ValueError(f"chart type must be one of {', '.join(CHART_TYPES)}.")
	title = cstr(spec.get("title")).strip()
	if len(title) > MAX_TITLE:
		raise ValueError(f"chart title must be at most {MAX_TITLE} characters.")
	return {"type": chart_type, "title": title}


def build_chart(
	spec: dict, rows: list[dict], keys: list[str], metrics: list[dict]
) -> tuple[dict | None, str]:
	"""The chart for these groups, or None and the reason there is none.

	`rows` are the groups the call returns, already ordered; `keys` their group
	columns, currency split included; `metrics` the metric definitions, each with
	`alias`, `label` and `numeric`.
	"""
	numeric = [metric for metric in metrics if metric["numeric"]]
	if not keys:
		return None, "No chart: a chart needs group_by."
	if not numeric:
		return None, "No chart: none of the metrics is a number."
	if not rows:
		return None, "No chart: nothing matched."
	if len(keys) > 2:
		return None, "No chart: it can show at most two groupings."

	if spec["type"] in ROUND_TYPES:
		labels, datasets, reason = _round(spec["type"], rows, keys, numeric)
	elif len(keys) == 1:
		labels, datasets, reason = _by_metric(rows, keys[0], numeric)
	else:
		labels, datasets, reason = _by_series(rows, keys, numeric)
	if reason:
		return None, reason

	return {
		"type": spec["type"],
		"title": spec["title"],
		"labels": labels,
		"datasets": datasets,
	}, ""


def _by_metric(rows: list[dict], key: str, numeric: list[dict]) -> tuple[list, list, str]:
	"""One grouping: one bar or line per metric."""
	labels = [_label(row.get(key)) for row in rows]
	datasets = [
		{"name": metric["label"], "values": [_value(row.get(metric["alias"])) for row in rows]}
		for metric in numeric
	]
	return labels, datasets, ""


def _by_series(rows: list[dict], keys: list[str], numeric: list[dict]) -> tuple[list, list, str]:
	"""Two groupings: the first along the axis, one series per value of the second."""
	if len(numeric) > 1:
		return [], [], "No chart: with two groupings a chart can show one metric only."
	metric = numeric[0]
	labels = _unique(_label(row.get(keys[0])) for row in rows)
	series = _unique(_label(row.get(keys[1])) for row in rows)
	if len(series) > MAX_SERIES:
		return [], [], f"No chart: {keys[1]} has more than {MAX_SERIES} values to draw as series."

	values = {(_label(row.get(keys[0])), _label(row.get(keys[1]))): row.get(metric["alias"]) for row in rows}
	datasets = [
		{"name": name, "values": [_value(values.get((label, name))) for label in labels]} for name in series
	]
	return labels, datasets, ""


def _round(chart_type: str, rows: list[dict], keys: list[str], numeric: list[dict]) -> tuple[list, list, str]:
	"""A pie or a donut: one grouping, one metric, nothing below zero."""
	if len(keys) > 1:
		return [], [], f"No chart: a {chart_type} chart shows one grouping only. Use bar."
	if len(numeric) > 1:
		return [], [], f"No chart: a {chart_type} chart shows one metric only."
	values = [_value(row.get(numeric[0]["alias"])) for row in rows]
	if any(value < 0 for value in values):
		return [], [], f"No chart: a {chart_type} chart cannot show values below zero. Use bar."
	labels = [_label(row.get(keys[0])) for row in rows]
	return labels, [{"name": numeric[0]["label"], "values": values}], ""


def _label(value: Any) -> str:
	return NONE_LABEL if value in (None, "") else cstr(value)


def _value(value: Any) -> int | float:
	return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _unique(values: Any) -> list:
	return list(dict.fromkeys(values))
