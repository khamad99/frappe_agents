# Copyright (c) 2026, Malik AlZubaidi and contributors
# For license information, please see LICENSE

"""Totals and grouping: the numbers are right, and they are only ever the user's numbers.

A total is harder to keep honest than a list. A list can mark a cell it may not
show; a sum cannot mark the row it should have left out. So every test here is
either "the arithmetic is right" or "a row or a field the user may not see did
not reach the number".

The cast is local to this module. `FA Test Payment` has what a total needs and
the shared fixtures do not: a date, two currencies, a masked amount and an
amount above permlevel 0. `LISTER_USER` may read a payment and may not report on
one, which is the Report View gate.
"""

from contextlib import contextmanager
from unittest import TestCase
from unittest.mock import patch

import frappe

from frappe_agents.tests.fixtures import (
	OPEN_USER,
	ORDER_DT,
	PERMLEVEL_ROLE,
	PROJECT_ALPHA,
	PROJECT_BETA,
	PROJECT_DT,
	READER_ROLE,
	RESTRICTED_USER,
	TICKET_DT,
	AgentTestCase,
	_make_doctype,
	_make_user,
	call_tool,
	make_matrix_agent,
	rule,
	tool_calls_for,
)
from frappe_agents.tools import aggregate_tools
from frappe_agents.tools.charts import NONE_LABEL, build_chart

PAYMENT_DT = "FA Test Payment"
LISTER_ROLE = "FA Test Lister"
LISTER_USER = "fa-lister@example.com"
HIDDEN_PARTY = "FA Hidden Party"
SUM_METRIC = {"alias": "sum_amount", "label": "Sum of Amount", "numeric": True}

# title, project, date, currency, amount, base_amount, party
PAYMENTS = (
	("FA Pay 1", PROJECT_ALPHA, "2026-01-15", "AED", 100, 100, "FA Party"),
	("FA Pay 2", PROJECT_ALPHA, "2026-02-10", "AED", 250, 250, "FA Party"),
	("FA Pay 3", PROJECT_ALPHA, "2026-02-20", "USD", 40, 147, HIDDEN_PARTY),
	("FA Pay 4", PROJECT_BETA, "2026-04-01", "AED", 1000, 1000, "FA Party"),
)


def deny_hidden_party(doc, ptype=None, user=None, debug=False):
	"""A per-row has_permission hook, the kind get_list never runs."""
	return doc.get("party") != HIDDEN_PARTY


@contextmanager
def row_hook_on_payments():
	"""Register `deny_hidden_party` the way an app's hooks.py would."""
	real = frappe.get_hooks

	def hooks(hook=None, *args, **kwargs):
		value = real(hook, *args, **kwargs)
		if hook != "has_permission":
			return value
		value = frappe._dict(value or {})
		value[PAYMENT_DT] = [f"{__name__}.deny_hidden_party"]
		return value

	with patch("frappe.get_hooks", hooks):
		yield


class AggregateTestCase(AgentTestCase):
	def setUp(self) -> None:
		super().setUp()
		_ensure_payments()
		self.agent = make_matrix_agent(
			[rule(PAYMENT_DT, can_read=1), rule(ORDER_DT, can_read=1)], autonomy="Suggest"
		)

	def aggregate(self, user: str, args: dict, agent: str | None = None) -> tuple[dict, object]:
		return call_tool(
			user,
			"aggregate_documents",
			dict(args, doctype=args.get("doctype", PAYMENT_DT)),
			agent or self.agent.name,
		)

	def result(self, user: str, args: dict) -> dict:
		payload, _ = self.aggregate(user, args)
		self.assertTrue(payload["ok"], payload["error"])
		return payload["result"]

	def refused(self, user: str, args: dict, agent: str | None = None) -> str:
		payload, _ = self.aggregate(user, args, agent)
		self.assertFalse(payload["ok"], payload.get("result"))
		return payload["error"]

	def by(self, result: dict, key: str) -> dict:
		return {row[key]: row for row in result["groups"]}


class TestTheArithmetic(AggregateTestCase):
	def test_a_count_with_no_group_is_one_overall_number(self):
		result = self.result(OPEN_USER, {})

		self.assertEqual(result["overall"], [{"count": 4}])
		self.assertEqual(result["rows_counted"], 4)

	def test_sum_avg_min_and_max(self):
		result = self.result(
			OPEN_USER,
			{
				"metrics": [
					{"fn": "sum", "field": "base_amount"},
					{"fn": "avg", "field": "base_amount"},
					{"fn": "min", "field": "posting_date"},
					{"fn": "max", "field": "base_amount"},
				]
			},
		)

		overall = result["overall"][0]
		self.assertEqual(overall["sum_base_amount"], 1497)
		self.assertAlmostEqual(overall["avg_base_amount"], 374.25)
		self.assertEqual(str(overall["min_posting_date"]), "2026-01-15")
		self.assertEqual(overall["max_base_amount"], 1000)

	def test_group_by_a_link_field(self):
		result = self.result(
			OPEN_USER, {"group_by": ["project"], "metrics": [{"fn": "sum", "field": "base_amount"}]}
		)

		groups = self.by(result, "project")
		self.assertEqual(groups[PROJECT_ALPHA]["sum_base_amount"], 497)
		self.assertEqual(groups[PROJECT_BETA]["sum_base_amount"], 1000)
		# Largest first by default.
		self.assertEqual(result["groups"][0]["project"], PROJECT_BETA)

	def test_dates_roll_up_by_month_and_quarter_in_date_order(self):
		months = self.result(OPEN_USER, {"group_by": ["posting_date:month"]})
		self.assertEqual(
			[(row["posting_date:month"], row["count"]) for row in months["groups"]],
			[("2026-01", 1), ("2026-02", 2), ("2026-04", 1)],
		)

		quarters = self.result(OPEN_USER, {"group_by": ["posting_date:quarter"]})
		self.assertEqual(
			[(row["posting_date:quarter"], row["count"]) for row in quarters["groups"]],
			[("2026-Q1", 3), ("2026-Q2", 1)],
		)

	def test_money_in_two_currencies_is_split_not_added(self):
		result = self.result(OPEN_USER, {"metrics": [{"fn": "sum", "field": "amount"}]})

		overall = {row["currency"]: row["sum_amount"] for row in result["overall"]}
		self.assertEqual(overall, {"AED": 1350, "USD": 40})
		self.assertIn("currency", result["group_by"])
		self.assertTrue(result["notes"])

	def test_one_currency_is_not_split(self):
		result = self.result(
			OPEN_USER,
			{"metrics": [{"fn": "sum", "field": "amount"}], "filters": {"currency": "AED"}},
		)

		self.assertEqual(result["overall"], [{"sum_amount": 1350}])
		self.assertEqual(result["notes"], [])

	def test_cancelled_documents_are_left_out_unless_asked(self):
		without = self.result(OPEN_USER, {"doctype": ORDER_DT})
		with_cancelled = self.result(OPEN_USER, {"doctype": ORDER_DT, "include_cancelled": 1})

		self.assertEqual(with_cancelled["rows_counted"] - without["rows_counted"], 1)

	def test_a_rule_cap_narrows_the_groups_returned(self):
		agent = make_matrix_agent(
			[rule(PAYMENT_DT, can_read=1, max_rows_per_call=1)], autonomy="Suggest", name="FA Capped Agent"
		)

		payload, _ = self.aggregate(OPEN_USER, {"group_by": ["project"]}, agent.name)

		result = payload["result"]
		self.assertEqual(len(result["groups"]), 1)
		self.assertEqual(result["group_count"], 2)
		self.assertTrue(result["truncated"])

	def test_too_many_groups_is_refused(self):
		with patch.object(aggregate_tools, "RAW_GROUP_CAP", 1):
			error = self.refused(OPEN_USER, {"group_by": ["project"]})

		self.assertIn("more than 1 groups", error)

	def test_the_audit_row_names_no_document(self):
		payload, run = self.aggregate(OPEN_USER, {"group_by": ["project"]})

		self.assertTrue(payload["ok"], payload["error"])
		call = tool_calls_for(run.name)[-1]
		self.assertEqual(call["docs_touched"], f"{PAYMENT_DT}: aggregate of 4 row(s)")
		self.assertEqual(call["result_summary"], "2 group(s) over 4 row(s)")


class TestOnlyTheUsersNumbers(AggregateTestCase):
	def test_a_user_permission_narrows_the_total(self):
		result = self.result(RESTRICTED_USER, {"metrics": [{"fn": "sum", "field": "base_amount"}]})

		self.assertEqual(result["overall"], [{"sum_base_amount": 497}])
		self.assertEqual(result["rows_counted"], 3)

	def test_a_row_hook_leaves_out_rows_the_user_may_not_open(self):
		with row_hook_on_payments():
			result = self.result(OPEN_USER, {"metrics": [{"fn": "sum", "field": "base_amount"}]})

		self.assertTrue(result["checked_row_by_row"])
		self.assertEqual(result["filtered_by_permission"], 1)
		self.assertEqual(result["overall"], [{"sum_base_amount": 1350}])

	def test_without_a_row_hook_sql_does_the_grouping(self):
		result = self.result(OPEN_USER, {})

		self.assertFalse(result["checked_row_by_row"])
		self.assertEqual(result["filtered_by_permission"], 0)

	def test_past_the_row_check_cap_the_call_is_refused(self):
		with row_hook_on_payments(), patch.object(aggregate_tools, "ROW_CHECK_CAP", 2):
			error = self.refused(OPEN_USER, {})

		self.assertIn("Add filters", error)

	def test_a_field_above_the_users_permlevel_is_refused_everywhere(self):
		uses = (
			{"metrics": [{"fn": "sum", "field": "internal_margin"}]},
			{"group_by": ["internal_margin"]},
			{"filters": {"internal_margin": 5}},
		)
		for args in uses:
			with self.subTest(args=args):
				self.assertIn("not allowed", self.refused(RESTRICTED_USER, args))

		self.result(OPEN_USER, {"metrics": [{"fn": "sum", "field": "internal_margin"}]})

	def test_a_masked_field_is_refused_everywhere(self):
		uses = (
			{"metrics": [{"fn": "sum", "field": "commission"}]},
			{"metrics": [{"fn": "max", "field": "commission"}]},
			{"group_by": ["commission"]},
			{"filters": {"commission": 5}},
		)
		for args in uses:
			with self.subTest(args=args):
				self.assertIn("masked", self.refused(OPEN_USER, args))

	def test_reading_without_report_permission_is_refused(self):
		error = self.refused(LISTER_USER, {})

		self.assertIn("not allowed to report", error)

	def test_a_doctype_the_rule_does_not_grant_is_refused(self):
		agent = make_matrix_agent([rule(TICKET_DT, can_read=1)], autonomy="Suggest", name="FA Ticket Agent")

		self.refused(OPEN_USER, {}, agent.name)


class TestBadInput(AggregateTestCase):
	def test_sum_needs_a_number(self):
		self.assertIn(
			"not a number", self.refused(OPEN_USER, {"metrics": [{"fn": "sum", "field": "posting_date"}]})
		)

	def test_long_text_cannot_be_a_group(self):
		self.assertIn("Cannot group", self.refused(OPEN_USER, {"group_by": ["remarks"]}))

	def test_a_period_needs_a_date(self):
		self.assertIn("period", self.refused(OPEN_USER, {"group_by": ["project:month"]}))

	def test_order_by_must_name_a_column(self):
		self.assertIn("order_by", self.refused(OPEN_USER, {"order_by": "amount desc"}))


class TestCharts(AggregateTestCase):
	def chart(self, args: dict) -> dict:
		return self.result(OPEN_USER, args)

	def test_no_chart_unless_asked(self):
		self.assertIsNone(self.chart({"group_by": ["project"]})["chart"])

	def test_one_grouping_is_one_series_per_metric(self):
		result = self.chart(
			{
				"group_by": ["posting_date:month"],
				"metrics": [{"fn": "sum", "field": "base_amount"}, {"fn": "count"}],
				"chart": {"type": "line", "title": "Payments by month"},
			}
		)

		chart = result["chart"]
		self.assertEqual(chart["type"], "line")
		self.assertEqual(chart["title"], "Payments by month")
		self.assertEqual(chart["labels"], ["2026-01", "2026-02", "2026-04"])
		self.assertEqual(
			chart["datasets"],
			[
				{"name": "Sum of Base Amount", "values": [100, 397, 1000]},
				{"name": "Count", "values": [1, 2, 1]},
			],
		)

	def test_two_groupings_are_one_series_per_value_of_the_second(self):
		result = self.chart(
			{
				"group_by": ["posting_date:quarter", "project"],
				"order_by": "posting_date:quarter asc",
				"chart": {"type": "bar"},
			}
		)

		chart = result["chart"]
		self.assertEqual(chart["labels"], ["2026-Q1", "2026-Q2"])
		series = {dataset["name"]: dataset["values"] for dataset in chart["datasets"]}
		self.assertEqual(series, {PROJECT_ALPHA: [3, 0], PROJECT_BETA: [0, 1]})

	def test_a_currency_split_becomes_the_series(self):
		result = self.chart(
			{"group_by": ["project"], "metrics": [{"fn": "sum", "field": "amount"}], "chart": {"type": "bar"}}
		)

		series = {dataset["name"]: dataset["values"] for dataset in result["chart"]["datasets"]}
		self.assertEqual(set(series), {"AED", "USD"})

	def test_a_chart_that_cannot_be_drawn_leaves_the_totals_and_says_why(self):
		result = self.chart(
			{
				"group_by": ["project"],
				"metrics": [{"fn": "sum", "field": "base_amount"}, {"fn": "count"}],
				"chart": {"type": "pie"},
			}
		)

		self.assertIsNone(result["chart"])
		self.assertEqual(len(result["groups"]), 2)
		self.assertTrue(any("one metric only" in note for note in result["notes"]))

	def test_a_chart_needs_group_by(self):
		result = self.chart({"chart": {"type": "bar"}})

		self.assertIsNone(result["chart"])
		self.assertTrue(any("needs group_by" in note for note in result["notes"]))

	def test_a_chart_shows_only_the_groups_returned_and_says_so(self):
		result = self.chart({"group_by": ["posting_date:month"], "limit": 2, "chart": {"type": "bar"}})

		self.assertEqual(result["chart"]["labels"], ["2026-01", "2026-02"])
		self.assertEqual(result["chart"]["note"], "The chart shows the first 2 of 3 groups.")

	def test_a_bad_type_or_a_long_title_is_refused(self):
		self.assertIn(
			"chart type", self.refused(OPEN_USER, {"group_by": ["project"], "chart": {"type": "radar"}})
		)
		self.assertIn(
			"at most",
			self.refused(OPEN_USER, {"group_by": ["project"], "chart": {"type": "bar", "title": "x" * 81}}),
		)


class TestChartShapes(TestCase):
	"""The builder alone, for the shapes the fixture data cannot make. No database."""

	def test_a_pie_refuses_values_below_zero(self):
		rows = [{"party": "A", "sum_amount": 5}, {"party": "B", "sum_amount": -2}]

		chart, reason = build_chart({"type": "pie", "title": ""}, rows, ["party"], [SUM_METRIC])

		self.assertIsNone(chart)
		self.assertIn("below zero", reason)

	def test_too_many_series_is_refused(self):
		rows = [{"month": "2026-01", "party": f"P{index}", "sum_amount": index} for index in range(7)]

		chart, reason = build_chart({"type": "bar", "title": ""}, rows, ["month", "party"], [SUM_METRIC])

		self.assertIsNone(chart)
		self.assertIn("more than 6", reason)

	def test_an_empty_group_value_has_a_label(self):
		rows = [{"party": None, "sum_amount": 5}, {"party": "", "sum_amount": 1}]

		chart, _ = build_chart({"type": "bar", "title": ""}, rows, ["party"], [SUM_METRIC])

		self.assertEqual(chart["labels"], [NONE_LABEL, NONE_LABEL])

	def test_a_date_metric_is_not_drawn(self):
		date_metric = {"alias": "max_posting_date", "label": "Maximum of Posting Date", "numeric": False}
		rows = [{"party": "A", "max_posting_date": "2026-01-01"}]

		chart, reason = build_chart({"type": "bar", "title": ""}, rows, ["party"], [date_metric])

		self.assertIsNone(chart)
		self.assertIn("none of the metrics is a number", reason)


def _ensure_payments() -> None:
	_ensure_lister_role()
	for code in ("AED", "USD"):
		if not frappe.db.exists("Currency", code):
			frappe.get_doc({"doctype": "Currency", "currency_name": code, "enabled": 1}).insert(
				ignore_permissions=True
			)

	_make_doctype(
		PAYMENT_DT,
		"payment_title",
		fields=[
			{"fieldname": "payment_title", "fieldtype": "Data", "label": "Title", "reqd": 1, "unique": 1},
			{"fieldname": "project", "fieldtype": "Link", "label": "Project", "options": PROJECT_DT},
			{"fieldname": "party", "fieldtype": "Data", "label": "Party"},
			{"fieldname": "posting_date", "fieldtype": "Date", "label": "Posting Date"},
			{"fieldname": "currency", "fieldtype": "Link", "label": "Currency", "options": "Currency"},
			{"fieldname": "amount", "fieldtype": "Currency", "label": "Amount", "options": "currency"},
			{"fieldname": "base_amount", "fieldtype": "Currency", "label": "Base Amount"},
			{"fieldname": "commission", "fieldtype": "Currency", "label": "Commission", "mask": 1},
			{
				"fieldname": "internal_margin",
				"fieldtype": "Currency",
				"label": "Internal Margin",
				"permlevel": 1,
			},
			{"fieldname": "remarks", "fieldtype": "Small Text", "label": "Remarks"},
		],
		permissions=[
			{"role": "System Manager", "read": 1, "write": 1, "create": 1, "delete": 1},
			{"role": READER_ROLE, "read": 1, "report": 1},
			{"role": PERMLEVEL_ROLE, "read": 1, "report": 1},
			{"role": PERMLEVEL_ROLE, "read": 1, "permlevel": 1},
			{"role": LISTER_ROLE, "read": 1, "report": 0},
		],
	)

	for title, project, posting_date, currency, amount, base_amount, party in PAYMENTS:
		if frappe.db.exists(PAYMENT_DT, title):
			continue
		frappe.get_doc(
			{
				"doctype": PAYMENT_DT,
				"payment_title": title,
				"project": project,
				"posting_date": posting_date,
				"currency": currency,
				"amount": amount,
				"base_amount": base_amount,
				"party": party,
				"commission": 7,
				"internal_margin": 3,
			}
		).insert(ignore_permissions=True)

	_make_user(LISTER_USER, "Lister", ["Agent User", LISTER_ROLE])


def _ensure_lister_role() -> None:
	if not frappe.db.exists("Role", LISTER_ROLE):
		frappe.get_doc({"doctype": "Role", "role_name": LISTER_ROLE, "desk_access": 1}).insert(
			ignore_permissions=True
		)
