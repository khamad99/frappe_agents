# Copyright (c) 2026, Malik AlZubaidi and contributors
# For license information, please see LICENSE

"""The workflow state of a document, in the generic read tools.

`status_word` says DRAFT, SUBMITTED or CANCELLED. On a doctype under an active
Workflow, that is not the stage people name: a submitted order can be "Approved"
or "Closed", a draft can wait at any of several approvals. Without the state the
model guesses the stage from the docstatus, and guesses wrong. So a document
under an active Workflow carries its `workflow_state` in the context, in search
rows and in links rows. A doctype with no active Workflow carries none, even
when the field a Workflow once added is still on it.
"""

import frappe
from frappe.model.workflow import apply_workflow

from frappe_agents.tests.fixtures import (
	DRAFT_USER,
	ORDER_DT,
	PROJECT_ALPHA,
	PROJECT_DT,
	WORKFLOW_ACTION,
	WORKFLOW_APPROVED_STATE,
	WORKFLOW_DRAFT_STATE,
	AgentTestCase,
	active_workflow,
	call_tool,
	make_order_draft,
)

STATE = "workflow_state"


class TestWorkflowStateInTheReadTools(AgentTestCase):
	def result(self, tool: str, args: dict) -> dict:
		payload, _run = call_tool(DRAFT_USER, tool, args)
		self.assertTrue(payload["ok"], payload["error"])
		return payload["result"]

	def context(self, name: str) -> dict:
		return self.result("get_document_context", {"doctype": ORDER_DT, "name": name})["doc"]

	def search_rows(self, name: str, **args) -> tuple[list, dict]:
		result = self.result("search_documents", {"doctype": ORDER_DT, "filters": {"name": name}, **args})
		return result["rows"], result

	def links_row(self, name: str) -> dict:
		"""The order's row in the project's links slice: the order points at the project."""
		result = self.result(
			"get_document_slice",
			{"doctype": PROJECT_DT, "name": PROJECT_ALPHA, "slice": "links", "direction": "down"},
		)
		for group in result["down"]:
			for doc in group["docs"]:
				if group["doctype"] == ORDER_DT and doc["name"] == name:
					return doc
		self.fail(f"{name} is not in the links slice of {PROJECT_ALPHA}")

	def test_the_context_of_a_workflow_document_gives_its_workflow_state(self):
		with active_workflow(ORDER_DT):
			order = make_order_draft()
			drafted = self.context(order.name)

			apply_workflow(order, WORKFLOW_ACTION)
			approved = self.context(order.name)

		self.assertEqual(drafted[STATE], WORKFLOW_DRAFT_STATE)
		self.assertEqual(drafted["status_word"], "DRAFT")
		self.assertEqual(approved[STATE], WORKFLOW_APPROVED_STATE)
		self.assertEqual(approved["status_word"], "SUBMITTED")

	def test_search_rows_give_the_workflow_state(self):
		with active_workflow(ORDER_DT):
			order = make_order_draft()
			default_rows, default_result = self.search_rows(order.name)
			chosen_rows, chosen_result = self.search_rows(order.name, fields=["name", "amount"])

		for rows, result in ((default_rows, default_result), (chosen_rows, chosen_result)):
			self.assertEqual([row[STATE] for row in rows], [WORKFLOW_DRAFT_STATE])
			self.assertIn(STATE, result["fields"])

	def test_a_links_row_gives_the_linked_documents_workflow_state(self):
		with active_workflow(ORDER_DT):
			order = make_order_draft()
			row = self.links_row(order.name)

		self.assertEqual(row[STATE], WORKFLOW_DRAFT_STATE)
		self.assertEqual(row["status_word"], "DRAFT")

	def test_the_fields_slice_already_gives_the_workflow_state(self):
		with active_workflow(ORDER_DT):
			order = make_order_draft()
			fields = self.result(
				"get_document_slice", {"doctype": ORDER_DT, "name": order.name, "slice": "fields"}
			)["fields"]

		self.assertEqual(fields[STATE], WORKFLOW_DRAFT_STATE)

	def test_no_workflow_state_without_an_active_workflow(self):
		"""The field a Workflow added stays on the doctype. The Workflow is what counts."""
		with active_workflow(ORDER_DT):
			pass
		self.assertTrue(frappe.get_meta(ORDER_DT).get_field(STATE))

		order = make_order_draft()
		rows, result = self.search_rows(order.name)

		self.assertNotIn(STATE, self.context(order.name))
		self.assertNotIn(STATE, rows[0])
		self.assertNotIn(STATE, result["fields"])
		self.assertNotIn(STATE, self.links_row(order.name))
