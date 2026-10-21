# Copyright (c) 2026, Malik AlZubaidi and contributors
# For license information, please see LICENSE

"""The versions slice answers for each field, not only for the document.

A Version row stores the raw value from both sides of an edit. Reading the
document is not permission to read every field in it: a field above the user's
permlevel comes back restricted, and a field the user sees masked comes back
masked — the same answers the field projection gives.
"""

import json

import frappe

from frappe_agents.context.slices import MASKED, RESTRICTED, get_slice
from frappe_agents.tests.fixtures import (
	DRAFT_USER,
	IBAN_FIELD,
	OPEN_USER,
	ORDER_DT,
	ORDER_LIVE,
	RESTRICTED_USER,
	TICKET_ALPHA,
	TICKET_DT,
	VENDOR_ACME,
	VENDOR_DT,
	AgentTestCase,
	as_user,
)

OLD_IBAN = "AE07 0331 2345 6789 0123 456"
NEW_IBAN = "AE99 9999 9999 9999 9999 999"


class TestTheVersionsSlice(AgentTestCase):
	def track(self, doctype: str) -> None:
		"""Turn on track_changes for this test only.

		The meta is cached, so the cache is cleared again after the rollback, or the
		flag would outlive this test in every later one.
		"""
		frappe.db.set_value("DocType", doctype, "track_changes", 1)
		frappe.clear_cache(doctype=doctype)
		self.addCleanup(frappe.clear_cache, doctype=doctype)
		self.addCleanup(frappe.db.rollback)

	def record_edit(self, doctype: str, name: str, fieldname: str, old: str, new: str) -> None:
		self.track(doctype)
		version = frappe.get_doc(
			{
				"doctype": "Version",
				"ref_doctype": doctype,
				"docname": name,
				"data": json.dumps({"changed": [[fieldname, old, new]]}),
			}
		)
		version.insert(ignore_permissions=True)

	def change_to(self, user: str, doctype: str, name: str, fieldname: str) -> dict:
		with as_user(user):
			payload = get_slice(doctype, name, "versions")
		for version in payload["versions"]:
			for change in version.get("changes", []):
				if change["field"] == fieldname:
					return change
		self.fail(f"no change to {fieldname} in the versions slice")

	def test_a_masked_field_comes_back_masked(self):
		self.record_edit(VENDOR_DT, VENDOR_ACME, IBAN_FIELD, OLD_IBAN, NEW_IBAN)

		change = self.change_to(DRAFT_USER, VENDOR_DT, VENDOR_ACME, IBAN_FIELD)

		self.assertEqual((change["from"], change["to"]), (MASKED, MASKED))

	def test_a_field_above_the_users_permlevel_comes_back_restricted(self):
		self.record_edit(TICKET_DT, TICKET_ALPHA, "secret_note", "old secret", "new secret")

		change = self.change_to(RESTRICTED_USER, TICKET_DT, TICKET_ALPHA, "secret_note")

		self.assertEqual((change["from"], change["to"]), (RESTRICTED, RESTRICTED))

	def test_a_user_who_may_read_the_field_sees_the_change(self):
		"""The control. Without it, a slice that withheld everything would pass the two above."""
		self.record_edit(TICKET_DT, TICKET_ALPHA, "secret_note", "old secret", "new secret")

		change = self.change_to(OPEN_USER, TICKET_DT, TICKET_ALPHA, "secret_note")

		self.assertEqual((change["from"], change["to"]), ("old secret", "new secret"))

	def test_a_child_table_above_the_users_permlevel_hides_its_rows(self):
		"""The child fields sit at permlevel 0. The table that holds them does not."""
		self.track(ORDER_DT)
		frappe.db.set_value("DocField", {"parent": ORDER_DT, "fieldname": "items"}, "permlevel", 1)
		frappe.clear_cache(doctype=ORDER_DT)
		row = frappe.get_doc(ORDER_DT, ORDER_LIVE).items[0]
		frappe.get_doc(
			{
				"doctype": "Version",
				"ref_doctype": ORDER_DT,
				"docname": ORDER_LIVE,
				"data": json.dumps({"row_changed": [["items", 0, row.name, [["qty", 2, 5]]]]}),
			}
		).insert(ignore_permissions=True)

		change = self.change_to(DRAFT_USER, ORDER_DT, ORDER_LIVE, "items[0].qty")

		self.assertEqual((change["from"], change["to"]), (RESTRICTED, RESTRICTED))
