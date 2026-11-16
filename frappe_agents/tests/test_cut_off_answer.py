# Copyright (c) 2026, Malik AlZubaidi and contributors
# For license information, please see LICENSE

"""A turn the token budget ends before the model has said anything.

A thinking model spends its output budget on thinking first. At 4096 tokens on
the Anthropic wire, a hard question could use all of it there: the reply ended
on `max_tokens` with one thinking block, no text and no tool call, and the run
was stored as Completed with an empty answer. The person saw nothing at all.

Two things are pinned here. The Anthropic request asks for the same budget as
the Responses wire. And a turn that is cut off before its first word fails, in
words the person can act on, so the next turn replays what it already did.

Only the socket is stubbed. The loop, the executor and the rows are real.
"""

import json
from unittest.mock import patch

import frappe

from frappe_agents.runner.providers import ANTHROPIC_MAX_TOKENS, call_model, call_model_stream
from frappe_agents.runner.run import CUT_OFF_MESSAGE, execute_run
from frappe_agents.tests.fixtures import (
	AGENT,
	PROFILE,
	PROVIDER,
	RESTRICTED_USER,
	TICKET_DT,
	AgentTestCase,
	as_user,
	make_conversation,
	make_run,
	tool_calls_for,
)
from frappe_agents.tests.test_provider_streams import FakeStream, named_frame, sse

MESSAGES = [{"role": "user", "content": "Summarise every open ticket."}]

THINKING = "There are many tickets. I should group them by project first, then"
SIGNATURE = "EqQBCgIYAhgC"


def frame(kind: str, **fields) -> str:
	return named_frame(kind, {"type": kind, **fields})


def block(index: int, start: dict, *deltas: dict) -> list[str]:
	"""One content block: its start, its deltas and its stop."""
	return [
		frame("content_block_start", index=index, content_block=start),
		*(frame("content_block_delta", index=index, delta=delta) for delta in deltas),
		frame("content_block_stop", index=index),
	]


def thinking_block(index: int = 0) -> list[str]:
	return block(
		index,
		{"type": "thinking", "thinking": ""},
		{"type": "thinking_delta", "thinking": THINKING},
		{"type": "signature_delta", "signature": SIGNATURE},
	)


def text_block(text: str, index: int = 0) -> list[str]:
	return block(index, {"type": "text", "text": ""}, {"type": "text_delta", "text": text})


def tool_block(index: int = 0) -> list[str]:
	arguments = json.dumps({"doctype": TICKET_DT, "fields": ["name"]})
	return block(
		index,
		{"type": "tool_use", "id": "toolu_1", "name": "search_documents", "input": {}},
		{"type": "input_json_delta", "partial_json": arguments},
	)


def reply(*blocks: list[str], stop: str, tokens_out: int = 10) -> FakeStream:
	"""One streamed Anthropic reply that ends on `stop`."""
	frames = [frame("message_start", message={"model": "fa-test-model", "usage": {"input_tokens": 50}})]
	for one in blocks:
		frames.extend(one)
	frames.append(frame("message_delta", delta={"stop_reason": stop}, usage={"output_tokens": tokens_out}))
	frames.append(frame("message_stop"))
	return FakeStream([sse(*frames)])


def cut_off() -> FakeStream:
	"""The reply seen on test-erp: the whole budget spent on one thinking block."""
	return reply(thinking_block(), stop="max_tokens", tokens_out=ANTHROPIC_MAX_TOKENS)


def run_value(run_name: str, field: str):
	return frappe.db.get_value("Agent Run", run_name, field)


class AnthropicCase(AgentTestCase):
	"""A test whose provider speaks the Anthropic Messages API."""

	def setUp(self) -> None:
		super().setUp()
		frappe.db.set_value("LLM Provider", PROVIDER, "provider_type", "Anthropic", update_modified=False)
		frappe.clear_document_cache("LLM Provider", PROVIDER)
		self.addCleanup(frappe.clear_document_cache, "LLM Provider", PROVIDER)

	def execute(self, *replies: FakeStream, conversation: str | None = None, message: str = "Hello"):
		"""One run against stubbed sockets. Returns the run and the post mock."""
		with as_user(RESTRICTED_USER):
			run = make_run(
				effective_user=RESTRICTED_USER, agent=AGENT, conversation=conversation, message=message
			)
		with patch("frappe_agents.runner.providers.requests.post", side_effect=list(replies)) as post:
			execute_run(run.name)
		return run, post


class TestAnthropicBudget(AnthropicCase):
	def test_a_streamed_request_asks_for_the_chat_budget(self):
		"""The run path. 4096 is what let a thinking model use it all up."""
		with patch("frappe_agents.runner.providers.requests.post", return_value=cut_off()) as post:
			list(call_model_stream(PROFILE, MESSAGES))

		payload = post.call_args.kwargs["json"]
		self.assertEqual(payload["max_tokens"], ANTHROPIC_MAX_TOKENS)
		self.assertEqual(ANTHROPIC_MAX_TOKENS, 16_000)

	def test_a_whole_request_asks_for_the_same_budget(self):
		"""The non-streaming builder is a second copy of the body. It must agree."""
		answer = {"content": [{"type": "text", "text": "Hi"}], "usage": {}}
		with patch("frappe_agents.runner.providers._post", return_value=answer) as post:
			call_model(PROFILE, MESSAGES)

		self.assertEqual(post.call_args.args[2]["max_tokens"], ANTHROPIC_MAX_TOKENS)


class TestCutOffAnswer(AnthropicCase):
	def test_a_reply_cut_off_while_thinking_fails_the_run_and_says_why(self):
		"""Scenario: max_tokens stop, one thinking block, no text, no tool call.

		It used to end Completed with an empty answer.
		"""
		run, post = self.execute(cut_off())

		self.assertEqual(post.call_count, 1)
		self.assertEqual(run_value(run.name, "status"), "Failed")
		self.assertEqual(run_value(run.name, "error"), CUT_OFF_MESSAGE)
		self.assertFalse(run_value(run.name, "output_message"))
		# What the turn cost is still recorded: the tokens were spent.
		self.assertEqual(run_value(run.name, "tokens_out"), ANTHROPIC_MAX_TOKENS)
		self.assertEqual(run_value(run.name, "steps_taken"), 1)

	def test_a_normal_reply_is_unchanged(self):
		"""The control: an answer that ends on end_turn completes with its text."""
		run, _post = self.execute(reply(thinking_block(), text_block("Three are open.", 1), stop="end_turn"))

		self.assertEqual(run_value(run.name, "status"), "Completed")
		self.assertEqual(run_value(run.name, "output_message"), "Three are open.")
		self.assertFalse(run_value(run.name, "error"))

	def test_a_reply_cut_off_after_it_started_keeps_what_it_wrote(self):
		"""A partial answer is still an answer. It stays Completed, as before."""
		run, _post = self.execute(
			reply(thinking_block(), text_block("Three are open, and", 1), stop="max_tokens")
		)

		self.assertEqual(run_value(run.name, "status"), "Completed")
		self.assertEqual(run_value(run.name, "output_message"), "Three are open, and")

	def test_a_cut_off_on_the_last_allowed_step_still_says_it_was_cut_off(self):
		"""Not "stopped after N steps": the step limit is not what ended it."""
		frappe.db.set_value("Agent", AGENT, "max_steps", 2, update_modified=False)
		frappe.clear_document_cache("Agent", AGENT)
		self.addCleanup(frappe.clear_document_cache, "Agent", AGENT)

		run, post = self.execute(reply(tool_block(), stop="tool_use"), cut_off())

		self.assertEqual(post.call_count, 2)
		self.assertEqual(run_value(run.name, "status"), "Failed")
		self.assertEqual(run_value(run.name, "error"), CUT_OFF_MESSAGE)

	def test_asking_again_replays_the_tool_results_of_the_cut_off_turn(self):
		"""A Failed run comes back with its work, so the retry does not pay twice."""
		conversation = make_conversation(RESTRICTED_USER).name
		first, _post = self.execute(
			reply(tool_block(), stop="tool_use"), cut_off(), conversation=conversation, message="Summarise."
		)
		self.assertEqual(run_value(first.name, "status"), "Failed")
		self.assertEqual(len(tool_calls_for(first.name)), 1)

		second, post = self.execute(
			reply(text_block("Here it is."), stop="end_turn"), conversation=conversation, message="Again."
		)

		self.assertEqual(run_value(second.name, "status"), "Completed")
		sent = post.call_args.kwargs["json"]["messages"]
		blocks = [part["type"] for message in sent for part in message["content"]]
		self.assertIn("tool_use", blocks)
		self.assertIn("tool_result", blocks)
		# The unsigned thinking of the cut-off turn is not sent back, and no
		# assistant turn goes out empty.
		self.assertNotIn("thinking", blocks)
		for message in sent:
			self.assertTrue(message["content"])
