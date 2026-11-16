# Copyright (c) 2026, Malik AlZubaidi and contributors
# For license information, please see LICENSE

"""A scanned PDF, on the Anthropic wire, as page images the model can read.

A PDF sent as a document block is turned into page images on Anthropic's side,
at a size of its choosing. On scanned delivery notes that size was too small for
the print on them: a whole reading cost about 3,000 tokens where one page at the
model's own limit costs 4,784, and "468" came back as "168", "612" as "312".

So a PDF with no text layer is drawn here, page by page, at the largest size the
model reads without shrinking it, and sent as one image per page. A PDF that
carries its own text still goes as itself: its text travels beside the pages.

Only the HTTP post is stubbed. What is asserted is the body that would be sent.
"""

import base64
import math
import sys
from io import BytesIO
from unittest.mock import patch

import frappe
from PIL import Image, ImageDraw

from frappe_agents.documents import reader
from frappe_agents.documents.pages import PAGE_LONG_EDGE, PAGE_VISUAL_TOKENS
from frappe_agents.extraction.pipeline import run_extraction
from frappe_agents.runner.providers import call_model_extract
from frappe_agents.tests.fixtures import (
	DRAFT_USER,
	EXTRACT_PROFILE,
	PROVIDER,
	make_pdf_attachment,
	make_run,
	queue_extraction_as,
)
from frappe_agents.tests.test_document_reading import ReadingTestCase, text_pdf

PDF = "application/pdf"
PATCH = 28


def scanned_pdf(pages: int = 1, size: tuple[int, int] = (1240, 1753)) -> bytes:
	"""A PDF whose pages are pictures of text: A4 at 150 dpi, no text layer."""
	images = []
	for number in range(1, pages + 1):
		image = Image.new("RGB", size, "white")
		ImageDraw.Draw(image).text((100, 100), f"Pipe HDPE 180mm  Mtr  468.00  page {number}", fill="black")
		images.append(image)
	buffer = BytesIO()
	images[0].save(buffer, "PDF", resolution=150, save_all=True, append_images=images[1:])
	return buffer.getvalue()


def visual_tokens(width: int, height: int) -> int:
	return math.ceil(width / PATCH) * math.ceil(height / PATCH)


def answered(text: str = "Pipe HDPE 180mm | Mtr | 468.00") -> dict:
	"""An Anthropic reply carrying one transcription."""
	return {
		"content": [{"type": "text", "text": frappe.as_json({"text": text})}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 5000, "output_tokens": 50},
	}


class AnthropicScanCase(ReadingTestCase):
	"""The extraction profile, pointed at Anthropic and a model that takes a schema."""

	def setUp(self) -> None:
		super().setUp()
		frappe.db.set_value("LLM Provider", PROVIDER, "provider_type", "Anthropic", update_modified=False)
		frappe.db.set_value(
			"LLM Model Profile", EXTRACT_PROFILE, "model_id", "claude-sonnet-5", update_modified=False
		)
		for doctype, name in (("LLM Provider", PROVIDER), ("LLM Model Profile", EXTRACT_PROFILE)):
			frappe.clear_document_cache(doctype, name)
			self.addCleanup(frappe.clear_document_cache, doctype, name)

	def sent(self, content: bytes) -> list[dict]:
		"""The user turn one transcription call would send."""
		with patch("frappe_agents.runner.providers._post", return_value=answered()) as post:
			call_model_extract(
				EXTRACT_PROFILE,
				content,
				PDF,
				"delivery-note.pdf",
				reader.TRANSCRIPTION_SCHEMA,
				reader.TRANSCRIPTION_INSTRUCTIONS,
			)
		return post.call_args.args[2]["messages"][0]["content"]

	def kinds(self, blocks: list[dict]) -> list[str]:
		return [block["type"] for block in blocks]

	def page_image(self, block: dict) -> Image.Image:
		self.assertEqual(block["source"]["media_type"], "image/png")
		return Image.open(BytesIO(base64.b64decode(block["source"]["data"])))

	def assert_sent_as_itself(self, blocks: list[dict], content: bytes) -> None:
		self.assertEqual(self.kinds(blocks), ["document", "text"])
		self.assertEqual(blocks[0]["source"]["data"], base64.b64encode(content).decode())


class TestScannedPages(AnthropicScanCase):
	def test_a_scan_goes_as_one_labelled_image_per_page(self):
		"""Scenario S-7: no text layer, so no document block. Pages in order."""
		blocks = self.sent(scanned_pdf(pages=2))

		self.assertEqual(self.kinds(blocks), ["text", "image", "text", "image", "text"])
		self.assertEqual(blocks[0]["text"], "Page 1:")
		self.assertEqual(blocks[2]["text"], "Page 2:")
		# The question still comes after the document, as Anthropic advises.
		self.assertEqual(blocks[-1]["text"], reader.TRANSCRIPTION_INSTRUCTIONS)

	def test_a_page_is_drawn_at_the_largest_size_the_model_reads_unshrunk(self):
		"""Big enough for small print, never so big that Anthropic shrinks it again."""
		page = self.page_image(self.sent(scanned_pdf())[1])
		width, height = page.size

		self.assertLessEqual(max(width, height), PAGE_LONG_EDGE)
		self.assertLessEqual(visual_tokens(width, height), PAGE_VISUAL_TOKENS)
		# Close to the limit, not merely under it: this is the whole fix.
		self.assertGreater(visual_tokens(width, height), PAGE_VISUAL_TOKENS * 0.95)
		# The page keeps its shape: A4 is 1 : 1.414.
		self.assertAlmostEqual(height / width, 1753 / 1240, places=2)

	def test_a_pdf_with_a_text_layer_still_goes_as_itself(self):
		"""Its own text travels beside Anthropic's page images. Nothing to gain here."""
		content = text_pdf(["Delivery order 26312419", "Quantity 468.00 Mtr"])

		self.assert_sent_as_itself(self.sent(content), content)

	def test_without_the_renderer_a_scan_goes_as_itself(self):
		"""pypdfium2 comes with ERPNext, not with Frappe. Without it, nothing breaks."""
		content = scanned_pdf()
		with patch.dict(sys.modules, {"pypdfium2": None}):
			blocks = self.sent(content)

		self.assert_sent_as_itself(blocks, content)

	def test_pages_too_heavy_for_the_wire_go_as_the_pdf_itself(self):
		"""Over the per-image limit, the PDF that already passed its own check is sent."""
		content = scanned_pdf()
		with patch("frappe_agents.runner.providers.ANTHROPIC_MAX_IMAGE_ENCODED", 1024):
			blocks = self.sent(content)

		self.assert_sent_as_itself(blocks, content)

	def test_more_pages_than_one_request_may_carry_as_images_go_as_the_pdf(self):
		"""Past 20 images Anthropic tightens the size of every image in the request."""
		content = scanned_pdf(pages=21, size=(124, 175))

		self.assert_sent_as_itself(self.sent(content), content)

	def test_the_transcription_marks_an_unsure_digit_and_never_guesses(self):
		"""A wrong quantity read with confidence is worse than one marked as doubtful."""
		instructions = reader.TRANSCRIPTION_INSTRUCTIONS

		self.assertIn("[?]", instructions)
		self.assertIn("never guess", instructions.lower())
		self.assertEqual(self.sent(scanned_pdf())[-1]["text"], instructions)


class TestScannedPagesOnEveryPath(AnthropicScanCase):
	"""The tool and the extraction job reach the model through the same call."""

	def test_read_document_sends_a_scan_as_page_images(self):
		file = self.attach("delivery-note.pdf", scanned_pdf())
		run = make_run(effective_user=DRAFT_USER, model_profile=EXTRACT_PROFILE)

		with patch("frappe_agents.runner.providers._post", return_value=answered()) as post:
			payload, _run = self.read_file(file, run=run)

		self.assertTrue(payload["ok"], payload["error"])
		self.assertIn("468.00", self.body(payload["result"]))
		blocks = post.call_args.args[2]["messages"][0]["content"]
		self.assertEqual(self.kinds(blocks), ["text", "image", "text"])

	def test_an_extraction_of_a_scan_sends_page_images_too(self):
		file = make_pdf_attachment(content=scanned_pdf())
		extraction = queue_extraction_as(DRAFT_USER, file.name)

		with patch("frappe_agents.runner.providers._post", return_value=answered()) as post:
			run_extraction(extraction)

		blocks = post.call_args_list[0].args[2]["messages"][0]["content"]
		self.assertIn("image", self.kinds(blocks))
		self.assertNotIn("document", self.kinds(blocks))
