"""A scanned PDF as the page images a model reads best.

A PDF sent whole is turned into page images on the provider's side, at a size
the provider picks. For a scan that image is all the model gets, and on scanned
delivery notes it was too small for their print: "468" came back as "168". So a
PDF with no text layer is drawn here instead, one PNG per page, at the largest
size the model reads without shrinking it again.

Drawing a page is not editing it. Every pixel comes from the file, nothing is
cropped, cleaned or retouched, and the File keeps the PDF exactly as it was
uploaded. The page is only drawn at a size the model can read.

pypdfium2 is not a Frappe dependency. ERPNext brings it (through pdfplumber), so
it is on most sites; where it is missing, the PDF goes whole, as it always did.
"""

import math
from io import BytesIO

# Below this, a PDF's "text layer" is the stray glyph a scanner leaves behind and
# the document has to be looked at instead. Kept small on purpose: sending a page
# that really does carry its own text to a model costs money and needs a profile
# that can see files, so the doubt has to be genuine. The reader asks the same
# question with this number, so pages are drawn for exactly the PDFs it reads by
# sight.
TEXT_LAYER_MIN_CHARS = 8

# Anthropic's high-resolution limit (Claude 4.7 and later): 2,576 px on the long
# edge and 4,784 visual tokens of 28 x 28 px. A larger image is shrunk on arrival,
# so drawing past it costs bytes and gains nothing. An older model shrinks these
# pages to its own limit, which is no worse than before.
PAGE_LONG_EDGE = 2576
PAGE_VISUAL_TOKENS = 4784
PATCH_PIXELS = 28
# Past 20 images in one request Anthropic tightens the size limit of every image
# in it, so a longer scan goes whole.
MAX_PAGE_IMAGES = 20


def scanned_pages(content: bytes) -> list[bytes] | None:
	"""Each page of a scan as PNG bytes, or None when the PDF should go whole."""
	pdfium = _renderer()
	if pdfium is None or has_text_layer(content):
		return None
	try:
		document = pdfium.PdfDocument(content)
	except Exception:
		return None
	try:
		if not 0 < len(document) <= MAX_PAGE_IMAGES:
			return None
		return [_png(document[index]) for index in range(len(document))]
	except Exception:
		# A page pdfium cannot draw is not a failed reading: the provider gets
		# the PDF and draws it itself.
		return None
	finally:
		document.close()


def has_text_layer(content: bytes) -> bool:
	from pypdf import PdfReader

	try:
		pages = PdfReader(BytesIO(content)).pages
		text = "".join((page.extract_text() or "") for page in pages)
	except Exception:
		return False
	return len(text.strip()) >= TEXT_LAYER_MIN_CHARS


def page_scale(width: float, height: float) -> float:
	"""Pixels per point that fit both of the model's limits, and no more."""
	scale = min(
		PAGE_LONG_EDGE / max(width, height),
		PATCH_PIXELS * math.sqrt(PAGE_VISUAL_TOKENS / (width * height)),
	)
	# Patches are counted whole, so the square root can land a patch over.
	while _visual_tokens(math.ceil(width * scale), math.ceil(height * scale)) > PAGE_VISUAL_TOKENS:
		scale *= 0.995
	return scale


def _png(page) -> bytes:
	width, height = page.get_size()
	image = page.render(scale=page_scale(width, height)).to_pil()
	buffer = BytesIO()
	image.save(buffer, "PNG", optimize=True)
	return buffer.getvalue()


def _visual_tokens(width: int, height: int) -> int:
	return math.ceil(width / PATCH_PIXELS) * math.ceil(height / PATCH_PIXELS)


def _renderer():
	try:
		import pypdfium2
	except ImportError:
		return None
	return pypdfium2
