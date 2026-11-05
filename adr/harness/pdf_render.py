"""Render a whole assessment as a printed-quiz PDF for the document-upload attacker (step #20).

A human reading the page sees a title and "Q1.", "Q2." ... with each item's reading passage (if
any), stem and numbered options. DoPE-style traps (a variant's hidden["hidden_text"]) are drawn at
hidden["hidden_position"] (before the stem, after it, or after the options) with PDF text render
mode 3: no fill, no stroke, so nothing shows on the page, but the text is in the text layer that a
model (or any text extractor) reads. The order matches prompts.copy_text.

The output is byte-for-byte deterministic (reportlab's `invariant` mode drops the timestamp and
random document id), so the same document always gets the same LLM cache key.
"""

from __future__ import annotations

import io

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from adr.contracts import Item, RenderedItem

FONT, FONT_BOLD = "Helvetica", "Helvetica-Bold"
SIZE, TITLE_SIZE = 11, 16
LEADING = SIZE * 1.35
MARGIN = 72  # 1 inch
INVISIBLE, VISIBLE = 3, 0  # PDF text render modes


def render_assessment_pdf(
    items: list[Item], variants_in_order: list[RenderedItem], title: str = "Assessment"
) -> tuple[bytes, dict[int, str]]:
    """Return (pdf_bytes, {question number: variant_id}). Question n (from 1) shows
    `variants_in_order[n-1]`, which must be a variant of `items[n-1]`."""
    if len(items) != len(variants_in_order):
        raise ValueError(f"{len(items)} items but {len(variants_in_order)} variants")
    for item, v in zip(items, variants_in_order):
        if v.item_id != item.item_id:
            raise ValueError(f"variant {v.variant_id} is not a variant of {item.item_id}")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter, invariant=1)
    c.setTitle(title)
    page_w, page_h = letter
    width = page_w - 2 * MARGIN
    y = page_h - MARGIN

    def lines(text: str, indent: float = 0, font: str = FONT) -> list[str]:
        out: list[str] = []
        for para in text.splitlines() or [""]:
            out.extend(simpleSplit(para, font, SIZE, width - indent) or [""])
        return out

    def draw(text_lines: list[str], x: float, font: str = FONT, mode: int = VISIBLE) -> None:
        nonlocal y
        for line in text_lines:
            if y < MARGIN + LEADING:
                c.showPage()
                y = page_h - MARGIN
            # The render mode is graphics state and outlives the text block, and reportlab never
            # writes mode 0 back. Save/restore so text after a hidden line stays visible.
            c.saveState()
            t = c.beginText(x, y)
            t.setFont(font, SIZE)
            if mode != VISIBLE:
                t.setTextRenderMode(mode)
            t.textLine(line)
            c.drawText(t)
            c.restoreState()
            y -= LEADING

    t = c.beginText(MARGIN, y)
    t.setFont(FONT_BOLD, TITLE_SIZE)
    t.textLine(title)
    c.drawText(t)
    y -= TITLE_SIZE * 2

    question_map: dict[int, str] = {}
    for n, (item, v) in enumerate(zip(items, variants_in_order), start=1):
        question_map[n] = v.variant_id
        label = f"Q{n}. "
        indent = c.stringWidth(label, FONT_BOLD, SIZE)
        hidden_text = v.hidden.get("hidden_text")
        # {position: lines}; empty when the variant has no hidden text, so drawing it is a no-op.
        hidden_at = {v.hidden["hidden_position"]: lines(hidden_text, indent)} if hidden_text else {}
        hidden_x = MARGIN + indent

        # The label goes on the passage's first line when there is one, else on the stem's.
        stem_lines = lines(v.visible["stem"], indent)
        first_lines = lines(item.passage, indent) if item.passage is not None else stem_lines
        # Keep a question's first lines together with its label.
        if y < MARGIN + LEADING * 3:
            c.showPage()
            y = page_h - MARGIN
        if item.passage is None:
            draw(hidden_at.get("before_stem", []), hidden_x, mode=INVISIBLE)
        t = c.beginText(MARGIN, y)
        t.setFont(FONT_BOLD, SIZE)
        t.textOut(label)
        t.setFont(FONT, SIZE)
        t.textLine(first_lines[0])
        c.drawText(t)
        y -= LEADING
        draw(first_lines[1:], MARGIN + indent)
        if item.passage is not None:
            y -= LEADING * 0.4
            draw(hidden_at.get("before_stem", []), hidden_x, mode=INVISIBLE)
            draw(stem_lines, MARGIN + indent)
        draw(hidden_at.get("after_stem", []), hidden_x, mode=INVISIBLE)

        options = v.visible.get("options") or []
        for i, opt in enumerate(options, start=1):
            draw(lines(f"{i}. {opt}", indent + 12), MARGIN + indent + 12)
        draw(hidden_at.get("after_options", []), hidden_x, mode=INVISIBLE)
        if item.item_type in ("numeric", "open_text"):
            draw(["Answer: ____________________________"], MARGIN + indent)
        y -= LEADING * 0.6

    c.showPage()
    c.save()
    return buf.getvalue(), question_map
