"""Browser test for mechanism injection (#19): what Chrome copies equals the variant's copy_text.

Archana's adr/mechanisms/CANARY_SPEC.md requires this check in an automated front-end test.
Runs the real server and a headless Chromium. Skipped unless both are installed:
    pip install -e ".[server,browser]"
    python -m playwright install chromium
"""

import json
import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
sync_api = pytest.importorskip("playwright.sync_api")
import uvicorn

from adr.contracts import Item
from adr.harness.prompts import copy_text
from server.app import DEFAULT_BANK, build_variants, create_app, load_bank
from tests.stubs import StubCanary, StubDecoy

MECHANISMS = {"decoy": StubDecoy(), "canary": StubCanary()}
ITEMS, ASSESSMENTS = load_bank(DEFAULT_BANK)
VARIANTS = build_variants(ITEMS, MECHANISMS)
# stem -> the trap variant the page shows for it (the stub bank has at most one per item)
TRAPS = {ITEMS[i].stem: v for i, vs in VARIANTS.items() for v in vs if v.mechanism}

SELECT_ITEM = """() => {
  const range = document.createRange();
  range.selectNodeContents(document.getElementById("item"));
  const sel = getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
  const text = sel.toString();
  sel.removeAllRanges();
  return text;
}"""


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch(hide, **options):
    port = _free_port()
    app = create_app(mechanisms=MECHANISMS, hide=hide, **options)
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            return server, f"http://127.0.0.1:{port}/"
        time.sleep(0.05)
    raise RuntimeError("test server did not start")


@pytest.fixture(scope="module")
def chromium():
    # One Playwright for the whole module: a second sync_playwright() while one is open fails.
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no Playwright Chromium: {e}")
        yield browser
        browser.close()


@pytest.fixture(scope="module", params=["clip", "clear", "clip_aria"])
def page(request, chromium):
    server, url = _launch(request.param)
    context = chromium.new_context()
    try:
        pg = context.new_page()
        pg.goto(url)
        yield request.param, pg
    finally:
        context.close()
        server.should_exit = True


def _trap_pages(pg, title, answer):
    """Start an assessment and yield the question card each time a trap question is shown."""
    pg.get_by_role("button", name=title).click()
    while True:
        pg.wait_for_selector("#question:not([hidden]), #done:not([hidden])")
        if pg.is_visible("#done"):
            pg.click("#again")
            return
        stem = pg.text_content("#stem")
        if stem in TRAPS:
            yield TRAPS[stem]
        answer(pg)
        pg.wait_for_function("s => document.getElementById('stem').textContent !== s ||"
                             " !document.getElementById('done').hidden", arg=stem)


def _answer(pg):
    if pg.locator("#answer input[type=radio]").count():
        pg.locator("#answer label").first.click()
    elif pg.locator("#answer textarea").count():
        pg.fill("#answer textarea", "Something honest.")
    elif pg.locator("#answer input[type=number]").count():
        pg.fill("#answer input[type=number]", "5")
    pg.click("#next")


@pytest.mark.parametrize("title", ["Stub quiz", "Stub survey"])
def test_copied_text_equals_copy_text_and_trap_is_invisible(page, title):
    hide, pg = page
    checked = 0
    for rendered in _trap_pages(pg, title, _answer):
        aux = pg.locator("#item .aux")
        assert aux.count() == 1
        item = ITEMS[rendered.item_id]
        # Stub quiz q4 has a passage: shown above the stem, focus starts there.
        assert pg.locator("#item .passage").count() == (1 if item.passage else 0)
        focused = "document.activeElement.id || document.activeElement.className"
        assert pg.evaluate(f"() => {focused}") == ("passage" if item.passage else "stem")
        box = aux.bounding_box()
        color = aux.evaluate("e => getComputedStyle(e).color")
        if hide == "clear":
            assert color == "rgba(0, 0, 0, 0)"
        else:
            assert box["width"] <= 1 and box["height"] <= 1
        assert aux.evaluate("e => getComputedStyle(e).display") != "none"
        assert (aux.get_attribute("aria-hidden") == "true") == (hide == "clip_aria")

        copied = pg.evaluate(SELECT_ITEM).rstrip("\n")  # the answer box adds a trailing newline
        assert copied == copy_text(rendered, item)  # the harness's text: passage first
        checked += 1
    assert checked == {"Stub quiz": 3, "Stub survey": 1}[title]


SELECT_ONLY = """() => {
  const range = document.createRange();
  range.selectNodeContents(document.getElementById("item"));
  getSelection().removeAllRanges();
  getSelection().addRange(range);
}"""


def test_real_copy_shortcut_gives_copy_text(chromium):
    """What a copy-paste user actually gets: Ctrl+C into the system clipboard, read back.
    Selection.toString() (used above) is close to this but not guaranteed identical."""
    server, url = _launch("clip")
    context = chromium.new_context(permissions=["clipboard-read", "clipboard-write"])
    try:
        pg = context.new_page()
        pg.goto(url)
        rendered = next(_trap_pages(pg, "Stub quiz", _answer))  # q1, a decoy
        pg.bring_to_front()  # the shortcut and the clipboard need the focused page
        pg.evaluate(SELECT_ONLY)
        pg.keyboard.press("ControlOrMeta+C")
        copied = pg.evaluate("""() => Promise.race([
          navigator.clipboard.readText(),
          new Promise((_, no) => setTimeout(() => no(new Error("clipboard timed out")), 5000)),
        ])""")
        # Windows puts \r\n line ends on the clipboard; any text box it's pasted into (a
        # chat window included) turns them back into \n. Nothing else may differ.
        assert copied.replace("\r\n", "\n").rstrip("\n") == rendered.copy_text
    finally:
        context.close()
        server.should_exit = True


# ---------------------------------------------------------------- items with a reading passage
# Two paragraphs, a line break inside one, and markup that must show as plain text.
PASSAGE = (
    "Mercury is the smallest planet. It has almost no air.\n"
    "A year there lasts 88 days.\n\n"
    "Its surface is <b>grey</b> & full of craters, like the Moon's."
)
PASSAGE_ITEMS = [
    Item(item_id="handwritten:p1", source="handwritten", assessment_type="quiz", item_type="mcq",
         passage=PASSAGE, stem="How long is a year on Mercury?",
         options=["88 days", "365 days", "12 days"], correct_option=0,
         eligible_mechanisms=["decoy"]),
    Item(item_id="handwritten:p2", source="handwritten", assessment_type="survey",
         item_type="open_text", passage=PASSAGE, stem="What surprised you in that text?"),
]


class _DecoyBeforeStem(StubDecoy):
    """StubDecoy with its text before the stem: it must land between passage and stem."""

    def generate(self, item, seed):
        r = super().generate(item, seed)
        hidden = {**r.hidden, "hidden_position": "before_stem"}
        opts = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(item.options))
        body = f"{hidden['hidden_text']}\n{item.stem}\n{opts}"
        return r.model_copy(update={"hidden": hidden, "copy_text": body})


@pytest.mark.parametrize("decoy", [StubDecoy(), _DecoyBeforeStem()], ids=["after", "before"])
def test_passage_is_shown_above_the_stem_and_copied_once(chromium, tmp_path, decoy):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({
        "items": [it.model_dump(exclude_none=True) for it in PASSAGE_ITEMS],
        "assessments": [
            {"assessment_id": "pq", "title": "Passage quiz", "assessment_type": "quiz",
             "item_ids": ["handwritten:p1"]},
            {"assessment_id": "ps", "title": "Passage survey", "assessment_type": "survey",
             "item_ids": ["handwritten:p2"]},
        ],
    }), encoding="utf-8")
    mechanisms = {"decoy": decoy}
    items = {it.item_id: it for it in PASSAGE_ITEMS}
    variants = build_variants(items, mechanisms)
    expected = {
        "Passage quiz": copy_text(variants["handwritten:p1"][1], items["handwritten:p1"]),
        "Passage survey": copy_text(variants["handwritten:p2"][0], items["handwritten:p2"]),
    }
    assert all(text.startswith(PASSAGE + "\n\n") for text in expected.values())

    port = _free_port()
    app = create_app(bank_path=bank, mechanisms=mechanisms, hide="clip")
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    context = chromium.new_context()
    try:
        pg = context.new_page()
        pg.goto(f"http://127.0.0.1:{port}/")
        for title, text in expected.items():
            pg.get_by_role("button", name=title).click()
            pg.wait_for_selector("#question:not([hidden])")
            passage = pg.locator("#item .passage")
            assert passage.count() == 1
            assert passage.text_content() == PASSAGE  # markup shown as text, not parsed
            assert passage.locator("b").count() == 0
            order = pg.evaluate("""() => [...document.getElementById("item").children]
                                   .map(e => e.id || e.className)""")
            assert order.index("passage") < order.index("stem")
            assert passage.evaluate("e => e === document.activeElement")  # focus starts here
            assert passage.get_attribute("tabindex") == "-1"  # focusable, not a tab stop
            assert pg.evaluate(SELECT_ITEM).rstrip("\n") == text
            pg.goto(f"http://127.0.0.1:{port}/")
    finally:
        context.close()
        server.should_exit = True
