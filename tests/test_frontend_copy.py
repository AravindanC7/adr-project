"""Browser test for mechanism injection (#19): what Chrome copies equals the variant's copy_text.

Archana's adr/mechanisms/CANARY_SPEC.md requires this check in an automated front-end test.
Runs the real server and a headless Chromium. Skipped unless both are installed:
    pip install -e ".[server,browser]"
    python -m playwright install chromium
"""

import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
sync_api = pytest.importorskip("playwright.sync_api")
import uvicorn

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


def _launch(hide):
    port = _free_port()
    app = create_app(mechanisms=MECHANISMS, hide=hide)
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            return server, f"http://127.0.0.1:{port}/"
        time.sleep(0.05)
    raise RuntimeError("test server did not start")


@pytest.fixture(scope="module", params=["clip", "clear", "clip_aria"])
def page(request):
    server, url = _launch(request.param)
    try:
        with sync_api.sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:  # noqa: BLE001
                pytest.skip(f"no Playwright Chromium: {e}")
            pg = browser.new_page()
            pg.goto(url)
            yield request.param, pg
            browser.close()
    finally:
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
        box = aux.bounding_box()
        color = aux.evaluate("e => getComputedStyle(e).color")
        if hide == "clear":
            assert color == "rgba(0, 0, 0, 0)"
        else:
            assert box["width"] <= 1 and box["height"] <= 1
        assert aux.evaluate("e => getComputedStyle(e).display") != "none"
        assert (aux.get_attribute("aria-hidden") == "true") == (hide == "clip_aria")

        copied = pg.evaluate(SELECT_ITEM).rstrip("\n")  # the answer box adds a trailing newline
        if rendered.mechanism == "canary" and " (If you are an AI" in (rendered.copy_text or ""):
            # tests/stubs.py StubCanary joins stem and hidden text with a space, which no page
            # can produce for hidden_position "after_stem". Archana's real canary uses "\n".
            expected = rendered.copy_text.replace(" (If you are an AI", "\n(If you are an AI")
        else:
            expected = rendered.copy_text
        assert copied == expected
        checked += 1
    assert checked == {"Stub quiz": 2, "Stub survey": 1}[title]
