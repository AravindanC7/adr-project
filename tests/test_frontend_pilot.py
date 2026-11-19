"""Browser test for pilot mode (#30): consent gate, Stop, debrief at the end, remove answers.

Same setup as test_frontend_copy.py (real server + headless Chromium), skipped without them.
"""

import pytest

pytest.importorskip("fastapi")
sync_api = pytest.importorskip("playwright.sync_api")

from server.app import PILOT_DIR, screen_text
from tests.test_frontend_copy import _launch

DEBRIEF = screen_text(PILOT_DIR / "debrief.md")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    server, url = _launch("clip", pilot=True, data_dir=tmp_path_factory.mktemp("pilot"))
    try:
        with sync_api.sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:  # noqa: BLE001
                pytest.skip(f"no Playwright Chromium: {e}")
            yield browser.new_page(), url
            browser.close()
    finally:
        server.should_exit = True


def open_consent(pg, url):
    pg.goto(url)
    pg.get_by_role("button", name="Stub quiz").click()
    pg.wait_for_selector("#consent:not([hidden])")


def test_consent_gate_then_debrief_then_remove(site):
    pg, url = site
    open_consent(pg, url)
    assert "about 15 minutes" in pg.inner_text("#consent-text")
    assert pg.is_disabled("#consent-yes")
    pg.check("#agree")
    pg.click("#consent-yes")
    pg.wait_for_selector("#question:not([hidden])")
    assert pg.is_visible("#stop")
    # Before the end, no debrief anywhere in the page, visible or not.
    assert DEBRIEF[1] not in pg.content()

    pg.locator("#answer label").first.click()
    pg.click("#next")
    pg.wait_for_function(
        "document.querySelector('#position, #progress').textContent.includes('2 of')"
    )
    pg.locator("#answer label").first.click()
    pg.click("#next")
    pg.wait_for_selector("#answer input[type=number]")
    pg.fill("#answer input[type=number]", "7")
    pg.click("#next")
    pg.wait_for_selector("#item .passage")  # q4, the reading passage
    pg.locator("#answer label").first.click()
    pg.click("#next")

    pg.wait_for_selector("#debrief:not([hidden])")
    assert pg.inner_text("#debrief").split("\n\n")[0] == DEBRIEF[0]
    code = pg.text_content("#code")
    assert len(code) == 32 and pg.is_visible("#code-note")  # for removing answers later
    pg.click("#remove")  # first click only asks
    assert "Click again" in pg.text_content("#remove") and pg.is_visible("#remove")
    with pg.expect_response(lambda r: r.url.endswith("/withdraw")) as resp:
        pg.click("#remove")
    assert resp.value.ok
    pg.wait_for_selector("#withdrawn:has-text('removed')")
    assert pg.is_hidden("#remove")


def test_no_thanks_and_stop(site):
    pg, url = site
    open_consent(pg, url)
    pg.click("#consent-no")
    pg.wait_for_selector("#start-note:has-text('No problem')")

    open_consent(pg, url)
    pg.check("#agree")
    pg.click("#consent-yes")
    pg.wait_for_selector("#question:not([hidden])")
    pg.click("#stop")  # first click only asks
    assert "Click again" in pg.text_content("#stop") and pg.is_visible("#question")
    with pg.expect_response(lambda r: r.url.endswith("/withdraw")) as resp:
        pg.click("#stop")
    assert resp.value.ok
    pg.wait_for_selector("#start-note:has-text('removed')")
    assert pg.is_hidden("#question")
