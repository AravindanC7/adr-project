"""Browser test for telemetry (#23): the page sends counts and timings, never what was typed.

Same setup as test_frontend_copy.py (real server + headless Chromium), skipped without them:
    pip install -e ".[server,browser]"
    python -m playwright install chromium
"""

import pytest

pytest.importorskip("fastapi")
sync_api = pytest.importorskip("playwright.sync_api")

from tests.test_frontend_copy import _launch

CONTRACT_FIELDS = {
    "time_on_item_ms", "paste_events", "focus_loss_count", "keystroke_count", "mean_interkey_ms",
}
SECRET = "my secret answer"


@pytest.fixture(scope="module")
def browser_page():
    server, url = _launch("clip")
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


@pytest.fixture
def survey(browser_page):
    """Fresh survey on its first question (open text)."""
    pg, url = browser_page
    pg.goto(url)
    pg.get_by_role("button", name="Stub survey").click()
    pg.wait_for_selector("#answer textarea")
    return pg


def submit(pg):
    """Click Next and return the telemetry the page sent."""
    with pg.expect_request(lambda r: r.url.endswith("/respond")) as req:
        pg.click("#next")
    body = req.value.post_data_json
    assert set(body) == {"token", "answer", "telemetry"}
    return body


def test_typing_is_counted_not_recorded(survey):
    survey.click("#answer textarea")
    survey.keyboard.type(SECRET, delay=60)
    survey.wait_for_timeout(200)
    body = submit(survey)
    t = body["telemetry"]
    assert set(t) == CONTRACT_FIELDS  # the extra fields are not sent until the contract has them
    assert t["keystroke_count"] == len(SECRET)
    assert 40 <= t["mean_interkey_ms"] <= 250
    assert t["paste_events"] == 0 and t["focus_loss_count"] == 0
    assert t["time_on_item_ms"] >= 60 * (len(SECRET) - 1)
    # The answer itself is sent (that's the response); the telemetry carries no text at all.
    assert body["answer"] == SECRET and SECRET not in str(t)


def test_paste_is_counted(survey):
    survey.evaluate("""() => {
      const box = document.querySelector('#answer textarea');
      const data = new DataTransfer();
      data.setData('text/plain', 'An answer copied from somewhere else.');
      box.dispatchEvent(new ClipboardEvent('paste', {clipboardData: data, bubbles: true}));
      box.value = 'An answer copied from somewhere else.';
      box.dispatchEvent(new Event('input', {bubbles: true}));
    }""")
    t = submit(survey)["telemetry"]
    assert t["paste_events"] == 1
    assert t["keystroke_count"] == 0 and t["mean_interkey_ms"] is None


def test_leaving_the_page_counts_once_per_absence(survey):
    survey.evaluate("""() => {
      window.dispatchEvent(new Event('blur'));
      window.dispatchEvent(new Event('blur'));   // same absence: blur + visibilitychange
      window.dispatchEvent(new Event('focus'));
      window.dispatchEvent(new Event('blur'));   // second absence
      window.dispatchEvent(new Event('focus'));
    }""")
    survey.fill("#answer textarea", "Back now.")
    t = submit(survey)["telemetry"]
    assert t["focus_loss_count"] == 2


def test_counters_reset_for_each_question(survey):
    survey.click("#answer textarea")
    survey.keyboard.type("first", delay=20)
    submit(survey)
    survey.wait_for_selector("#answer input[type=radio]")  # question 2 is a Likert item
    survey.locator("#answer label").nth(3).click()
    t = submit(survey)["telemetry"]
    assert t["keystroke_count"] == 0 and t["paste_events"] == 0 and t["focus_loss_count"] == 0
