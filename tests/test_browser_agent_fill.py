"""fill() on a fake page: what text goes into the boxes. No server or browser needed."""

import pytest

from adr.harness.browser_agent import BrowserAgentConfig, fill, number_text


class _Box:
    def __init__(self, typed):
        self.typed = typed

    def locator(self, _selector):
        return self

    def fill(self, text):
        self.typed.append(text)


class _Page:
    def __init__(self):
        self.typed: list[str] = []

    def locator(self, _selector):
        return _Box(self.typed)


@pytest.mark.parametrize("answer", [1234567.0, 1234567, 0, -42.0, 3.5, 0.1 + 0.2, 1e-7, 98765.4321])
def test_numeric_answer_round_trips_exactly(answer):
    page = _Page()
    fill(page, "numeric", answer, BrowserAgentConfig(run_id="t", models=["m"]))
    (text,) = page.typed
    assert float(text) == answer
    assert text == number_text(answer)


def test_seven_digit_answer_is_not_truncated():
    assert number_text(1234567.0) == "1234567"   # format(.., "g") gave "1.23457e+06"
    assert number_text(3.0) == "3"
