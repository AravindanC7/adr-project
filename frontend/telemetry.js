// Per-question telemetry (step #23). Owner: Harshini.
//
// Records HOW a question was answered, never WHAT was typed: only counts and timings. No key
// values, no pasted text and no answer text are stored or sent (AGENTS.md rule 7). Pasted text
// is measured for its length only, and that number is not sent yet (see EXTRA below).
//
// Sent to the server as the contract Telemetry (adr/contracts/mechanism.py):
//   time_on_item_ms   question shown -> answer submitted
//   paste_events      paste events inside the answer
//   focus_loss_count  times the person left the page (tab switch, other window), counted once
//                     per absence even though browsers fire both blur and visibilitychange
//   keystroke_count   key presses inside the answer (auto-repeat not counted)
//   mean_interkey_ms  mean gap between consecutive key presses, null with fewer than 2
//
// Measured but NOT sent until the group adds them to the contract (proposal for #25):
//   first_key_ms      question shown -> first key press in the answer
//   away_ms           total time spent away from the page
//   pasted_chars      total length of pasted text
//   edit_keys         Backspace / Delete presses
"use strict";

const ItemTelemetry = (() => {
  const SEND_EXTRA = false;   // flip once the contract has the extra fields

  let t = null;               // counters for the question on screen
  let answerArea = null;

  function now() {
    return performance.now();
  }

  function inAnswer(event) {
    return answerArea !== null && answerArea.contains(event.target);
  }

  function onKeyDown(event) {
    if (!t || event.repeat || !inAnswer(event)) return;
    const at = now();
    if (t.lastKeyAt !== null) t.gapSum += at - t.lastKeyAt;
    if (t.firstKeyAt === null) t.firstKeyAt = at;
    t.lastKeyAt = at;
    t.keys += 1;
    // Only whether it was an edit key; the key itself is never kept.
    if (event.key === "Backspace" || event.key === "Delete") t.editKeys += 1;
  }

  function onPaste(event) {
    if (!t || !inAnswer(event)) return;
    t.pastes += 1;
    const text = event.clipboardData ? event.clipboardData.getData("text") : "";
    t.pastedChars += text.length;   // the length, never the text
  }

  function leave() {
    if (!t || t.awaySince !== null) return;   // already away: one absence, not two
    t.awaySince = now();
    t.leaves += 1;
  }

  function back() {
    if (!t || t.awaySince === null) return;
    t.awayMs += now() - t.awaySince;
    t.awaySince = null;
  }

  function onVisibility() {
    if (document.visibilityState === "hidden") leave();
    else back();
  }

  window.addEventListener("keydown", onKeyDown, true);
  window.addEventListener("paste", onPaste, true);
  window.addEventListener("blur", leave);
  window.addEventListener("focus", back);
  document.addEventListener("visibilitychange", onVisibility);

  return {
    // Call when a question is drawn. `area` is the element holding the answer inputs.
    start(area) {
      answerArea = area;
      t = {
        shownAt: now(), keys: 0, firstKeyAt: null, lastKeyAt: null, gapSum: 0,
        pastes: 0, pastedChars: 0, editKeys: 0, leaves: 0, awaySince: null, awayMs: 0,
      };
      // Not focused when the question appears: time away starts now, but that isn't a "leave".
      if (document.visibilityState === "hidden" || !document.hasFocus()) t.awaySince = now();
    },

    // Call on submit. Returns the contract Telemetry for the question on screen.
    report() {
      if (!t) return { time_on_item_ms: 0 };
      back();   // submitting means they're here
      const at = now();
      const out = {
        time_on_item_ms: Math.round(at - t.shownAt),
        paste_events: t.pastes,
        focus_loss_count: t.leaves,
        keystroke_count: t.keys,
        mean_interkey_ms: t.keys >= 2 ? Math.round(t.gapSum / (t.keys - 1)) : null,
      };
      const extra = {
        first_key_ms: t.firstKeyAt === null ? null : Math.round(t.firstKeyAt - t.shownAt),
        away_ms: Math.round(t.awayMs),
        pasted_chars: t.pastedChars,
        edit_keys: t.editKeys,
      };
      return SEND_EXTRA ? { ...out, ...extra } : out;
    },
  };
})();
