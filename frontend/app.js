// Question page. Shows whatever item the server sends, one at a time, and posts back the
// answer with that question's one-time token. Notes for the team: docs/frontend.md.
//
// All item text goes in with textContent, never innerHTML: passages, stems and options are
// plain text.
"use strict";

const $ = (id) => document.getElementById(id);

let sessionId = null;
let current = null;           // the ShownItem being answered
let getAnswer = () => null;   // current answer, or null while unanswered
let consentText = null;       // paragraphs to agree to before starting, if any

async function api(path, body) {
  const init = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const res = await fetch(path, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : `Server error ${res.status}`);
  }
  return data;
}

function show(screen) {
  for (const id of ["start", "consent", "question", "done"]) $(id).hidden = id !== screen;
}

function setError(message) {
  $("error").textContent = message || "";
}

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

function paragraphs(box, texts) {
  box.replaceChildren(...texts.map((t) => el("p", { textContent: t })));
}

// ------------------------------------------------------------------ one renderer per item type
// Each draws into `area` and returns a function giving the current answer (or null).

function renderChoice(area, item, onChange) {
  const group = el("div", { className: item.item_type === "likert" ? "choices likert" : "choices" });
  group.setAttribute("role", "radiogroup");
  group.setAttribute("aria-labelledby", "stem");
  item.options.forEach((text, i) => {
    const input = el("input", { type: "radio", name: "choice", value: String(i) });
    input.addEventListener("change", onChange);
    // "1. Venus" as one run of text (see docs/frontend.md).
    const label = el("span", {}, el("b", { className: "num", textContent: `${i + 1}.` }), ` ${text}`);
    group.append(el("label", {}, input, label));
  });
  area.append(group);
  return () => {
    const checked = group.querySelector("input:checked");
    return checked ? Number(checked.value) : null;
  };
}

function renderRanking(area, item, onChange) {
  const order = item.options.map((_, i) => i);   // option indexes, first = ranked highest
  const list = el("ol", { className: "ranking" });
  const announce = el("p", { className: "sr-only" });
  announce.setAttribute("aria-live", "polite");

  function move(pos, delta) {
    const to = pos + delta;
    [order[pos], order[to]] = [order[to], order[pos]];
    draw();
    const moved = list.children[to];
    const same = moved.querySelector(delta < 0 ? ".up" : ".down");
    (same.disabled ? moved.querySelector(delta < 0 ? ".down" : ".up") : same).focus();
    announce.textContent = `${item.options[order[to]]} is now number ${to + 1}`;
    onChange();
  }

  function draw() {
    list.replaceChildren();
    order.forEach((opt, pos) => {
      const text = item.options[opt];
      const up = el("button", { type: "button", className: "up", textContent: "↑", disabled: pos === 0 });
      const down = el("button", {
        type: "button", className: "down", textContent: "↓", disabled: pos === order.length - 1,
      });
      up.setAttribute("aria-label", `Move ${text} up`);
      down.setAttribute("aria-label", `Move ${text} down`);
      up.addEventListener("click", () => move(pos, -1));
      down.addEventListener("click", () => move(pos, +1));
      list.append(el("li", {}, el("span", { textContent: text }), up, down));
    });
  }

  draw();
  area.append(el("p", { className: "hint", textContent: "Use the arrows to reorder. The top is first." }), list, announce);
  return () => order.slice();   // the starting order is a valid answer too
}

function renderNumeric(area, item, onChange) {
  const input = el("input", { type: "number", step: "any", inputMode: "decimal" });
  input.setAttribute("aria-labelledby", "stem");
  input.addEventListener("input", onChange);
  area.append(input);
  return () => {
    const raw = input.value.trim();
    const n = Number(raw);
    return raw !== "" && Number.isFinite(n) ? n : null;
  };
}

function renderOpenText(area, item, onChange) {
  const box = el("textarea", { rows: 5 });
  box.setAttribute("aria-labelledby", "stem");
  box.addEventListener("input", onChange);
  area.append(box);
  return () => (box.value.trim() !== "" ? box.value : null);
}

// Reading text for the question, if any, above the stem (see docs/frontend.md).
// Returns the passage element (focusable but not in the tab order), or null.
function placePassage(text) {
  if (text == null) return null;
  const node = el("p", { className: "passage", textContent: text });
  node.tabIndex = -1;
  $("stem").before(node);
  return node;
}

// Extra item text from the server, placed where it says (see docs/frontend.md).
function placeAux(aux) {
  if (!aux) return;
  // A div, not a p (see docs/frontend.md).
  const node = el("div", { className: aux.style === "clear" ? "aux aux-clear" : "aux aux-clip",
                           textContent: aux.text });
  if (aux.style === "clip_aria") node.setAttribute("aria-hidden", "true");
  if (aux.at === "before_stem") $("stem").before(node);
  else if (aux.at === "after_options") $("answer").after(node);
  else $("stem").after(node);   // after_stem
}

const RENDERERS = {
  mcq: renderChoice,
  likert: renderChoice,
  ranking: renderRanking,
  numeric: renderNumeric,
  open_text: renderOpenText,
};

// ------------------------------------------------------------------ flow

function refresh() {
  $("next").disabled = getAnswer() === null;
}

function render(step) {
  sessionId = step.session_id;
  if (step.finished) {   // finished (or paused for review; the respondent sees the same page)
    current = null;
    $("bar").style.width = "100%";
    show("done");
    $("again").focus();
    if (consentText) showDebrief();
    return;
  }
  current = step.item;
  const renderer = RENDERERS[current.item_type];
  if (!renderer) throw new Error(`Can't show item type ${current.item_type}`);

  $("progress").textContent = `Question ${current.position} of ${current.total}`;
  $("bar").style.width = `${(100 * (current.position - 1)) / current.total}%`;
  $("stem").textContent = current.stem;
  const card = $("form");
  card.classList.remove("enter");
  void card.offsetWidth;   // restart the entrance animation for each new question
  card.classList.add("enter");
  const area = $("answer");
  area.replaceChildren();
  for (const old of document.querySelectorAll("#item .aux, #item .passage")) old.remove();
  getAnswer = renderer(area, current, refresh);
  const passage = placePassage(current.passage);   // first, so text placed before the stem goes after it
  placeAux(current.aux);
  refresh();
  show("question");
  (passage || $("stem")).focus();   // start at the top: the passage when there is one
  ItemTelemetry.start(area);
}

async function begin(assessmentId, title, consent = false) {
  setError("");
  $("brand").textContent = title;
  try {
    render(await api("/session/start", { assessment_id: assessmentId, consent }));
  } catch (e) {
    setError(e.message);
  }
}

async function submit(event) {
  event.preventDefault();
  const answer = getAnswer();
  if (answer === null || !current) return;
  $("next").disabled = true;
  const body = {
    token: current.token,   // one-time token for this question
    answer,
    telemetry: ItemTelemetry.report(),   // counts and timings only (telemetry.js)
  };
  try {
    render(await api(`/session/${encodeURIComponent(sessionId)}/respond`, body));
    setError("");
  } catch (e) {
    setError(e.message);
    refresh();
  }
}

async function loadAssessments() {
  const list = $("assessments");
  list.replaceChildren();
  for (const a of await api("/assessments")) {
    const button = el(
      "button",
      { type: "button", className: "pick" },
      el("span", { className: "chip", textContent: ICONS[a.assessment_type] || "✦" }),
      el(
        "span",
        {},
        el("span", { className: "pick-title", textContent: a.title }),
        el("span", { className: "pick-meta", textContent: `${a.n_items} questions · ${a.assessment_type}` }),
      ),
      el("span", { className: "pick-go", textContent: "→" }),
    );
    button.querySelector(".chip").setAttribute("aria-hidden", "true");
    button.querySelector(".pick-go").setAttribute("aria-hidden", "true");
    button.addEventListener("click", () => (consentText ? askConsent(a) : begin(a.assessment_id, a.title)));
    list.append(el("li", {}, button));
  }
}

const ICONS = { quiz: "🧠", survey: "💬" };

// ------------------------------------------------------------------ pilot: consent, stop, debrief

let consentFor = null;   // the assessment waiting for "I agree"

function askConsent(a) {
  setError("");
  consentFor = a;
  $("agree").checked = false;
  $("consent-yes").disabled = true;
  show("consent");
  $("consent-text").focus();
}

async function withdraw() {
  await api(`/session/${encodeURIComponent(sessionId)}/withdraw`, {});
  sessionId = null;
}

// The closing text is fetched only after the last question.
async function showDebrief() {
  try {
    const { paragraphs: texts } = await api(`/session/${encodeURIComponent(sessionId)}/debrief`);
    paragraphs($("debrief"), texts);
    $("debrief").hidden = false;
    $("remove").hidden = false;
    $("code").textContent = sessionId;  // random; the only way to find these answers later
    $("code-note").hidden = false;
  } catch (e) {
    setError(e.message);
  }
}

function backToStart(message = "") {
  setError("");
  $("brand").textContent = "Quick Questions";
  $("debrief").hidden = true;
  $("debrief").replaceChildren();
  $("remove").hidden = true;
  $("code-note").hidden = true;
  $("code").textContent = "";
  $("withdrawn").textContent = "";
  disarm($("stop"));
  disarm($("remove"));
  $("start-note").textContent = message;
  show("start");
}

$("agree").addEventListener("change", () => { $("consent-yes").disabled = !$("agree").checked; });
$("consent-yes").addEventListener("click", () => begin(consentFor.assessment_id, consentFor.title, true));
$("consent-no").addEventListener("click", () => backToStart("No problem, thanks for looking. You can close this page."));
// Removing answers can't be undone, so it takes a second click (no pop-up dialog). The button
// goes back to normal after a few seconds if the second click doesn't come.
const CONFIRM = "Click again to remove everything you've answered";
function disarm(button) {
  if (button.dataset.label) button.textContent = button.dataset.label;
  delete button.dataset.armed;
  clearTimeout(Number(button.dataset.timer));
}
function confirmed(button) {
  if (button.dataset.armed) {
    disarm(button);
    return true;
  }
  button.dataset.label = button.textContent;
  button.dataset.armed = "1";
  button.textContent = CONFIRM;
  button.dataset.timer = String(setTimeout(() => disarm(button), 6000));
  return false;
}

$("stop").addEventListener("click", async () => {
  if (!confirmed($("stop"))) return;
  try {
    await withdraw();
    current = null;
    backToStart("You stopped, and everything you answered has been removed. Thank you anyway!");
  } catch (e) {
    setError(e.message);
  }
});
$("remove").addEventListener("click", async () => {
  if (!confirmed($("remove"))) return;
  try {
    await withdraw();
    $("remove").hidden = true;
    $("code-note").hidden = true;
    $("withdrawn").textContent = "Done: your answers have been removed. Thank you for taking part.";
  } catch (e) {
    setError(e.message);
  }
});

$("form").addEventListener("submit", submit);
$("again").addEventListener("click", () => backToStart());

async function init() {
  const study = await api("/study");
  consentText = study.consent;
  if (consentText) {
    paragraphs($("consent-text"), consentText);
    $("consent-text").tabIndex = -1;
    $("stop").hidden = false;
  }
  await loadAssessments();
}
init().catch((e) => setError(e.message));
