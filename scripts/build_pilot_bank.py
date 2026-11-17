"""Build the human pilot's question bank (#35): server/pilot_items.json.

Owner: Harshini.

    python scripts/build_pilot_bank.py --arc <ARC-Challenge test .jsonl> --mmlu <MMLU data dir>
        --opinionqa <OpinionQA human_resp dir> --ipip-data <IPIP data.csv>
        --ipip-codebook <IPIP codebook.txt>

One flow, taken in one session (server "flow"): consent, then the quiz part, then the survey
part, then one debrief. About 24 questions, 15-20 minutes.

  Quiz (12 MCQ, every one can carry a decoy): 6 ARC-Challenge + 6 MMLU from general-knowledge
    subjects. Short stems only, no "figure/diagram/table" questions, four short options.
  Survey (12): 6 OpinionQA (Pew) opinion questions, 4 IPIP items as 2 reverse-keyed pairs (the
    opinion trap's contradiction check needs both halves of a pair), and 2 handwritten open-text
    questions eligible for the canary (served plain until Archana's canary is on main).

No held-out data: ASURRE holds the held-out survey (AGENTS.md section 8), so it isn't used, and
every item is split "train". The selection is seeded, so the same raw files give the same bank.

The output contains dataset questions (Pew's terms limit redistribution), so it is git-ignored
and rebuilt from the raw files on the shared drive.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from adr.contracts import Assessment, HumanDistribution, Item
from adr.data.loaders import load_arc, load_ipip, load_mmlu, load_opinionqa
from adr.mechanisms.decoy import DecoyMechanism

ROOT = Path(__file__).resolve().parent.parent
SEED = 576
N_ARC, N_MMLU, N_OPINION = 6, 6, 6
IPIP_PAIRS = [("E1", "E2"), ("C1", "C2")]  # one +/one - keyed item each, same trait
MMLU_SUBJECTS = [
    "high_school_geography", "high_school_world_history", "nutrition", "miscellaneous",
    "high_school_psychology", "us_foreign_policy", "global_facts", "sociology",
]
MAX_STEM_WORDS, MAX_OPTION_WORDS = 40, 10
NEEDS_A_PICTURE = re.compile(r"\b(figure|diagram|graph|table|picture|image|shown|below)\b",
                             re.IGNORECASE)

# Pew American Trends Panel Wave 41 (Dec 11-23, 2018; one of OpinionQA's waves), from Pew's
# published topline: https://www.pewresearch.org/wp-content/uploads/sites/20/2019/03/
# US-2050_topline-FINAL.pdf. Asked of all 2,524 adults. Percentages are Pew's weighted totals
# with "No answer" left out (renormalized below); n is the unweighted sample, as in gss.py.
# Used when the OpinionQA human_resp files aren't available. SOLVPROB is one row of a grid,
# reworded to stand alone. Divisive topics (abortion, immigration, race) left out on purpose.
PEW_W41_N = 2524
PEW_W41 = [
    ("OPTIMIST", ("Please think about what things will be like in 2050, about 30 years from now. "
     "Thinking about the future of the United States, would you say you are …"),
     [("Very optimistic", 12), ("Somewhat optimistic", 44), ("Somewhat pessimistic", 31),
      ("Very pessimistic", 13)]),
    ("AVGFAM", ("Over the next 30 years, do you think that the average American family will see "
     "its standard of living …"),
     [("Get better", 20), ("Stay about the same", 35), ("Get worse", 44)]),
    ("ENVC", "By the year 2050, do you think the overall condition of the environment will be …",
     [("Better than it is now", 16), ("Worse than it is now", 59),
      ("About the same as it is now", 25)]),
    ("ROBWRK", ("Overall, how likely do you think it is that 30 years from now robots and computers "
     "will do much of the work currently done by humans? Do you think this will …"),
     [("Definitely happen", 30), ("Probably happen", 52), ("Probably not happen", 16),
      ("Definitely not happen", 2)]),
    ("NATDEBT", ("Which do you think is most likely to occur over the next 30 years? "
     "The national debt will …"),
     [("Grow larger", 63), ("Stay about the same", 21), ("Be reduced", 14),
      ("Be eliminated", 2)]),
    ("SOLVPROB_a", ("In the future, what kind of impact do you think science and technology will "
     "have in solving the biggest problems facing the country?"),
     [("A very positive impact", 43), ("A somewhat positive impact", 44),
      ("A somewhat negative impact", 8), ("A very negative impact", 3)]),
]


def pew_w41() -> tuple[list[Item], list[HumanDistribution]]:
    items, dists = [], []
    for key, stem, answers in PEW_W41:
        item_id = f"opinionqa:w41:{key}"
        total = sum(p for _, p in answers)
        items.append(Item(
            item_id=item_id, source="opinionqa", assessment_type="survey", item_type="mcq",
            stem=stem, options=[o for o, _ in answers], eligible_mechanisms=["opinion_trap"],
            human_dist_key=f"{item_id}:human",
        ))
        dists.append(HumanDistribution(
            key=f"{item_id}:human", source="opinionqa",
            option_probs=[p / total for _, p in answers], n_respondents=PEW_W41_N,
        ))
    return items, dists

OPEN_TEXT = [
    Item(
        item_id="handwritten:pilot-open-1",
        source="handwritten",
        assessment_type="survey",
        item_type="open_text",
        stem="Describe a skill you learned outside of class, and how you learned it. "
        "Two or three sentences is plenty.",
        eligible_mechanisms=["canary"],
    ),
    Item(
        item_id="handwritten:pilot-open-2",
        source="handwritten",
        assessment_type="survey",
        item_type="open_text",
        stem="What is one change that would make online classes better for students? "
        "Explain why in two or three sentences.",
        eligible_mechanisms=["canary"],
    ),
]


def words(text: str) -> int:
    return len(text.split())


def readable_mcq(item: Item) -> bool:
    opts = item.options or []
    return (
        item.passage is None
        and len(opts) == 4
        and words(item.stem) <= MAX_STEM_WORDS
        and all(0 < words(o) <= MAX_OPTION_WORDS for o in opts)
        and not NEEDS_A_PICTURE.search(item.stem)
        and not any("�" in t for t in [item.stem, *opts])  # broken characters
        and DecoyMechanism().applies_to(item)
    )


def readable_opinion(item: Item) -> bool:
    opts = item.options or []
    return (
        item.item_type in ("mcq", "likert")
        and 3 <= len(opts) <= 5
        and words(item.stem) <= 30
        and all(words(o) <= MAX_OPTION_WORDS for o in opts)
        and item.human_dist_key is not None
    )


def pick(items: Iterable[Item], ok: Callable[[Item], bool], n: int, rng: random.Random,
         spread: Callable[[Item], str] | None = None) -> list[Item]:
    """n items passing `ok`, seeded; with `spread`, at most one per group while groups last."""
    pool = sorted((i for i in items if ok(i)), key=lambda i: i.item_id)
    rng.shuffle(pool)
    if spread is None:
        chosen = pool[:n]
    else:
        chosen, groups = [], set()
        for it in pool:
            if len(chosen) < n and spread(it) not in groups:
                chosen.append(it)
                groups.add(spread(it))
        chosen += [it for it in pool if it not in chosen][: n - len(chosen)]
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} usable items, need {n}")
    return chosen


def as_train(items: Iterable[Item], **update) -> list[Item]:
    return [it.model_copy(update={"split": "train", **update}) for it in items]


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(SEED)
    arc = pick(load_arc(args.arc), readable_mcq, N_ARC, rng)
    mmlu = pick(load_mmlu(args.mmlu, subjects=MMLU_SUBJECTS), readable_mcq, N_MMLU, rng,
                spread=lambda i: i.item_id.split(":")[1].rsplit("-", 1)[0])
    quiz = as_train(arc + mmlu, eligible_mechanisms=["decoy"])
    rng.shuffle(quiz)

    if args.opinionqa:
        opinion_items, dists = load_opinionqa(args.opinionqa)
        opinion = pick(opinion_items, readable_opinion, N_OPINION, rng,
                       spread=lambda i: i.item_id.split(":")[1].split("_")[0])
    else:
        print("No --opinionqa: using the 6 Pew Wave 41 questions from Pew's published topline.")
        opinion, dists = pew_w41()
    ipip_list, ipip_dists = load_ipip(args.ipip_data, args.ipip_codebook)
    dists = list(dists) + list(ipip_dists)
    ipip_items = {it.item_id.split(":")[1]: it for it in ipip_list}
    first_halves = [ipip_items[a.lower()] for a, _ in IPIP_PAIRS]
    second_halves = [ipip_items[b.lower()] for _, b in IPIP_PAIRS]
    # Survey order: the halves of a pair are kept apart, open text near the end.
    survey = as_train(opinion[:3] + first_halves + opinion[3:] + second_halves) + OPEN_TEXT

    quiz_ids = [it.item_id for it in quiz]
    survey_ids = [it.item_id for it in survey]
    # Human answer distributions for the survey items, so #39 can score the opinion trap.
    wanted = {it.human_dist_key for it in survey if it.human_dist_key}
    human = [d.model_dump() for d in dists if d.key in wanted]
    if len(human) != len(wanted):
        raise ValueError(f"missing human distributions: {wanted - {d['key'] for d in human}}")
    assessments = [
        Assessment(assessment_id="pilot-quiz", title="Quiz", assessment_type="quiz",
                   item_ids=quiz_ids),
        Assessment(assessment_id="pilot-survey", title="Survey", assessment_type="survey",
                   item_ids=survey_ids),
    ]
    return {
        "_note": "PILOT BANK (#35), built by scripts/build_pilot_bank.py. Git-ignored: it "
        "contains dataset questions. Rebuild from the raw files rather than editing by hand.",
        "flow": {"title": "Quiz and survey", "parts": ["pilot-quiz", "pilot-survey"]},
        "items": [it.model_dump(exclude_defaults=True) | {"split": "train"}
                  for it in quiz + survey],
        "assessments": [a.model_dump(exclude_none=True) for a in assessments],
        "human_distributions": human,  # the server ignores this; #39 reads it
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arc", required=True, type=Path)
    p.add_argument("--mmlu", required=True, type=Path)
    p.add_argument("--opinionqa", type=Path, help="OpinionQA human_resp dir (Pew)")
    p.add_argument("--ipip-data", required=True, type=Path)
    p.add_argument("--ipip-codebook", required=True, type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "server" / "pilot_items.json")
    args = p.parse_args(argv)
    bank = build(args)
    args.out.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n = {a["assessment_id"]: len(a["item_ids"]) for a in bank["assessments"]}
    print(f"Wrote {args.out}: {n}")


if __name__ == "__main__":
    main()
