"""Where a replayed respondent's outcomes come from: real rows of the outcome table.

Owner: Arav (step #27).

A simulated respondent is built from one *donor session*: a logged session (one model or person
taking the assessment once) from the same respondent group. Whenever the router asks "what did
this person get on check M for item I?", the answer comes from that session's own row. That
keeps the real correlations between one respondent's checks (a model that falls for one decoy
tends to fall for the next), which independent draws per check would throw away.

When the donor session has no row for (item, mechanism), we fall back in two steps, and count
each so a result can say how much of it was real:
  1. fallback:  pool every session of the same group that does have that pair, and draw one.
  2. synthetic: no logged row at all (e.g. no human data for MMLU decoys yet). Draw from the
     failure map's p_ai / p_human for that cell.
If neither exists, `Donor.outcome` raises NoObservation and the replay skips the check.

Respondent groups:
  AI:    (model, attack_mode, attacker_level)   e.g. gpt-mini / copy_paste / L0
  human: data_origin                            e.g. public_human, pilot, dry_run
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from adr.contracts import FailureMap, Item, NoObservation

Key = tuple[str, str]  # (item_id, mechanism)


@dataclass(frozen=True)
class Group:
    """One kind of respondent. Two respondents in the same group are exchangeable."""

    kind: str  # "ai" or "human"
    model: str | None = None
    attack_mode: str | None = None
    attacker_level: str | None = None
    data_origin: str | None = None

    @classmethod
    def ai(cls, model: str, attack_mode: str, attacker_level: str) -> Group:
        return cls("ai", model=model, attack_mode=attack_mode, attacker_level=attacker_level)

    @classmethod
    def human(cls, data_origin: str) -> Group:
        return cls("human", data_origin=data_origin)

    @property
    def label(self) -> str:
        if self.kind == "ai":
            return f"ai:{self.model}:{self.attack_mode}:{self.attacker_level}"
        return f"human:{self.data_origin}"


def group_of(row: dict) -> Group:
    if row["respondent_kind"] == "ai":
        return Group.ai(row["model"], row["attack_mode"], row["attacker_level"])
    return Group.human(row["data_origin"])


@dataclass
class Donor:
    """One simulated respondent: answers from one donor session, with counted fallbacks."""

    group: Group
    session_id: str | None  # the donor session; None = the group has no logged sessions
    _own: dict[Key, list[str]]
    _pooled: dict[Key, list[str]]
    _items: dict[str, Item]
    _fmap: FailureMap | None
    _rng: random.Random
    fallbacks: int = 0  # answered from another session of the same group
    synthetic: int = 0  # answered from the failure map: no logged row anywhere
    missing: int = 0    # no row and no map cell: NoObservation was raised
    sources: list[tuple[str, str, str]] = field(default_factory=list)  # (item, mech, source)

    @property
    def kind(self) -> str:
        return self.group.kind

    def outcome(self, item_id: str, mechanism: str) -> str:
        key = (item_id, mechanism)
        if key in self._own:
            return self._draw(key, self._own[key], "session")
        if key in self._pooled:
            self.fallbacks += 1
            return self._draw(key, self._pooled[key], "fallback")
        cell = None
        if self._fmap is not None and item_id in self._items:
            cell = self._fmap.lookup(
                mechanism,
                self._items[item_id].item_type,
                self.group.attack_mode,
                self.group.attacker_level,
            )
        if cell is not None:
            self.synthetic += 1
            probs = cell.p_ai if self.kind == "ai" else cell.p_human
            labels = list(probs)
            self.sources.append((item_id, mechanism, "synthetic"))
            return self._rng.choices(labels, weights=[probs[k] for k in labels], k=1)[0]
        self.missing += 1
        raise NoObservation(f"no logged row and no map cell for {mechanism} on {item_id}")

    def _draw(self, key: Key, outcomes: list[str], source: str) -> str:
        self.sources.append((*key, source))
        return outcomes[0] if len(outcomes) == 1 else self._rng.choice(outcomes)


class Donors:
    """The outcome table indexed by (group, session, item_id, mechanism)."""

    def __init__(
        self,
        table: pd.DataFrame,
        items: dict[str, Item] | None = None,
        fallback_map: FailureMap | None = None,
        extra_groups: tuple[Group, ...] = (),
    ):
        self.items = items or {}
        self.fallback_map = fallback_map
        self._sessions: dict[Group, dict[str, dict[Key, list[str]]]] = defaultdict(dict)
        self._pooled: dict[Group, dict[Key, list[str]]] = defaultdict(lambda: defaultdict(list))
        # Plain rows (outcome "none") carry no evidence, so they are never needed. Sorting first
        # makes the index, and so every draw, independent of the order rows were read in.
        rows = table[table["mechanism"].notna()].sort_values(
            ["session_id", "item_id", "mechanism", "variant_id", "outcome"]
        )
        for row in rows.to_dict(orient="records"):
            g = group_of(row)
            key = (row["item_id"], row["mechanism"])
            self._sessions[g].setdefault(row["session_id"], {}).setdefault(key, []).append(
                row["outcome"]
            )
            self._pooled[g][key].append(row["outcome"])
        for g in extra_groups:  # e.g. a human group with no rows: every draw is synthetic
            self._sessions.setdefault(g, {})

    def groups(self, kind: str | None = None) -> list[Group]:
        out = [g for g in self._sessions if kind is None or g.kind == kind]
        return sorted(out, key=lambda g: g.label)

    def sessions(self, group: Group) -> list[str]:
        return sorted(self._sessions.get(group, {}))

    def respondent(self, group: Group, seed: int) -> Donor:
        """A simulated respondent of `group`, copying one logged session chosen by `seed`."""
        rng = random.Random(f"{seed}:{group.label}")
        sessions = self.sessions(group)
        session_id = rng.choice(sessions) if sessions else None
        own = self._sessions[group][session_id] if session_id is not None else {}
        return Donor(
            group=group,
            session_id=session_id,
            _own=own,
            _pooled=dict(self._pooled.get(group, {})),
            _items=self.items,
            _fmap=self.fallback_map,
            _rng=rng,
        )
