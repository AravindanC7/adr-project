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

Same-type draws (optional, `type_match_kinds`): when a donor session never saw the replayed item
at all (e.g. AI runs on the MMLU/ARC banks replayed on the pilot bank), it can answer check M on
item I from ITS OWN outcomes for mechanism M on items of the same item_type, drawn without
replacement. That keeps the session's real behaviour and its correlations; it assumes items of
one type are exchangeable, the same assumption as the failure map's per-item-type cells. Counted
as `same_type`.

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
    # A prompt variant within a level (e.g. "L2g" = L2-generic, adr/replay/composite.py). It
    # separates the group; map lookups still use attacker_level.
    attacker_variant: str | None = None

    @classmethod
    def ai(
        cls, model: str, attack_mode: str, attacker_level: str, attacker_variant: str | None = None
    ) -> Group:
        variant = None if attacker_variant in (None, attacker_level) else attacker_variant
        return cls(
            "ai",
            model=model,
            attack_mode=attack_mode,
            attacker_level=attacker_level,
            attacker_variant=variant,
        )

    @classmethod
    def human(cls, data_origin: str) -> Group:
        return cls("human", data_origin=data_origin)

    @property
    def label(self) -> str:
        if self.kind == "ai":
            level = self.attacker_variant or self.attacker_level
            return f"ai:{self.model}:{self.attack_mode}:{level}"
        return f"human:{self.data_origin}"


def group_of(row: dict) -> Group:
    if row["respondent_kind"] == "ai":
        variant = row.get("attacker_variant")
        return Group.ai(
            row["model"],
            row["attack_mode"],
            row["attacker_level"],
            variant if isinstance(variant, str) and variant else None,
        )
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
    _own_type: dict[Key, list[str]] = field(default_factory=dict)  # (item_type, mech) -> outcomes
    # (item_type, mech) -> outcomes of the whole group, drawn with replacement when neither this
    # session nor the group has the exact item (resampled honest sessions, adr/replay/crossfit.py)
    _pooled_type: dict[Key, list[str]] = field(default_factory=dict)
    fallbacks: int = 0  # answered from another session of the same group
    same_type: int = 0  # answered from this session's own rows on another item of the same type
    synthetic: int = 0  # answered from the failure map: no logged row anywhere
    missing: int = 0    # no row and no map cell: NoObservation was raised
    sources: list[tuple[str, str, str]] = field(default_factory=list)  # (item, mech, source)
    _type_queue: dict[Key, list[str]] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.group.kind

    def outcome(self, item_id: str, mechanism: str) -> str:
        key = (item_id, mechanism)
        if key in self._own:
            return self._draw(key, self._own[key], "session")
        item_type = self._items[item_id].item_type if item_id in self._items else None
        if (item_type, mechanism) in self._own_type:
            self.same_type += 1
            self.sources.append((item_id, mechanism, "same_type"))
            return self._next_same_type((item_type, mechanism))
        if key in self._pooled:
            self.fallbacks += 1
            return self._draw(key, self._pooled[key], "fallback")
        if (item_type, mechanism) in self._pooled_type:
            self.fallbacks += 1
            return self._draw(key, self._pooled_type[(item_type, mechanism)], "fallback_type")
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

    def _next_same_type(self, key: Key) -> str:
        """Without replacement: shuffle this session's outcomes once, take them in turn, and
        reshuffle only if the replayed assessment needs more than the session has."""
        queue = self._type_queue.setdefault(key, [])
        if not queue:
            queue.extend(self._own_type[key])
            self._rng.shuffle(queue)
        return queue.pop()

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
        type_match_kinds: tuple[str, ...] = (),
    ):
        self.items = items or {}
        self.type_match_kinds = type_match_kinds
        self._session_types: dict[Group, dict[str, dict[Key, list[str]]]] = defaultdict(dict)
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
            if g.kind in type_match_kinds:
                tkey = (row["item_type"], row["mechanism"])
                self._session_types[g].setdefault(row["session_id"], {}).setdefault(
                    tkey, []
                ).append(row["outcome"])
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
        return self._donor(group, session_id, rng)

    def respondent_from(self, group: Group, session_id: str, seed: int) -> Donor:
        """A simulated respondent copying exactly `session_id` (cross-fitting replays every
        held-out person once, rather than sampling)."""
        if session_id not in self._sessions.get(group, {}):
            raise KeyError(f"no session {session_id!r} in group {group.label}")
        return self._donor(group, session_id, random.Random(f"{seed}:{group.label}:{session_id}"))

    def _donor(self, group: Group, session_id: str | None, rng: random.Random) -> Donor:
        own = self._sessions[group][session_id] if session_id is not None else {}
        own_type = (
            self._session_types.get(group, {}).get(session_id, {}) if session_id is not None else {}
        )
        return Donor(
            group=group,
            session_id=session_id,
            _own=own,
            _pooled=dict(self._pooled.get(group, {})),
            _items=self.items,
            _fmap=self.fallback_map,
            _rng=rng,
            _own_type=own_type,
        )
