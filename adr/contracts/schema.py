"""Item schema: the shared format for every survey or quiz question.

Owners: Harshini + Arav. DRAFT until the team freezes it; after that, changes need all three.

Every loader (Archana) returns `list[Item]`; the front end (Harshini) renders any `Item`;
the harness (Arav) sends `Item`s to models. Because these are Pydantic models, a loader that
produces a bad item fails immediately with a clear error, instead of breaking someone else's
code later.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Source = Literal[
    "opinionqa",
    "gss",
    "ipip",
    "asurre_nature_phd_2019",
    "asurre_springer_social_2017",
    "asurre_osmi_2017_18",
    "mmlu",
    "arc",
    "handwritten",  # small test items written by the team before real loaders exist
]
# "test" = a quiz with right answers and a time limit; "assignment" = written answers.
AssessmentType = Literal["survey", "quiz", "test", "assignment"]
ItemType = Literal["mcq", "likert", "ranking", "numeric", "open_text"]
AttackMode = Literal["copy_paste", "doc_upload", "browser_agent"]
MechanismName = Literal["decoy", "opinion_trap", "canary", "glitch_token", "behaviour", "style"]
Split = Literal["train", "heldout_survey"]

# Item types whose answer is one or more option indexes.
OPTION_TYPES: frozenset[str] = frozenset({"mcq", "likert", "ranking"})
# Assessment types whose items can have a right answer.
ANSWER_KEY_TYPES: frozenset[str] = frozenset({"quiz", "test"})


class Item(BaseModel):
    """One question, exactly as a human would see it, plus what checks it can carry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=3, description='"<source>:<native id>", unique and stable')
    source: Source
    assessment_type: AssessmentType
    item_type: ItemType
    stem: str = Field(min_length=1, description="Question text as shown to a human")
    passage: str | None = Field(
        default=None,
        min_length=1,
        description="Reading-comprehension text shown before the stem; the stem asks about it",
    )
    options: list[str] | None = Field(
        default=None, description="MCQ / Likert / ranking labels, in display order"
    )
    correct_option: int | None = Field(
        default=None, description="Index into options; quiz and test items only"
    )
    reverse_keyed_with: str | None = Field(
        default=None, description="IPIP: item_id of the paired reverse-keyed item"
    )
    eligible_mechanisms: list[MechanismName] = Field(default_factory=list)
    human_dist_key: str | None = Field(
        default=None, description="Key into the human answer distribution store"
    )
    split: Split = "train"

    @model_validator(mode="after")
    def _check_consistency(self) -> Item:
        if ":" not in self.item_id:
            raise ValueError(f'item_id must look like "<source>:<native id>", got {self.item_id!r}')
        if self.item_type in OPTION_TYPES:
            if not self.options or len(self.options) < 2:
                raise ValueError(f"{self.item_id}: {self.item_type} items need at least 2 options")
        elif self.options:
            raise ValueError(f"{self.item_id}: {self.item_type} items must not have options")
        if self.correct_option is not None:
            if self.assessment_type not in ANSWER_KEY_TYPES:
                raise ValueError(
                    f"{self.item_id}: only quiz or test items have a correct_option "
                    f"(this one is a {self.assessment_type})"
                )
            if not self.options or not (0 <= self.correct_option < len(self.options)):
                raise ValueError(f"{self.item_id}: correct_option is not a valid option index")
        if "decoy" in self.eligible_mechanisms and self.correct_option is None:
            raise ValueError(f"{self.item_id}: decoys need an item with a correct_option")
        if len(set(self.eligible_mechanisms)) != len(self.eligible_mechanisms):
            raise ValueError(f"{self.item_id}: eligible_mechanisms has duplicates")
        return self


class Assessment(BaseModel):
    """An ordered set of items shown together (one survey, quiz, test or assignment)."""

    model_config = ConfigDict(extra="forbid")

    assessment_id: str = Field(min_length=1)
    title: str
    assessment_type: AssessmentType
    item_ids: list[str] = Field(min_length=1, description="Display order; the router may reorder")
    time_limit_s: int | None = Field(
        default=None, gt=0, description="Time allowed for the whole assessment, in seconds"
    )

    @model_validator(mode="after")
    def _unique_items(self) -> Assessment:
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError(f"{self.assessment_id}: item_ids has duplicates")
        return self


class HumanDistribution(BaseModel):
    """PROPOSED (not in the Oct 17 draft): how often real people chose each option.

    Opinion traps need this to tell an unusual answer from a common one. Archana's loaders
    would write one per item, keyed by `Item.human_dist_key`. Raise at the next sync.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    source: Source
    option_probs: list[float] = Field(description="Same order as Item.options; sums to ~1")
    n_respondents: int = Field(ge=1)

    @model_validator(mode="after")
    def _sums_to_one(self) -> HumanDistribution:
        total = sum(self.option_probs)
        if any(p < 0 for p in self.option_probs) or abs(total - 1.0) > 1e-3:
            raise ValueError(f"{self.key}: option_probs must be non-negative and sum to 1")
        return self
