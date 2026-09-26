"""Data shapes: the fixed content, a user submission, and a score breakdown."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .text import normalize

MAX_FIXED_WORDS = 100
HEADLINE_MAX_CHARS = 120
BODY_MIN_CHARS = 20
BODY_MAX_CHARS = 2000


class Stance(str, Enum):
    """The multi-choice property of a submission."""

    SUPPORT = "support"
    OPPOSE = "oppose"
    MIXED = "mixed"
    UNDECIDED = "undecided"


@dataclass(frozen=True)
class FixedContent:
    """The piece of content (<= 100 words) that users respond to, e.g. a news brief."""

    id: str
    title: str
    text: str

    def __post_init__(self) -> None:
        words = len(self.text.split())
        if words > MAX_FIXED_WORDS:
            raise ValueError(f"fixed content is {words} words; max is {MAX_FIXED_WORDS}")

    @property
    def embedding_text(self) -> str:
        return f"{self.title}\n\n{self.text}"


@dataclass(frozen=True)
class Submission:
    """User generated content: three discrete, user-provided properties."""

    headline: str
    body: str
    stance: Stance
    id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.headline, str) or not isinstance(self.body, str):
            raise ValueError("headline and body must be strings")
        # Normalise before validating, so invisible characters cannot pad a body past the minimum.
        headline = " ".join(normalize(self.headline).split())
        body = normalize(self.body)
        if not headline:
            raise ValueError("headline is required")
        if len(headline) > HEADLINE_MAX_CHARS:
            raise ValueError(f"headline exceeds {HEADLINE_MAX_CHARS} characters")
        if not BODY_MIN_CHARS <= len(body) <= BODY_MAX_CHARS:
            raise ValueError(f"body must be {BODY_MIN_CHARS}-{BODY_MAX_CHARS} characters")
        object.__setattr__(self, "headline", headline)
        object.__setattr__(self, "body", body)
        try:
            object.__setattr__(self, "stance", Stance(self.stance))
        except ValueError:
            raise ValueError(f"stance must be one of {[s.value for s in Stance]}") from None

    @property
    def text(self) -> str:
        """Free-text portion used for semantic and lexical comparison."""
        return f"{self.headline}\n\n{self.body}"

    @classmethod
    def from_dict(cls, d: dict) -> Submission:
        return cls(headline=d["headline"], body=d["body"], stance=d["stance"], id=d.get("id"))


@dataclass(frozen=True)
class Neighbor:
    id: str | None
    similarity: float


@dataclass(frozen=True)
class ScoreBreakdown:
    """Final reward plus every intermediate signal, so a score can be explained."""

    score: float  # final reward in [0, 1] = novelty * relevance_gate
    novelty: float  # [0, 1] = min(novelty signals) * product(modifier signals)
    semantic_novelty: float  # [0, 1], whole-text novelty vs. the corpus (robust z -> normal CDF)
    clause_novelty: float  # [0, 1], novelty of the most novel relevant, substantive clause
    raw_novelty: float  # uncalibrated whole-text distance to nearest neighbours
    stance_rarity: float  # [0, 1], 0 = most common stance in the corpus
    relevance: float  # [0, 1], margin / typical on-topic margin
    relevance_margin: float | None  # sim(topic) - sim(generic chatter); <= 0 means off-topic
    relevance_gate: float  # [0, 1], product of relevance signals; 0 = not rewarded
    near_duplicate_of: str | None  # id of the copied entry ("article" = the fixed content)
    nearest: list[Neighbor] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)  # every signal's value, by name
    admitted: bool | None = None  # set by submit(): whether it joined the reference corpus
    detail: dict[str, Any] = field(default_factory=dict)  # per-signal diagnostics
