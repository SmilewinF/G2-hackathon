"""Text normalisation and lightweight linguistic helpers shared by every signal.

Every submission goes through ``normalize`` first (it runs inside ``Submission``), so evasion
tricks like zero-width characters or Cyrillic look-alike letters cannot make a copy look new to
the lexical or semantic checks. The fixed content and the off-topic anchors are not normalised.
"""

from __future__ import annotations

import re
import unicodedata

_INVISIBLE = re.compile(r"[­᠎​-‏‪-‮⁠-⁤﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# Letters that render like Latin ones. Mapped only when the text is essentially Latin-script
# (see ``normalize``), so genuine Cyrillic/Greek text is left alone.
_CONFUSABLE_MAP = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    "α": "a", "ο": "o", "ρ": "p", "ν": "v", "ι": "i", "κ": "k", "τ": "t", "υ": "u", "Α": "A",
    "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
_CONFUSABLES = str.maketrans(_CONFUSABLE_MAP)
_LATIN_MAX_FOREIGN_SHARE = 0.2  # below this share of non-ASCII, non-look-alike letters, text counts as Latin-script

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being it
    its this that these those i we you they he she my our your their me us them not no so do does
    did will would can could should just than then there here what who whom which when where why
    how all any some more most other such only own same too very about into over after before up
    down out off also has have had his her him""".split()
)

_NON_WORD = re.compile(r"[^a-z0-9]+")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_CLAUSE_SPLIT = re.compile(r"[.!?;:]+\s*|,\s+(?:and\s+|but\s+)?|\s+(?:and|but)\s+")


def normalize(text: str) -> str:
    """Canonical form of user text: NFKC; invisible characters removed; ASCII control characters
    other than tab and newline turned into spaces; look-alike Cyrillic/Greek letters mapped to Latin
    when the text is essentially Latin-script; runs of spaces and tabs collapsed to one space; three
    or more newlines in a row cut to two; leading/trailing whitespace stripped."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub(" ", text)
    letters = [c for c in text if c.isalpha()]
    foreign = sum(1 for c in letters if not c.isascii() and c not in _CONFUSABLE_MAP)
    if letters and foreign / len(letters) < _LATIN_MAX_FOREIGN_SHARE:
        text = text.translate(_CONFUSABLES)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _stem(w: str) -> str:
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokens(text: str) -> list[str]:
    """Stemmed content words (for TF-IDF)."""
    return [_stem(w) for w in words(text) if len(w) > 2 and w not in STOPWORDS]


def shingles(text: str, n: int = 5) -> frozenset[str]:
    norm = _NON_WORD.sub(" ", text.lower()).strip()
    if len(norm) <= n:
        return frozenset({norm})
    return frozenset(norm[i : i + n] for i in range(len(norm) - n + 1))


def containment(a: frozenset[str], b: frozenset[str]) -> float:
    """Share of the smaller shingle set found in the larger one (1.0 = one text contains the other)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def clauses(text: str, min_words: int = 4) -> list[str]:
    """Split into clause-sized units; fragments shorter than ``min_words`` merge into the previous one."""
    out: list[str] = []
    for part in (p.strip() for p in _CLAUSE_SPLIT.split(text)):
        if not part:
            continue
        if out and len(part.split()) < min_words:
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def is_substantive(clause: str) -> bool:
    """Reject clauses that carry no statement: keyword lists and word repetition.

    Natural English sentences are ~30-50% function words; a bag of topic keywords is ~0%.
    """
    ws = words(clause)
    if len(ws) < 3:
        return False
    if len(ws) >= 6:
        if sum(w in STOPWORDS for w in ws) / len(ws) < 0.1:
            return False
        if len(set(ws)) / len(ws) < 0.5:
            return False
    return True


def distinct_share(text: str) -> float:
    ws = words(text)
    return len(set(ws)) / len(ws) if ws else 0.0
