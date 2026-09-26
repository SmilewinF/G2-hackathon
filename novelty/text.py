"""Text normalisation and lightweight linguistic helpers shared by every signal.

Every submission goes through ``normalize`` first (it runs inside ``Submission``), so evasion
tricks like zero-width characters or Cyrillic look-alike letters cannot make a copy look new to
the lexical or semantic checks. The fixed content and the off-topic anchors are not normalised.
"""

from __future__ import annotations

import re
import unicodedata

# Invisible characters that are not Unicode format characters (category Cf, removed separately):
# combining grapheme joiner, Hangul fillers, Khmer inherent vowels, Mongolian variation selectors,
# braille blank, variation selectors. A fixed list of Cf ranges missed the bidi isolates
# (U+2066-2069) and tag characters, so a copy with one after every third letter scored 0.78.
_INVISIBLE = re.compile(
    "[͏ᅟᅠ឴឵᠋-᠍᠏⠀ㅤ︀-️ﾠ\U000e0100-\U000e01ef]"
)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# Letters that render like Latin ones. Mapped only when the text is essentially Latin-script
# (see ``normalize``), so genuine Cyrillic/Greek/Armenian text is left alone.
_CONFUSABLE_MAP = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w", "ү": "y", "А": "A", "В": "B",
    "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "І": "I", "Ј": "J", "Ѕ": "S",
    "α": "a", "ο": "o", "ρ": "p", "ν": "v", "ι": "i", "κ": "k", "τ": "t", "υ": "u", "ϲ": "c",
    "ϳ": "j", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "օ": "o", "ս": "u", "հ": "h", "ո": "n", "ց": "g", "Օ": "O", "Ս": "U",  # Armenian
    "ı": "i", "ȷ": "j", "ɑ": "a", "ɡ": "g",  # Latin letters outside ASCII
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

# Words that evaluate or refer to the proposal without saying anything specific about it.
GENERIC_WORDS = frozenset(
    """love like liked great good bad nice awesome terrible horrible amazing wonderful excellent
    fantastic awful best worst fine happy glad sad agree disagree support oppose hate dislike idea
    plan project proposal decision thing stuff way think thought feel felt opinion really totally
    definitely yes yeah wow thanks thank please sure just well much""".split()
)
# Inflections of a generic word are generic too: (suffix, what replaces it) → "loved" → "love",
# "agreed" → "agree", "ideas" → "idea", "loving" → "love", "greatly" → "great".
_INFLECTIONS = (("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"), ("d", ""), ("es", ""), ("s", ""), ("ly", ""))
_SPECIFIC_INFLECTIONS = frozenset({"fines", "fined", "goods", "wells"})  # "parking fines" is a concrete point


def is_generic(word: str) -> bool:
    """``word`` (lower case) is a generic word or a regular inflection of one. Checked on the word
    itself: the stemmed tokens used to be compared with this list, so "loved" became "lov", matched
    nothing, and "I loved this plan, liked the idea, agreed, supported" scored 0.86 as specific."""
    if word in GENERIC_WORDS:
        return True
    if word in _SPECIFIC_INFLECTIONS:
        return False
    return any(word.endswith(suffix) and word[: -len(suffix)] + repl in GENERIC_WORDS for suffix, repl in _INFLECTIONS)


_NON_WORD = re.compile(r"[^a-z0-9]+")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_CLAUSE_SPLIT = re.compile(r"[.!?;:]+\s*|,\s+(?:and\s+|but\s+)?|\s+(?:and|but)\s+")


def normalize(text: str) -> str:
    """Canonical form of user text: NFKC; invisible characters (every format character, plus the
    invisible marks and fillers in ``_INVISIBLE``) and lone surrogates removed; ASCII control
    characters other than tab and newline turned into spaces; look-alike Cyrillic/Greek/Armenian
    letters mapped to Latin when the text is essentially Latin-script; runs of spaces and tabs
    collapsed to one space; three or more newlines in a row cut to two; leading/trailing whitespace
    stripped."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    if not text.isascii():
        text = "".join(c for c in text if unicodedata.category(c) not in ("Cf", "Cs"))
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


def fold(text: str) -> str:
    """Lower-case text with accents dropped ("ė" → "e"), for matching: without it an accented copy
    ("thė gȧragė") splits into unseen fragments and reads as new to the shingle and TF-IDF checks."""
    if text.isascii():
        return text.lower()
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def stem(w: str) -> str:
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokens(text: str) -> list[str]:
    """Stemmed content words, accents folded (for TF-IDF)."""
    return [stem(w) for w in words(fold(text)) if len(w) > 2 and w not in STOPWORDS]


def shingles(text: str, n: int = 5) -> frozenset[str]:
    norm = _NON_WORD.sub(" ", fold(text)).strip()
    if len(norm) <= n:
        return frozenset({norm})
    return frozenset(norm[i : i + n] for i in range(len(norm) - n + 1))


def containment(a: frozenset[str], b: frozenset[str]) -> float:
    """Share of the smaller shingle set found in the larger one (1.0 = one text contains the other)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def clause_spans(text: str, min_words: int = 4) -> list[tuple[str, int, int]]:
    """``clauses`` plus where each one lies in ``text``: (clause, start, end). A merged clause spans
    from its first fragment's start to its last fragment's end."""
    out: list[tuple[str, int, int]] = []
    starts = [0, *(m.end() for m in _CLAUSE_SPLIT.finditer(text))]
    ends = [*(m.start() for m in _CLAUSE_SPLIT.finditer(text)), len(text)]
    for start, end in zip(starts, ends, strict=True):
        raw = text[start:end]
        part = raw.strip()
        if not part:
            continue
        start += len(raw) - len(raw.lstrip())
        end = start + len(part)
        if out and len(part.split()) < min_words:
            prev, prev_start, _ = out[-1]
            out[-1] = (f"{prev} {part}", prev_start, end)
        else:
            out.append((part, start, end))
    return out


def clauses(text: str, min_words: int = 4) -> list[str]:
    """Split into clause-sized units; fragments shorter than ``min_words`` merge into the previous one."""
    return [c for c, _, _ in clause_spans(text, min_words)]


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


def specific_words(text: str) -> set[str]:
    """Stems of the distinct specific content words: not function words, not generic ones."""
    return {stem(w) for w in words(fold(text)) if len(w) > 2 and w not in STOPWORDS and not is_generic(w)}


def distinct_share(text: str) -> float:
    ws = words(text)
    return len(set(ws)) / len(ws) if ws else 0.0
