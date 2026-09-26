"""Text preparation: turn what a user typed into what the signals should compare.

Misspellings are free novelty. An unseen token looks new to TF-IDF and shifts the embedding, so
"Withot the garaje peple cant park" scored 0.86 while the same stock take spelled correctly
scored 0.02. ``EnglishPreparer`` therefore, per sentence:

1. decides whether the sentence is English, from its function words ("the", "is" vs "la", "de",
   "und") and script, which typos do not disturb. Non-English sentences are left untouched and
   flagged ``foreign``; the English-only signals exclude them instead of "correcting" them into
   nonsense English ("La calle Elm se inunda" → "La call Elm se inundate");
2. expands texting shorthand (u, ppl, tbh, ...);
3. spell-corrects unknown words, preferring words from the article and corpus (so "pavment" →
   "pavement", not "payment"), and leaving proper nouns, numbers and domain words alone.

The display text is never changed; only the analysis sees the prepared version. A different
language or domain plugs in by implementing ``TextPreparer``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Protocol

from .text import STOPWORDS, clause_spans, clauses, fold, is_substantive, words

log = logging.getLogger(__name__)

SLANG = {
    "u": "you", "ur": "your", "r": "are", "ppl": "people", "pls": "please", "plz": "please",
    "thx": "thanks", "tbh": "to be honest", "imo": "in my opinion", "imho": "in my opinion",
    "idk": "I do not know", "bc": "because", "cuz": "because", "coz": "because", "b4": "before",
    "n": "and", "w/": "with", "w/o": "without", "gonna": "going to", "wanna": "want to",
    "gotta": "have to", "shud": "should", "cud": "could", "wud": "would", "thru": "through",
    "tho": "though", "st": "street", "govt": "government", "lol": "", "lmao": "", "omg": "",
}

# Frequent function words of other Latin-script languages (es, fr, de, it, pt), minus any that
# are also English words ("die", "son", "pour", "come", "per", ...).
FOREIGN_FUNCTION_WORDS = frozenset(
    """de en y el la los las del que un una unos por para con es se su sus lo al como pero muy sobre este
    esta ser hay nos ya donde cuando porque tambien le les des du et est une dans qui pas sur au aux
    avec ce cette sont mais je nous vous ils elle sa ses leur merci der das und ist nicht ein eine mit
    auf dem zu sich auch sind wir ich sie werden oder aber wenn noch nur dass il di che gli della
    delle sono anche questo questa molto os uma com mais pelo pela muito isso""".split()
)

_SENTENCE = re.compile(r"[^.!?;\n]+[.!?;\n]*")
_TOKEN = re.compile(r"[^\W\d_][\w'/]*")  # Unicode-aware, so "debería" stays one token
_FOREIGN_SCRIPT_SHARE = 0.3  # more non-ASCII letters than this share and a sentence is foreign outright


@dataclass(frozen=True)
class PreparedText:
    """What the signals compare. ``clauses``, ``substantive`` and ``foreign`` are aligned: one
    entry per clause of ``body``."""

    headline: str
    body: str
    clauses: tuple[str, ...]
    substantive: tuple[bool, ...]  # sentence-like English clause that makes a statement
    foreign: tuple[bool, ...]  # clause is not in a supported language
    corrections: tuple[tuple[str, str], ...]  # (original, replacement) pairs, for transparency
    english_body: str = ""  # prepared body without its non-English sentences and clauses
    english_headline: str = ""  # prepared headline without its non-English sentences

    @property
    def text(self) -> str:
        return f"{self.headline}\n\n{self.body}"

    @property
    def analysis_text(self) -> str:
        """What similarity is computed on: the English part of the prepared headline and body, so
        untranslated words cannot make a text look "unusual", i.e. novel. If no body clause is
        English, the whole body is kept (the quality check then gives it nothing)."""
        return f"{self.english_headline}\n\n{self.english_body or self.body}"


class TextPreparer(Protocol):
    def prepare(self, headline: str, body: str) -> PreparedText: ...


def _without(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """``text`` minus the character ranges ``spans`` (overlaps allowed), stripped. With no spans
    it is ``text`` unchanged, so English-only text is analysed exactly as typed (and prepared)."""
    out, pos = [], 0
    for start, end in sorted((max(s, 0), max(e, 0)) for s, e in spans):
        if start > pos:
            out.append(text[pos:start])
        pos = max(pos, end)
    out.append(text[pos:])
    return "".join(out).strip()


def looks_foreign(text: str) -> bool:
    """True if ``text`` is not English: more than ``_FOREIGN_SCRIPT_SHARE`` of its letters are
    non-ASCII, or it has at least two foreign function words and more of them than English stopwords."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(not c.isascii() for c in letters) / len(letters) > _FOREIGN_SCRIPT_SHARE:
        return True  # another script or heavily accented text (look-alike letters were already mapped)
    # Whole Unicode words: the ASCII ``words`` split "debería" into "deber" + "a" and counted an
    # English "a"; accents are folded only to match the function-word list ("también" → "tambien").
    ws = [w.lower() for w in _TOKEN.findall(text)]
    foreign = sum(fold(w) in FOREIGN_FUNCTION_WORDS for w in ws)
    return foreign >= 2 and foreign > sum(w in STOPWORDS for w in ws)


@lru_cache(maxsize=1)
def _spellchecker():
    """Single-edit corrections only: most typos are one edit away, and an unknown word with no
    close match is more often a real term ("bioswales") than a typo. Also 100x faster.
    Returns None (spelling correction disabled, everything else still works) if unavailable."""
    try:
        from spellchecker import SpellChecker

        return SpellChecker(distance=1)
    except Exception as e:  # missing package or unreadable dictionary
        log.warning("spelling correction disabled: pyspellchecker unavailable (%s)", e)
        return None


class EnglishPreparer:
    def __init__(self, domain_texts: Iterable[str] = ()) -> None:
        self._sp = _spellchecker()
        self.domain = frozenset(w for t in domain_texts for w in words(t))
        self._cache: dict[str, str] = {}

    # ------------------------------------------------------------------ word level

    def _correct_word(self, word: str) -> str:
        lower = word.lower()
        if lower in SLANG:
            return SLANG[lower]
        if self._sp is None or len(lower) < 3 or lower in self.domain or not self._sp.unknown([lower]):
            return word
        if lower in self._cache:
            return self._cache[lower]
        # sorted(): candidates() returns a set, whose iteration order changes with per-process string
        # hashing, so ties in word frequency would otherwise be corrected differently on every run.
        candidates = sorted(self._sp.candidates(lower) or ())
        in_domain = [c for c in candidates if c in self.domain]
        pool = in_domain or [c for c in candidates if c != lower]
        best = max(pool, key=self._sp.word_usage_frequency, default=word)
        self._cache[lower] = best
        return best

    def _correct_sentence(self, sentence: str) -> tuple[str, list[tuple[str, str]]]:
        fixes: list[tuple[str, str]] = []
        start_of_sentence = True

        def repl(m: re.Match) -> str:
            nonlocal start_of_sentence
            tok = m.group(0)
            first, start_of_sentence = start_of_sentence, False
            if not tok.isascii():
                return tok  # accented / non-Latin words are never "corrected" into English
            if tok[0].isupper() and not first and tok.lower() not in SLANG:
                return tok  # a capitalised word mid-sentence is probably a proper noun
            new = self._correct_word(tok)
            if new != tok and new.lower() != tok.lower():
                fixes.append((tok, new))
                return new
            return tok

        out = _TOKEN.sub(repl, sentence)
        return re.sub(r"\s{2,}", " ", out), fixes

    # ------------------------------------------------------------------ text level

    def _prepare_prose(self, text: str) -> tuple[str, list[tuple[int, int]], list[tuple[str, str]]]:
        """Returns (prepared text, (start, end) of each foreign sentence in it, corrections)."""
        parts, foreign_spans, fixes, pos = [], [], [], 0
        for sentence in _SENTENCE.findall(text) or [text]:
            foreign = looks_foreign(sentence)
            # Correct only sentences with no foreign function words at all: the Spanish words of a
            # code-mixed sentence must not become "deer"/"tender".
            if not foreign and not any(w in FOREIGN_FUNCTION_WORDS for w in words(sentence)):
                sentence, f = self._correct_sentence(sentence)
                fixes.extend(f)
            if foreign:
                foreign_spans.append((pos, pos + len(sentence)))
            parts.append(sentence)
            pos += len(sentence)
        joined = "".join(parts)
        lead = len(joined) - len(joined.lstrip())
        return joined.strip(), [(s - lead, e - lead) for s, e in foreign_spans], fixes

    def prepare(self, headline: str, body: str) -> PreparedText:
        h, h_foreign, hf = self._prepare_prose(headline)
        b, b_foreign, bf = self._prepare_prose(body)
        spans = clause_spans(b)
        # A clause is foreign if it looks foreign itself or lies in a sentence that does: code-mixed
        # sentences ("Elm Street se inunda every spring, so the park debería tener rain gardens")
        # pass the test clause by clause, and their Spanish words then read as novelty (0.90).
        foreign = tuple(looks_foreign(c) or any(s < fe and e > fs for fs, fe in b_foreign) for c, s, e in spans)
        cl = tuple(c for c, _, _ in spans)
        substantive = tuple(not f and is_substantive(c) for c, f in zip(cl, foreign))
        foreign_clauses = [(s, e) for (_, s, e), f in zip(spans, foreign) if f]
        return PreparedText(headline=h, body=b, clauses=cl, substantive=substantive, foreign=foreign,
                            corrections=tuple(hf + bf), english_body=_without(b, [*b_foreign, *foreign_clauses]),
                            english_headline=_without(h, h_foreign))


class PassthroughPreparer:
    """No language handling or correction (useful for tests with toy embedders)."""

    def prepare(self, headline: str, body: str) -> PreparedText:
        cl = tuple(clauses(body))
        return PreparedText(headline=headline, body=body, clauses=cl,
                            substantive=tuple(is_substantive(c) for c in cl),
                            foreign=tuple(False for _ in cl), corrections=(), english_body=body,
                            english_headline=headline)
