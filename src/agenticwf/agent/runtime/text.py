"""Words for matching by words: tool and skill search (BM25) and memory recall."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by can do for from has have how i in is it me my of on or our please "
                  "should that the this to use using want we what when which will with you your".split())


def terms(text: str, unique: bool = True):
    """Lowercase words without stop words, plural `s` cut off: a set, or (unique=False) every occurrence."""
    found = []
    for word in _WORD.findall(text.lower()):
        if word in _STOP:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        found.append(word)
    return set(found) if unique else found
