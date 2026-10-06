"""Cleaning text that came from outside, before the model reads it.

Anything fetched from the web is *untrusted*: a page can say "ignore your instructions and ...", and a model may
obey. Telling the model "treat this as reference material" (the tool descriptions do) is one layer of defence. This is
a second, independent one, aimed at two tricks that hide an instruction from the person reading but not from the model:

* **Invisible characters.** Zero-width spaces, direction overrides and similar characters take no room on screen, so an
  instruction can be tucked between visible words, or a trigger word split so that a filter misses it. They are removed.
* **Lookalike letters.** A Cyrillic "o" and a Latin "o" look identical but are different characters, so a filter looking
  for the word "ignore" misses a spelling with the Cyrillic one, while a model still reads it as the same word. Lookalikes
  are replaced by the ordinary Latin letter.

One deliberate difference from the .NET editions: they replace *every* lookalike letter, which also turns genuine Russian
or Greek text into nonsense. Here a lookalike is only replaced inside a word that **also contains plain Latin letters**
(the signature of a disguised English word), so a page written in Russian or Greek reads normally.

The tables below are written as **code point numbers** (``0x200B``), not as the characters themselves: an invisible
character pasted into source code is invisible to the next reader too, and editors and formatters are free to change it.

Python ideas used here:

* ``str.translate`` with a table: map code points to a replacement, or to ``None`` to delete them, in one pass.
* ``chr(number)`` turns a code point into the character, ``ord(char)`` the other way.
* ``re.sub`` with a function: decide what to do for each match.
"""

from __future__ import annotations

import re

# Characters that take no space on screen: zero-width space and joiners, left/right marks, direction overrides, the word
# joiner and invisible operators, the byte order mark, and the soft hyphen.
ZERO_WIDTH_CODE_POINTS = (
    *range(
        0x200B, 0x2010
    ),  # zero-width space, non-joiner, joiner, left-to-right and right-to-left marks
    *range(0x202A, 0x202F),  # embeddings and overrides
    *range(0x2060, 0x2065),  # word joiner and invisible operators
    0xFEFF,  # byte order mark / zero-width no-break space
    0x00AD,  # soft hyphen
)
_DELETE = dict.fromkeys(ZERO_WIDTH_CODE_POINTS)

# Greek and Cyrillic letters that look like a Latin one: (code point, the Latin letter it imitates).
_LOOKALIKES = (
    # Greek capitals
    (0x0391, "A"), (0x0392, "B"), (0x0395, "E"), (0x0396, "Z"), (0x0397, "H"), (0x0399, "I"),
    (0x039A, "K"), (0x039C, "M"), (0x039D, "N"), (0x039F, "O"), (0x03A1, "P"), (0x03A4, "T"),
    (0x03A5, "Y"), (0x03A7, "X"),
    # Cyrillic capitals
    (0x0410, "A"), (0x0412, "B"), (0x0415, "E"), (0x041A, "K"), (0x041C, "M"), (0x041D, "H"),
    (0x041E, "O"), (0x0420, "P"), (0x0421, "C"), (0x0422, "T"), (0x0425, "X"),
    # Cyrillic lowercase
    (0x0430, "a"), (0x0435, "e"), (0x043E, "o"), (0x0440, "p"), (0x0441, "c"), (0x0445, "x"), (0x0443, "y"),
)  # fmt: skip
HOMOGLYPHS = {chr(code): latin for code, latin in _LOOKALIKES}
_LOOKALIKE_TABLE = str.maketrans(HOMOGLYPHS)

_WORD = re.compile(r"\w+")
_PLAIN_LATIN = re.compile(r"[A-Za-z]")


def _fix_word(match: re.Match[str]) -> str:
    word = match.group()
    if _PLAIN_LATIN.search(word) and any(char in HOMOGLYPHS for char in word):
        return word.translate(_LOOKALIKE_TABLE)  # a Latin word with a foreign letter smuggled in
    return word


def sanitize_untrusted(text: str | None) -> str:
    """``text`` without invisible characters, and with disguised lookalike letters turned back into Latin ones."""
    if not text:
        return text or ""
    return _WORD.sub(_fix_word, text.translate(_DELETE))
