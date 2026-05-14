"""IPA-to-ASCII phoneme token helpers for phonemic TTS training."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List

WORD_TOKEN = "<sep_word>"
SHORT_PAUSE_TOKEN = "<pause_short>"
MEDIUM_PAUSE_TOKEN = "<pause_medium>"
SENT_DECL_TOKEN = "<sent_decl>"
SENT_Q_TOKEN = "<sent_q>"
SENT_EXCL_TOKEN = "<sent_excl>"
STRUCTURAL_TOKENS = (
    WORD_TOKEN,
    SHORT_PAUSE_TOKEN,
    MEDIUM_PAUSE_TOKEN,
    SENT_DECL_TOKEN,
    SENT_Q_TOKEN,
    SENT_EXCL_TOKEN,
)
PHONEME_TOKEN_PREFIX = "<ph_"
PHONEME_TOKEN_SUFFIX = ">"
INVENTORY_VERSION = "phoneme-ascii-v2"

PRIMARY_STRESS = "\u02c8"
SECONDARY_STRESS = "\u02cc"
LONG_MARK = "\u02d0"
HALF_LONG_MARK = "\u02d1"
PALATALIZATION_MARK = "\u02b2"
NASAL_MARK = "\u0303"
TIE_BARS = {"\u0361", "\u035c"}


IPA_SYMBOL_NAMES = {
    "a": "a",
    "ɑ": "a_back",
    "ɐ": "a_central",
    "ɒ": "o_open",
    "æ": "ae",
    "b": "b",
    "β": "b_fricative",
    "c": "c",
    "ç": "c_fricative",
    "d": "d",
    "ð": "dh",
    "e": "e",
    "ə": "schwa",
    "ɚ": "schwa_r",
    "ɛ": "eh",
    "ɜ": "er",
    "ɞ": "oe_open",
    "ɘ": "e_central",
    "f": "f",
    "ɡ": "g",
    "g": "g",
    "ɣ": "gh",
    "h": "h",
    "ɦ": "h_voiced",
    "i": "i",
    "ɪ": "ih",
    "ɨ": "i_bar",
    "j": "y",
    "k": "k",
    "l": "l",
    "ɫ": "l_dark",
    "m": "m",
    "n": "n",
    "ɲ": "gn",
    "ŋ": "ng",
    "o": "o",
    "ɔ": "oh",
    "ø": "eu",
    "œ": "oe",
    "p": "p",
    "r": "r",
    "ɾ": "r_tap",
    "ɹ": "r_en",
    "ʀ": "r_uvular",
    "ʁ": "r_fr",
    "s": "s",
    "ʃ": "sh",
    "ɕ": "sh_alveolopalatal",
    "ʂ": "sh_retroflex",
    "t": "t",
    "θ": "th",
    "u": "u",
    "ʊ": "uh",
    "ʉ": "u_bar",
    "v": "v",
    "w": "w",
    "y": "u_front",
    "ʏ": "u_front_short",
    "z": "z",
    "ʒ": "zh",
    "ʑ": "zh_alveolopalatal",
    "ʐ": "zh_retroflex",
    "x": "x",
    "χ": "x_uvular",
    "ɟ": "j_bar",
    "ʎ": "ly",
    "ɭ": "l_retroflex",
    "ɳ": "n_retroflex",
    "ʈ": "t_retroflex",
    "ɖ": "d_retroflex",
    "ɤ": "gamma_ram",
    "ɯ": "m_turn",
    "ɰ": "w_unrounded",
    "ʔ": "glottal_stop",
}

IPA_CLUSTER_NAMES = {
    "tʃ": "tsh",
    "dʒ": "dzh",
    "ts": "ts",
    "dz": "dz",
    "tɕ": "t_sh_alveolopalatal",
    "dʑ": "d_zh_alveolopalatal",
    "pf": "pf",
}

TOKEN_RE = re.compile(r"^<ph_[a-z0-9_]+>$|^<sep_word>$|^<pause_(short|medium)>$|^<sent_(decl|q|excl)>$")


@dataclass(frozen=True)
class PhonemeToken:
    """One tokenized IPA phone with provenance."""

    ipa: str
    token: str


def ipa_to_ascii_token_items(ipa_text: str) -> List[PhonemeToken]:
    """Convert spaced IPA text into ASCII phoneme tokens and word boundaries."""
    items: List[PhonemeToken] = []
    for piece in str(ipa_text or "").split():
        if piece in {"/", WORD_TOKEN}:
            items.append(PhonemeToken(ipa=piece, token=WORD_TOKEN))
            continue
        if piece in {
            SHORT_PAUSE_TOKEN,
            MEDIUM_PAUSE_TOKEN,
            SENT_DECL_TOKEN,
            SENT_Q_TOKEN,
            SENT_EXCL_TOKEN,
        }:
            items.append(PhonemeToken(ipa=piece, token=piece))
            continue
        token = ipa_phone_to_token(piece)
        if token:
            items.append(PhonemeToken(ipa=piece, token=token))
    return items


def ipa_to_ascii_tokens(ipa_text: str) -> List[str]:
    return [item.token for item in ipa_to_ascii_token_items(ipa_text)]


def ipa_to_ascii_text(ipa_text: str) -> str:
    return " ".join(ipa_to_ascii_tokens(ipa_text))


def validate_ascii_phoneme_tokens(tokens: Iterable[str]) -> None:
    invalid = [token for token in tokens if not TOKEN_RE.match(token)]
    if invalid:
        raise ValueError(f"Invalid phoneme ASCII token(s): {invalid[:10]}")


def ipa_phone_to_token(phone: str) -> str:
    """Convert one IPA phone string to a stable ASCII token."""
    raw = str(phone or "").strip()
    if not raw:
        return ""

    normalized = unicodedata.normalize("NFD", raw)
    stress: str | None = None
    while normalized and normalized[0] in {PRIMARY_STRESS, SECONDARY_STRESS}:
        if normalized[0] == PRIMARY_STRESS:
            stress = "primary"
        elif stress != "primary":
            stress = "secondary"
        normalized = normalized[1:]

    suffixes: List[str] = []
    if stress:
        suffixes.append(stress)

    chars: List[str] = []
    nasal = False
    long = False
    half_long = False
    palatalized = False
    for char in normalized:
        if char == NASAL_MARK:
            nasal = True
        elif char == LONG_MARK:
            long = True
        elif char == HALF_LONG_MARK:
            half_long = True
        elif char == PALATALIZATION_MARK:
            palatalized = True
        elif char in TIE_BARS:
            continue
        elif unicodedata.category(char).startswith("M"):
            suffixes.append(_combining_suffix(char))
        else:
            chars.append(char)

    base = "".join(chars)
    name = _base_phone_name(base)
    if nasal:
        suffixes.append("nasal")
    if long:
        suffixes.append("long")
    if half_long:
        suffixes.append("half_long")
    if palatalized:
        suffixes.append("palatalized")

    parts = [name, *suffixes]
    slug = "_".join(part for part in parts if part)
    slug = re.sub(r"[^a-z0-9_]+", "_", slug.lower()).strip("_")
    return f"{PHONEME_TOKEN_PREFIX}{slug}{PHONEME_TOKEN_SUFFIX}"


def _base_phone_name(base: str) -> str:
    if not base:
        return "empty"
    if base in IPA_CLUSTER_NAMES:
        return IPA_CLUSTER_NAMES[base]
    if base in IPA_SYMBOL_NAMES:
        return IPA_SYMBOL_NAMES[base]
    parts = [IPA_SYMBOL_NAMES.get(char, f"u{ord(char):04x}") for char in base]
    return "_".join(parts)


def _combining_suffix(char: str) -> str:
    name = unicodedata.name(char, f"u{ord(char):04x}").lower()
    name = name.replace("combining", "").replace("mark", "")
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_") or f"u{ord(char):04x}"
