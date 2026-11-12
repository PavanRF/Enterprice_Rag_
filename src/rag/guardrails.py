"""Input and output guardrails: PII masking, prompt-injection and out-of-scope detection."""
from __future__ import annotations

import re

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # account before phone: the phone pattern can otherwise swallow an account number
    "account": re.compile(r"(?i)\baccount\s*(?:no\.?|number|#)?\s*[:#]?\s*\d{6,}\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
}
# Card candidates are checked with the Luhn algorithm to avoid masking ordinary numbers.
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

INJECTION_PATTERNS = [re.compile(p, re.I) for p in [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above) (instructions|prompts?)",
    r"reveal (your|the) (system )?prompt",
    r"you are now (in )?(developer|dan|unrestricted) mode",
    r"show (me )?(all )?restricted (documents|data)",
]]

OUT_OF_SCOPE_PATTERNS = [re.compile(p, re.I) for p in [
    r"\b(write|generate|create) (me )?(a |an )?(poem|song|story|essay|joke)\b",
    r"\b(write|generate|debug) (me )?(some |a |the )?(python|javascript|java|sql|typescript)\b",
    r"\b(stock|crypto|bitcoin) (tip|picks?)\b",
    r"\b(diagnose|prescribe)\b",
    r"\b(sue|lawsuit against|divorce)\b",
]]


def _luhn_valid(candidate: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", candidate)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII with placeholders. Returns the masked text and the PII types found."""
    found: list[str] = []

    def mask_card(match: re.Match) -> str:
        if _luhn_valid(match.group(0)):
            found.append("card")
            return "[CARD]"
        return match.group(0)

    text = _CARD_CANDIDATE.sub(mask_card, text)
    for name, pattern in PII_PATTERNS.items():
        def sub(match: re.Match, name: str = name) -> str:
            found.append(name)
            return f"[{name.upper()}]"
        text = pattern.sub(sub, text)
    return text, found


def is_prompt_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


def is_out_of_scope(text: str) -> bool:
    return any(p.search(text) for p in OUT_OF_SCOPE_PATTERNS)
