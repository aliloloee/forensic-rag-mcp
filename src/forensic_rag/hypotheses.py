"""The thesis hypotheses and their ground truth, copied from
Thesis codes/B- Built Dataset and Annotations/Annotations.txt in the thesis repo
(https://github.com/aliloloee/forensic-analysis).

Any free-text hypothesis can be investigated; these three are the ones that can be evaluated."""

HYPOTHESES = {
    "H1": {
        "topic": 201,
        "text": (
            "Employees described prepay transactions as loans or financing in order to keep "
            "associated debt off the balance sheet or to improve the company's reported "
            "financial position."
        ),
        "high": [541, 1052],
        "medium": [420, 656, 1028, 1043, 1051, 1068, 1074, 1214, 1040, 1078,
                   369, 592, 1020, 1041, 1085, 1086],
    },
    "H2": {
        "topic": 204,
        "text": (
            "Employees communicated about altering, deleting, or selectively retaining documents "
            "in anticipation of regulatory scrutiny, audits, or legal investigations."
        ),
        "high": [38, 93, 906, 957, 53, 441, 947],
        "medium": [320, 334, 408, 589, 811, 1089, 1204, 914],
    },
    "H3": {
        "topic": 205,
        "text": (
            "Employees discussed adjusting, restricting, or reallocating energy schedules, bids, "
            "or load volumes in ways that could affect supply availability or market prices."
        ),
        "high": [110, 242, 387, 902, 949, 1121, 1283, 1291, 210, 175, 1159, 76, 425, 135],
        "medium": [116, 168, 213, 126, 170, 20],
    },
}


def get_hypothesis(hypothesis_id: str) -> dict:
    key = hypothesis_id.strip().upper()
    if key not in HYPOTHESES:
        raise ValueError(f"Unknown hypothesis '{hypothesis_id}'. Use one of {list(HYPOTHESES)}")
    return HYPOTHESES[key]


def resolve(hypothesis: str) -> tuple[str | None, str]:
    """Accept a known id ("H3") or any free-text hypothesis.

    Returns (known_id or None, hypothesis_text). Free text that equals a known hypothesis
    is also mapped to its id, so evaluation against ground truth still works.
    """
    text = hypothesis.strip()
    if text.upper() in HYPOTHESES:
        return text.upper(), HYPOTHESES[text.upper()]["text"]
    for hid, h in HYPOTHESES.items():
        if _norm(h["text"]) == _norm(text):
            return hid, h["text"]
    return None, text


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("’", "'").split()).rstrip(".")


def annotation(hypothesis_id: str, email_id: int) -> str:
    h = get_hypothesis(hypothesis_id)
    if email_id in h["high"]:
        return "high"
    if email_id in h["medium"]:
        return "medium"
    return "not_relevant"
