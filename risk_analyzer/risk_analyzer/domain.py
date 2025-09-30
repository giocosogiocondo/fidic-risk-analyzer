from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# --- Data structures ---------------------------------------------------------
@dataclass
class CaseFragment:
    text: str
    case_id: str
    tag: Optional[str]
    clause: Optional[str]
    section: str  # core_issues | risk_lessons | factual_triggers | errors_identified | snippets


@dataclass
class ClauseBlock:
    clause: str   # e.g., "13.8" or "General"
    section: str  # "General Conditions", "Particular Conditions", "Appendix to Bid", or "General"
    text: str


@dataclass
class SearchHit:
    score: float
    fragment: CaseFragment


# --- Domain configuration (fixed to user's schema) ---------------------------
TAGS = [
    "ChangeInLegislation",
    "CostEscalation",
    "NoticeTimeBar",
    "ArbSeatRulesMismatch",
    "NaturalJustice",
]

DEFAULT_CLAUSE_MAP: Dict[str, Optional[str]] = {
    "ChangeInLegislation": "13.7",
    "CostEscalation": "13.8",
    "NoticeTimeBar": "20.1",
    "ArbSeatRulesMismatch": "20.6",
    "NaturalJustice": None,
}

TAG_WEIGHTS: Dict[str, float] = {
    "ArbSeatRulesMismatch": 1.3,
    "NoticeTimeBar": 1.2,
    "ChangeInLegislation": 1.0,
    "CostEscalation": 1.0,
    "NaturalJustice": 0.8,
}

TAG_KEYWORDS: Dict[str, List[str]] = {
    "ChangeInLegislation": ["legislation", "change in law", "법령", "입법", "고시", "notification"],
    "CostEscalation": ["price adjustment", "coefficient", "계수", "index", "WPI", "CPI", "formula"],
    "NoticeTimeBar": ["28 days", "42 days", "notice", "particulars", "records", "증빙", "time-bar"],
    "ArbSeatRulesMismatch": ["ICC", "seat", "Singapore", "Arbitration and Conciliation Act", "1996", "lex arbitri", "IAA", "interest", "costs"],
    "NaturalJustice": ["parallel arbitration", "병행", "동일", "procedural", "fairness", "due process"],
}

TAG_QUERY_TEMPLATES: Dict[str, str] = {
    "ChangeInLegislation": "13.7 legislative change vs administrative notification ambiguity",
    "CostEscalation": "13.8 price adjustment formula coefficients and indices clarity or modification",
    "NoticeTimeBar": "20.1 notice within 28 days and detailed particulars within 42 days and records requirement",
    "ArbSeatRulesMismatch": "20.6 seat/institution vs interest/costs governing law mismatch (ICC Singapore vs Indian Arbitration Act)",
    "NaturalJustice": "procedural fairness safeguards for parallel arbitrations and equal treatment of arbitrators",
}

SUGGESTIONS: Dict[str, str] = {
    "ArbSeatRulesMismatch": "Align interest/costs provisions with the selected seat/institution (e.g., ICC Singapore with IAA); remove conflicting references to external arbitration acts.",
    "NoticeTimeBar": "Clarify 28/42-day notice and particulars requirements, add contemporaneous records obligation and exceptions (e.g., force majeure, legislative change).",
    "CostEscalation": "Specify 13.8 formula coefficients, indices (WPI/CPI), source, frequency, and provide a sample calculation.",
    "ChangeInLegislation": "Define whether administrative notifications are included in 'changes in legislation' and how they interplay with 13.8.",
    "NaturalJustice": "If parallel arbitrations are possible, add due-process safeguards (disclosure, opportunity to comment on external materials).",
}
