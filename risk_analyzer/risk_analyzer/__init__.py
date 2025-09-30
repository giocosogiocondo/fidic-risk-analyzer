"""risk_analyzer modular package."""

from .domain import (
    CaseFragment, ClauseBlock, SearchHit,
    TAGS, DEFAULT_CLAUSE_MAP, TAG_WEIGHTS, TAG_KEYWORDS, TAG_QUERY_TEMPLATES, SUGGESTIONS
)

__all__ = [
    "CaseFragment", "ClauseBlock", "SearchHit",
    "TAGS", "DEFAULT_CLAUSE_MAP", "TAG_WEIGHTS", "TAG_KEYWORDS", "TAG_QUERY_TEMPLATES", "SUGGESTIONS",
]
