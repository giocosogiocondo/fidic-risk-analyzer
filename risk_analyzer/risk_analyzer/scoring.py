from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from .domain import (
    CaseFragment, ClauseBlock, SearchHit,
    DEFAULT_CLAUSE_MAP, TAG_KEYWORDS, TAG_WEIGHTS,
)
from .embeddings import compute_similarity_matrix


def search_topk(sim_row: np.ndarray, fragments: List[CaseFragment], topk: int = 5) -> List[SearchHit]:
    idx_scores = list(enumerate(sim_row.tolist()))
    idx_scores.sort(key=lambda x: x[1], reverse=True)
    hits: List[SearchHit] = []
    for idx, sc in idx_scores[:max(1, topk)]:
        hits.append(SearchHit(score=float(sc), fragment=fragments[idx]))
    return hits


def keyword_boost(text: str, keywords: List[str]) -> int:
    if not text:
        return 0
    lower = text.lower()
    count = sum(1 for k in keywords if k.lower() in lower)
    if count >= 4:
        return 5
    if count >= 2:
        return 3
    if count >= 1:
        return 1
    return 0


def section_bonus(hits: List[SearchHit]) -> int:
    """
    강화된 섹션 보너스:
      - errors_identified: +6
      - risk_lessons: +3
      - factual_triggers: +2
    """
    bonus = 0
    sections = {h.fragment.section for h in hits}
    if "errors_identified" in sections:
        bonus += 6
    if "risk_lessons" in sections:
        bonus += 3
    if "factual_triggers" in sections:
        bonus += 2
    return min(bonus, 12)


def clause_match_bonus(selected_clause: str, tag: str) -> int:
    """
    강화된 조항 보정:
      - 완전 일치: +7
      - 접두 일치(예: 13.8.*): +5
    """
    desired = DEFAULT_CLAUSE_MAP.get(tag)
    if desired and selected_clause:
        if selected_clause == desired:
            return 7
        if selected_clause.startswith(desired + "."):
            return 5
    return 0


def compute_tag_score(mean_sim: float, tag: str, bonuses: int) -> float:
    base = mean_sim * 100.0
    weighted = TAG_WEIGHTS.get(tag, 1.0) * base
    score = min(100.0, weighted + float(bonuses))
    return score


def find_best_clause_for_tag(blocks: List[ClauseBlock], tag: str) -> ClauseBlock:
    desired = DEFAULT_CLAUSE_MAP.get(tag)
    if desired:
        exact = [b for b in blocks if b.clause == desired or b.clause.startswith(desired + ".")]
        if exact:
            return exact[0]
    kw = TAG_KEYWORDS.get(tag, [])
    best = None
    best_score = -1
    for b in blocks:
        sc = keyword_boost(b.text, kw)
        if sc > best_score or (sc == best_score and tag == "CostEscalation" and "Particular" in b.section):
            best = b
            best_score = sc
    if best is not None and best_score > 0:
        return best
    return blocks[0] if blocks else ClauseBlock("General", "General", "")
