from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .domain import (
    TAGS, DEFAULT_CLAUSE_MAP, TAG_QUERY_TEMPLATES, SUGGESTIONS
)
from .domain import CaseFragment, ClauseBlock, SearchHit
from .io import extract_text_from_file, split_into_clauses, build_case_fragments, clip_text
from .embeddings import select_backend, cosine_sim_matrix
from .scoring import (
    search_topk, clause_match_bonus, keyword_boost, section_bonus,
    compute_tag_score, find_best_clause_for_tag
)
from .reporting import render_markdown_report
from .domain import TAG_KEYWORDS

# --- Optional deps
try:
    import faiss  # faiss-cpu
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    _CE_AVAILABLE = True
except Exception:
    _CE_AVAILABLE = False


# --------- NEW: filename sanitizer (for cache file names)
def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s or "base")


# --------- NEW: load or build index embeddings cache
def load_or_build_embeddings(
    cases_dir: Path,
    backend,
    index_texts: List[str]
) -> tuple[np.ndarray, bool, Path]:
    """
    Case JSON으로부터 만든 인덱스 텍스트(index_texts)를 임베딩하고 .npz로 캐싱합니다.
    - 반환: (index_embeddings, cache_used, cache_file_path)
    - 모델이 바뀌거나, 사례 개수가 변하면 자동으로 재생성합니다.
    """
    cache_name = f".case_embeddings__{_sanitize_filename(getattr(backend, 'name', 'base'))}.npz"
    meta_name  = f".case_embeddings__{_sanitize_filename(getattr(backend, 'name', 'base'))}.meta.json"
    cache_file = cases_dir / cache_name
    meta_file  = cases_dir / meta_name

    if cache_file.exists() and meta_file.exists():
        try:
            with meta_file.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if int(meta.get("num_items", -1)) == len(index_texts):
                arr = np.load(cache_file)["embeddings"]
                return arr, True, cache_file
        except Exception:
            logging.warning("Embedding cache exists but failed to load or mismatched; will rebuild.")

    # 캐시 미존재 또는 불일치 → 새로 생성
    embeddings = backend.encode(index_texts)
    np.savez(cache_file, embeddings=embeddings)
    meta = {
        "model_name": getattr(backend, "name", "base"),
        "num_items": len(index_texts),
    }
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return embeddings, False, cache_file


def analyze_contract(
    cases_dir: Path,
    contract_file: Path,
    out_json: Path,
    out_md: Optional[Path] = None,
    model_name: Optional[str] = None,
    topk: int = 5,
    max_snippet_chars: int = 500,
    use_faiss: Optional[bool] = True,
    rerank: bool = True,
    retrieval_only: bool = False,
    retrieval_k: int = 5,
    retrieval_topn_global: int = 0,
    min_sim: float = 0.0,
    dedup_case: bool = False,
    section_filter: str = "",
) -> Dict[str, Any]:

    # 1) 사례 조각 로드 (변경 없음)
    fragments, case_cache = build_case_fragments(cases_dir)  # 케이스 JSON → CaseFragment 리스트
    index_texts = [f.text for f in fragments]

    # 2) 임베딩 백엔드 선택 (문장 임베딩 or TF-IDF)
    backend = select_backend(index_texts, model_name=model_name)
    logging.info("Using embedding backend: %s", getattr(backend, "name", "unknown"))

    # 3) NEW: 사례 인덱스 임베딩 캐시 로드/생성
    index_embeddings, cache_used, cache_path = load_or_build_embeddings(
        cases_dir=cases_dir,
        backend=backend,
        index_texts=index_texts,
    )

    # --- Build FAISS index if requested & available
    faiss_index = None
    faiss_used = False
    if use_faiss and _FAISS_AVAILABLE:
        I = np.asarray(index_embeddings, dtype="float32")
        # Ensure L2-normalized for IP ≈ cosine
        norms = np.linalg.norm(I, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        I_norm = I / norms
        d = I_norm.shape[1]
        faiss_index = faiss.IndexFlatIP(d)
        faiss_index.add(I_norm)
        faiss_used = True

    # --- Prepare CrossEncoder once (optional)
    ce_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = None
    rerank_used = False
    if rerank and _CE_AVAILABLE:
        try:
            cross_encoder = CrossEncoder(ce_model_name)
            rerank_used = True
        except Exception:
            cross_encoder = None
            rerank_used = False

    # 4) 계약 텍스트 로드 및 조항 분리 (변경 없음)
    raw_text = extract_text_from_file(contract_file).strip()
    if not raw_text:
        raise RuntimeError("Contract text appears empty after extraction. Provide a readable .txt/.pdf/.docx.")

    blocks = split_into_clauses(raw_text) or [ClauseBlock("General", "General", raw_text)]

    # === Retrieval-only 모드 ===
    if retrieval_only:
        # 계약서 블록을 쿼리로 보고, 케이스 인덱스에서 Top-K 매칭
        block_matches = []
        all_global = []
        
        # 섹션(클로즈) 필터 준비
        prefixes = [s.strip() for s in (section_filter or "").split(",") if s.strip()]
        for b in blocks:
            # 1) 섹션 필터: 예) '13.,20.' 만 통과
            if prefixes:
                clause_str = (b.clause or "")
                if not any(clause_str.startswith(p) for p in prefixes):
                    continue
            qtext = clip_text(b.text, max_chars=max_snippet_chars)
            Q = backend.encode([qtext]).astype("float32")
            # 1차 검색
            if faiss_index is not None:
                qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
                D, I = faiss_index.search(qn, max(50, retrieval_k))
                idxs = I[0].tolist()
                sims = D[0].tolist()
            else:
                sim = cosine_sim_matrix(Q, np.asarray(index_embeddings, dtype="float32"))
                pairs = list(enumerate(sim[0].tolist()))
                pairs.sort(key=lambda x: x[1], reverse=True)
                idxs = [i for i,_ in pairs[:max(50,retrieval_k)]]
                sims = [s for _,s in pairs[:max(50,retrieval_k)]]
            # (옵션) 리랭크
            order = list(range(len(idxs)))
            if cross_encoder is not None and len(idxs) > 0:
                cands = [fragments[i].text for i in idxs]
                pairs_ce = [(qtext, t) for t in cands]
                ce_scores = cross_encoder.predict(pairs_ce).tolist()
                order = list(np.argsort(ce_scores)[::-1][:retrieval_k])
            else:
                order = order[:retrieval_k]
            
            # 2) 후보 집합
            final_idx = [idxs[i] for i in order]
            final_sims = [float(sims[i]) for i in order]
            
            # 3) 유사도 하한 필터
            if min_sim > 0.0:
                keep = [(ii, sc) for ii, sc in zip(final_idx, final_sims) if sc >= min_sim]
                final_idx = [ii for ii, _ in keep]
                final_sims = [sc for _, sc in keep]
            
            # 4) case_id 중복 제거(가장 높은 sim만)
            if dedup_case:
                seen = set()
                dedup_idx, dedup_sims = [], []
                for ii, sc in zip(final_idx, final_sims):
                    cid = fragments[ii].case_id
                    if cid in seen: 
                        continue
                    seen.add(cid)
                    dedup_idx.append(ii); dedup_sims.append(sc)
                final_idx, final_sims = dedup_idx, dedup_sims
            
            # 5) K개로 컷
            if retrieval_k > 0:
                final_idx, final_sims = final_idx[:retrieval_k], final_sims[:retrieval_k]
            match_items = []
            for ii, sc in zip(final_idx, final_sims):
                frag = fragments[ii]
                item = {
                    "case_id": frag.case_id,
                    "clause": frag.clause,
                    "section": frag.section,
                    "similarity": round(sc, 3),
                    "snippet_case": clip_text(frag.text, 300),
                }
                match_items.append(item)
                all_global.append({
                    "contract_section": b.section,
                    "contract_clause": b.clause,
                    "contract_snippet": clip_text(b.text, 220),
                    **item
                })
            # 매칭이 하나도 없으면 블록 자체를 숨김
            if match_items:
                block_matches.append({
                    "contract_section": b.section,
                    "contract_clause": b.clause,
                    "contract_snippet": qtext,
                    "matches": match_items
                })
        # 전역 Top-N (선택)
        global_top = []
        if retrieval_topn_global and len(all_global) > 0:
            all_global.sort(key=lambda x: x["similarity"], reverse=True)
            if min_sim > 0.0:
                all_global = [g for g in all_global if g["similarity"] >= min_sim]
            if dedup_case:
                seen = set(); tmp=[]
                for g in all_global:
                    if g["case_id"] in seen: 
                        continue
                    seen.add(g["case_id"]); tmp.append(g)
                all_global = tmp
            global_top = all_global[:retrieval_topn_global]
        report = {
            "contract_id": contract_file.stem,
            "mode": "retrieval_only",
            "retrieval_k": int(retrieval_k),
            "by_block": block_matches,
            "global_top": global_top,
            "meta": {
                "cases_dir": str(cases_dir.resolve()),
                "contract_file": str(contract_file.resolve()),
                "embedding_backend": getattr(backend, "name", "unknown"),
                "cache_used": bool(cache_used),
                "cache_file": str(cache_path),
                "faiss_used": bool(faiss_used),
                "rerank_used": bool(rerank_used),
            }
        }
        # 저장·리포트
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        if out_md:
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text(render_markdown_report(report), encoding="utf-8")
        return report

    # === 기존 태그 기반 모드 ===
    findings: List[Dict[str, Any]] = []
    tag_scores: List[Tuple[str, float]] = []

    # 5) 태그별 분석 루프
    for tag in TAGS:
        block = find_best_clause_for_tag(blocks, tag)
        contract_snippet = clip_text(block.text, max_chars=max_snippet_chars)
        template = TAG_QUERY_TEMPLATES.get(tag, tag)
        query = f"{template} || CONTRACT_SNIPPET: {contract_snippet}"

        # (A) 질의 임베딩
        Q = backend.encode([query]).astype("float32")

        # (B) 1차 검색: FAISS(가능) or cosine
        if faiss_index is not None:
            qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
            D, I = faiss_index.search(qn, 50)  # top-50
            idx50 = I[0].tolist()
            sims50 = D[0].tolist()
        else:
            sim = cosine_sim_matrix(Q, np.asarray(index_embeddings, dtype="float32"))
            idx_scores = list(enumerate(sim[0].tolist()))
            idx_scores.sort(key=lambda x: x[1], reverse=True)
            idx50 = [i for i, _ in idx_scores[:50]]
            sims50 = [s for _, s in idx_scores[:50]]

        # (C) Rerank: CrossEncoder로 top-50 → top-10
        order = list(range(len(idx50)))
        if cross_encoder is not None and idx50:
            cands = [fragments[i].text for i in idx50]
            pairs = [(query, t) for t in cands]
            ce_scores = cross_encoder.predict(pairs).tolist()
            order = list(np.argsort(ce_scores)[::-1][:10])
        else:
            order = order[:10]

        # 최종 후보 10개 정렬
        final_idx = [idx50[i] for i in order]
        final_sims = [sims50[i] for i in order]
        hits = [SearchHit(score=float(sc), fragment=fragments[ii]) for ii, sc in zip(final_idx, final_sims)]

        # (D) 확신도(Confidence) 계산
        top_sims = final_sims
        mean_top3 = float(np.mean(top_sims[:3])) if top_sims else 0.0
        std_top3 = float(np.std(top_sims[:3])) if top_sims else 0.0
        margin = float(top_sims[0] - (top_sims[3] if len(top_sims) > 3 else 0.0)) if top_sims else 0.0
        if (mean_top3 >= 0.40 and margin >= 0.10):
            confidence = "High"
        elif (mean_top3 >= 0.30 or margin >= 0.05):
            confidence = "Medium"
        else:
            confidence = "Low"

        # (E) 규칙 보정: 태그 키워드 세트 사용
        mean_sim = float(np.mean(top_sims[:5])) if top_sims else 0.0
        bonuses = (
            clause_match_bonus(block.clause, tag)
            + keyword_boost(block.text, TAG_KEYWORDS.get(tag, []))
            + section_bonus(hits)
        )

        score = compute_tag_score(mean_sim, tag, bonuses)
        tag_scores.append((tag, score))

        seen: set[str] = set()
        sims: List[Dict[str, Any]] = []
        default_clause = DEFAULT_CLAUSE_MAP.get(tag)

        def pick_line(pool: List[str]) -> Optional[str]:
            if not pool:
                return None
            if default_clause:
                for line in pool:
                    if default_clause and default_clause in line:
                        return line
            return pool[0]

        for h in hits:
            cid = h.fragment.case_id
            if cid in seen:
                continue
            seen.add(cid)
            cache = case_cache.get(cid, {})
            why = (
                pick_line(cache.get("errors_identified", []))
                or pick_line(cache.get("risk_lessons", []))
                or h.fragment.text
            )

            sims.append({
                "case_id": cid,
                "why_similar": clip_text(why, 240) if why else "",
                "support_text": clip_text(h.fragment.text, 240),
                "score": round(h.score, 3),
            })
            if len(sims) >= 2:
                break

        findings.append({
            "tag": tag,
            "score": round(score, 1),
            "clause": block.clause,
            "section": block.section,
            "evidence_contract": clip_text(block.text, 300),
            "similar_cases": sims,
            "suggestion": SUGGESTIONS.get(tag, ""),
            "confidence": confidence,
            "mean_top3": round(mean_top3, 3),
            "std_top3": round(std_top3, 3),
            "margin": round(margin, 3),
        })

    total_w = sum([1.0 if t not in DEFAULT_CLAUSE_MAP else 1.0 for t, _ in tag_scores]) or 1.0
    overall = sum([1.0 * s for _, s in tag_scores]) / total_w
    overall = round(float(overall), 1)

    report: Dict[str, Any] = {
        "contract_id": contract_file.stem,
        "overall_risk": overall,
        "by_tag": sorted(findings, key=lambda x: x["score"], reverse=True),
        "meta": {
            "cases_dir": str(cases_dir.resolve()),
            "contract_file": str(contract_file.resolve()),
            "embedding_backend": getattr(backend, "name", "unknown"),
            "topk": int(topk),
            # NEW: 캐시 사용 여부/경로 기록
            "cache_used": bool(cache_used),
            "cache_file": str(cache_path),
            "faiss_used": bool(faiss_used),
            "rerank_used": bool(rerank_used),
            "cross_encoder": ce_model_name if rerank_used else "",
        }
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_markdown_report(report), encoding="utf-8")

    return report
