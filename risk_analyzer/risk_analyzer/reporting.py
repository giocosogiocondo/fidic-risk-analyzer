from __future__ import annotations

from typing import Dict, List

def render_markdown_report(report: Dict[str, any]) -> str:
    # Retrieval-only 모드
    if report.get("mode") == "retrieval_only":
        lines = []
        lines.append(f"# Contract Similarity Report: `{report.get('contract_id','')}`")
        meta = report.get("meta", {})
        lines.append("")
        lines.append(f"_Embedding_: {meta.get('embedding_backend')} | cache={meta.get('cache_used')} | "
                     f"faiss={meta.get('faiss_used')} | rerank={meta.get('rerank_used')}")
        lines.append("")
        lines.append(f"## By Contract Block (Top-{report.get('retrieval_k')})")
        for blk in report.get("by_block", []):
            title = f"### {blk.get('contract_section','') or 'Block'} — {blk.get('contract_clause','') or ''}".strip()
            lines.append(title)
            cs = blk.get('contract_snippet','')
            lines.append(f"- **Contract snippet**: {cs[:140]}{'…' if len(cs)>140 else ''}")
            lines.append(f"- **Matches:**")
            for m in blk.get("matches", []):
                sn = m['snippet_case']
                lines.append(f"  - `{m['case_id']}` | clause {m.get('clause','') or '-'} | {m.get('section','') or '-'} "
                             f"(sim {m['similarity']}): {sn[:140]}{'…' if len(sn)>140 else ''}")
            lines.append("")
        if report.get("global_top"):
            lines.append("## Global Top Matches")
            for g in report["global_top"]:
                lines.append(f"- (sim {g['similarity']}) `{g['case_id']}` | clause {g.get('clause','')} | {g.get('section','')}")
                lines.append(f"  - **Contract**: [{g.get('contract_section','')}/{g.get('contract_clause','')}] {g.get('contract_snippet','')}")
                lines.append(f"  - **Case**: {g.get('snippet_case','')}")
        return "\n".join(lines)

    # 기존: 태그 기반 리포트 렌더링
    lines: List[str] = []
    lines.append(f"# Contract Risk Report: `{report.get('contract_id','')}`")
    lines.append("")
    lines.append(f"**Overall Risk**: **{report.get('overall_risk', 0)} / 100**")
    lines.append("")
    lines.append("## Findings by Tag")
    for f in report.get("by_tag", []):
        lines.append(f"### {f['tag']} — {f['score']}")
        lines.append(f"- **Clause**: {f['clause']}")
        lines.append(f"- **Section**: {f.get('section','')}")
        lines.append(f"- **Evidence (contract)**: {f['evidence_contract']}")
        if f.get("confidence"):
            lines.append(f"- **Confidence**: {f['confidence']} "
                         f"(mean_top3={f.get('mean_top3')}, std_top3={f.get('std_top3')}, margin={f.get('margin')})")
        if f.get("suggestion"):
            lines.append(f"- **Suggestion**: {f['suggestion']}")
        if f.get("similar_cases"):
            lines.append(f"- **Similar cases:**")
            for sc in f["similar_cases"]:
                lines.append(f"  - `{sc['case_id']}` (sim {sc['score']}): {sc['why_similar']}")
        lines.append("")
    lines.append("---")
    meta = report.get("meta", {})
    lines.append(f"_Embedding_: {meta.get('embedding_backend')} | _topk_: {meta.get('topk')} | "
                 f"cache={meta.get('cache_used')} | faiss={meta.get('faiss_used')} | "
                 f"rerank={meta.get('rerank_used')} {('('+meta.get('cross_encoder','')+')') if meta.get('rerank_used') else ''}")
    return "\n".join(lines)
