from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from .analyzer import analyze_contract

def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description="Analyze a new contract against case JSON files and produce a risk report.")
    p.add_argument("--cases_dir", type=str, required=True, help="Directory with case JSON files.")
    p.add_argument("--contract_file", type=str, required=True, help="Contract path (.txt/.pdf/.docx).")
    p.add_argument("--out_json", type=str, default="./outputs/report.json", help="Output JSON path.")
    p.add_argument("--out_md", type=str, default="", help="Optional Markdown output path.")
    p.add_argument("--model_name", type=str, default="", help="Sentence-Transformers model name (if installed).")
    p.add_argument("--topk", type=int, default=5, help="Top-k case fragments per tag.")
    p.add_argument("--max_snippet_chars", type=int, default=500, help="Contract snippet length in queries.")
    p.add_argument("--loglevel", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    p.add_argument("--use_faiss", action="store_true", help="Use FAISS if available (cosine IP).")
    p.add_argument("--no_rerank", action="store_true", help="Disable cross-encoder reranking even if available.")
    # Retrieval-only mode (no tags, no scores)
    p.add_argument("--retrieval_only", action="store_true", help="Show top-K most similar case fragments to contract blocks (no tagging/scoring).")
    p.add_argument("--retrieval_k", type=int, default=5, help="Top-K similar fragments per contract block in retrieval-only mode.")
    p.add_argument("--retrieval_topn_global", type=int, default=0, help="Additionally list global top-N matches across all blocks (0=disable).")
    p.add_argument("--min_sim", type=float, default=0.0, help="Filter out matches below this cosine/IP similarity in retrieval-only mode.")
    p.add_argument("--dedup_case", action="store_true", help="Deduplicate matches by case_id (per block and global).")
    p.add_argument("--section_filter", type=str, default="", help="Comma-separated prefixes of contract clause to include (e.g., '13.,20.,14.'). Empty=all.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    cases_dir = Path(args.cases_dir)
    contract_file = Path(args.contract_file)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md) if args.out_md else None
    model_name = args.model_name or None

    if not cases_dir.exists() or not cases_dir.is_dir():
        print(f"ERROR: cases_dir not found or not a directory: {cases_dir}")
        raise SystemExit(2)
    if not contract_file.exists() or not contract_file.is_file():
        print(f"ERROR: contract_file not found: {contract_file}")
        raise SystemExit(2)

    try:
        report = analyze_contract(
            cases_dir=cases_dir,
            contract_file=contract_file,
            out_json=out_json,
            out_md=out_md,
            model_name=model_name,
            topk=int(args.topk),
            max_snippet_chars=int(args.max_snippet_chars),
            use_faiss=bool(args.use_faiss),
            rerank=not bool(args.no_rerank),
            retrieval_only=bool(args.retrieval_only),
            retrieval_k=int(args.retrieval_k),
            retrieval_topn_global=int(args.retrieval_topn_global),
            min_sim=float(args.min_sim),
            dedup_case=bool(args.dedup_case),
            section_filter=args.section_filter,
        )
        print(f"[OK] JSON saved: {out_json}")
        if out_md:
            print(f"[OK] Markdown saved: {out_md}")
        print(f"Overall risk: {report.get('overall_risk')}")
    except Exception as e:
        logging.error("Analysis failed: %s", e)
        raise SystemExit(1)
