from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import docx  # python-docx
    _DOCX_AVAILABLE = True
except Exception:
    _DOCX_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
    _PDF_AVAILABLE = True
except Exception:
    _PDF_AVAILABLE = False

from .domain import CaseFragment, ClauseBlock
from .domain import TAGS  # not used here, but handy when importing io elsewhere


# --- Small utilities ---------------------------------------------------------
def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def clip_text(s: str, max_chars: int = 350) -> str:
    s = normalize_spaces(s)
    return (s[:max_chars] + "…") if len(s) > max_chars else s


# --- JSON & text extraction --------------------------------------------------
def read_json_file(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON file '{path}': {e}")


def extract_text_from_file(file_path: Path) -> str:
    """Extract text from .txt/.pdf/.docx. Unknown types are attempted as UTF‑8 text.
    If a parser is missing, a clear error is raised.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        if not _PDF_AVAILABLE:
            raise RuntimeError(
                "PDF provided but 'pdfminer.six' is not installed.\n"
                "Install: pip install pdfminer.six\n"
                "Or convert the contract to .txt/.docx and re-run."
            )
        return pdf_extract_text(str(file_path)) or ""

    if suffix == ".docx":
        if not _DOCX_AVAILABLE:
            raise RuntimeError(
                "DOCX provided but 'python-docx' is not installed.\n"
                "Install: pip install python-docx\n"
                "Or convert the contract to .txt and re-run."
            )
        doc = docx.Document(str(file_path))
        return "\n".join([p.text for p in doc.paragraphs])

    # Fallback: try UTF‑8 text
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Unsupported file type '{suffix}'. Provide .txt/.pdf/.docx. Details: {e}")


# --- Contract structure detection -------------------------------------------
def detect_section_headers(text: str) -> List[Tuple[int, str]]:
    """Detect common section headers (General/Particular Conditions, Appendix to Bid)."""
    headers = [
        ("General Conditions", re.compile(r"(?im)^\s*(general\s+conditions)\b")),
        ("Particular Conditions", re.compile(r"(?im)^\s*(particular\s+conditions)\b")),
        ("Appendix to Bid", re.compile(r"(?im)^\s*(appendix\s+to\s+bid)\b")),
        ("Conditions of Particular Application", re.compile(r"(?im)^\s*(conditions\s+of\s+particular\s+application)\b")),
    ]
    found: List[Tuple[int, str]] = []
    for name, pat in headers:
        for m in pat.finditer(text):
            found.append((m.start(), name))
    found.sort(key=lambda x: x[0])
    return found


def split_into_clauses(text: str) -> List[ClauseBlock]:
    """Split contract text into FIDIC-like clause blocks."""
    txt = text.replace("\r\n", "\n").replace("\r", "\n")
    section_marks = detect_section_headers(txt)

    def section_for_index(idx: int) -> str:
        current = "General"
        for pos, name in section_marks:
            if pos <= idx:
                current = name
            else:
                break
        return current

    clause_pat = re.compile(r"(?m)^\s*(?:Sub-?\s*Clause|Clause)?\s*(?P<num>\d{1,2}(?:\.\d+){0,2})\b")
    indices = [(m.start(), m.group("num")) for m in clause_pat.finditer(txt)]
    blocks: List[ClauseBlock] = []

    if not indices:
        blocks.append(ClauseBlock(clause="General", section="General", text=txt.strip()))
        return blocks

    ends = indices[1:] + [(len(txt), "END")]
    for (start_i, clause_num), (end_i, _) in zip(indices, ends):
        chunk = txt[start_i:end_i].strip()
        blocks.append(ClauseBlock(clause=clause_num, section=section_for_index(start_i), text=chunk))
    return blocks


# --- Case ingestion ----------------------------------------------------------
def build_case_fragments(cases_dir: Path) -> tuple[list[CaseFragment], dict[str, dict]]:
    fragments: List[CaseFragment] = []
    cache: Dict[str, Dict[str, Any]] = {}
    any_json = False

    for p in sorted(cases_dir.glob("*.json")):
        any_json = True
        data = read_json_file(p)
        case_id = str(data.get("case_id") or p.stem)
        cache[case_id] = data

        for it in data.get("core_issues", []) or []:
            text = f"{it.get('tag','')} | {it.get('clause','')} | {it.get('desc','')}"
            fragments.append(CaseFragment(text=normalize_spaces(text), case_id=case_id, tag=it.get("tag"), clause=it.get("clause"), section="core_issues"))

        for s in data.get("risk_lessons", []) or []:
            fragments.append(CaseFragment(normalize_spaces(s), case_id, None, None, "risk_lessons"))

        for s in data.get("factual_triggers", []) or []:
            fragments.append(CaseFragment(normalize_spaces(s), case_id, None, None, "factual_triggers"))

        for s in data.get("errors_identified", []) or []:
            fragments.append(CaseFragment(normalize_spaces(s), case_id, None, None, "errors_identified"))

        for s in data.get("snippets", []) or []:
            t = f"{s.get('clause','')} | {s.get('text','')}"
            fragments.append(CaseFragment(normalize_spaces(t), case_id, None, s.get("clause"), "snippets"))

    if not any_json:
        raise RuntimeError(f"No JSON files found in cases_dir: {cases_dir}")

    if not fragments:
        raise RuntimeError("No usable fragments extracted from the case JSON files.")

    return fragments, cache
