"""
Ingestion pipeline skeleton: raw Bac PDF -> validated structured exercise JSON.

Design principle: extraction is cheap and often wrong; human review is expensive
and right. Everything below the confidence threshold goes to the admin review
queue. Nothing enters the validated pool unseen.

Each stage is a small, replaceable function. The prototype implements digital-PDF
extraction; scanned/OCR + Mathpix are stubbed behind the same interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Stage outputs
# --------------------------------------------------------------------------- #

@dataclass
class FieldValue:
    """Every extracted field carries a value, where it came from, and a confidence."""
    value: object
    source: str            # "filename" | "header" | "regex" | "llm" | "manual"
    confidence: float


@dataclass
class DocumentClassification:
    is_scanned: bool
    year: FieldValue
    session: FieldValue
    section: FieldValue
    subject: FieldValue
    doc_type: FieldValue   # "exam" | "correction"


@dataclass
class IngestResult:
    classification: DocumentClassification
    cleaned_text: str
    exercises: list[dict] = field(default_factory=list)   # ProcessedExercise dicts
    overall_confidence: float = 0.0
    needs_review: bool = True


# --------------------------------------------------------------------------- #
# Stage 0 — classify (filename gets you ~70% for free)
# --------------------------------------------------------------------------- #

_YEAR_RE = re.compile(r"(19[89]\d|20[0-4]\d)")
_SESSION_RE = re.compile(r"\b(principale|control|contr[ôo]le|rattrapage)\b", re.I)
_SUBJECT_HINTS = {"math": "MATH", "physi": "PHYSIQUE", "svt": "SVT", "science": "SVT"}
_SECTION_HINTS = {"sciences": "SC_EXP", "scexp": "SC_EXP", "exp": "SC_EXP", "math": "MATH"}


def classify_from_filename(filename: str) -> DocumentClassification:
    name = filename.lower()

    year_m = _YEAR_RE.search(name)
    year = FieldValue(int(year_m.group()), "filename", 0.95) if year_m else FieldValue(None, "filename", 0.0)

    sess_m = _SESSION_RE.search(name)
    session_val = "controle" if (sess_m and sess_m.group().lower().startswith(("control", "contr", "rattr"))) \
        else ("principale" if sess_m else None)
    session = FieldValue(session_val, "filename", 0.9 if sess_m else 0.0)

    subject_val = next((code for hint, code in _SUBJECT_HINTS.items() if hint in name), None)
    subject = FieldValue(subject_val, "filename", 0.8 if subject_val else 0.0)

    section_val = next((code for hint, code in _SECTION_HINTS.items() if hint in name), None)
    section = FieldValue(section_val, "filename", 0.7 if section_val else 0.0)

    doc_type_val = "correction" if any(t in name for t in ("correction", "corrige", "corrigé")) else "exam"
    doc_type = FieldValue(doc_type_val, "filename", 0.8)

    return DocumentClassification(
        is_scanned=False,  # refined in extract stage
        year=year, session=session, section=section, subject=subject, doc_type=doc_type,
    )


# --------------------------------------------------------------------------- #
# Stage 1 — extract text (digital implemented; scanned stubbed)
# --------------------------------------------------------------------------- #

def extract_text(pdf_path: str) -> tuple[str, bool, str]:
    """
    Returns (raw_text, is_scanned, ocr_engine).
    Digital: pdfplumber. If the text layer is empty/garbage -> scanned -> OCR route.
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed; see requirements.txt")

    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)

    if len(text.strip()) > 200:
        return text, False, "pdfplumber"
    # Empty text layer -> scanned. OCR is implemented later (Tesseract for text,
    # Mathpix for math-heavy pages). Keep the same return contract.
    return _ocr_scanned(pdf_path)


def _ocr_scanned(pdf_path: str) -> tuple[str, bool, str]:
    """Stub. Week 2+: render pages to images, route math-heavy pages to Mathpix,
    the rest to Tesseract (fra+ara). Do NOT OCR math with Tesseract."""
    raise NotImplementedError("Scanned OCR not implemented in the prototype.")


# --------------------------------------------------------------------------- #
# Stage 2 — clean
# --------------------------------------------------------------------------- #

def clean_text(raw: str) -> str:
    text = re.sub(r"-\n(?=\w)", "", raw)          # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # TODO: normalize Arabic presentation forms, normalize math to LaTeX.
    return text.strip()


# --------------------------------------------------------------------------- #
# Stage 3 — segment into exercise/question tree (rule-based; LLM fallback later)
# --------------------------------------------------------------------------- #

_EXERCISE_RE = re.compile(r"(?im)^\s*(exercice|exercise)\s+(\d+)")
_POINTS_RE = re.compile(r"\(?\s*(\d+(?:[.,]\d+)?)\s*(?:points?|pts?)\s*\)?", re.I)


def segment_exercises(cleaned: str) -> list[dict]:
    """Split flat text into exercises. Returns minimal dicts to be enriched by tagging."""
    matches = list(_EXERCISE_RE.finditer(cleaned))
    exercises: list[dict] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        body = cleaned[start:end].strip()
        pts = _POINTS_RE.search(body)
        exercises.append({
            "number": int(m.group(2)),
            "exercise_text": body,
            "total_points": float(pts.group(1).replace(",", ".")) if pts else None,
            "questions": [],   # filled by a finer segmenter / LLM
        })
    return exercises


# --------------------------------------------------------------------------- #
# Stage 5 — confidence + review gate
# --------------------------------------------------------------------------- #

def overall_confidence(c: DocumentClassification, ocr_conf: float) -> float:
    """Weighted MIN of critical fields: an unknown section makes the doc untrustworthy."""
    critical = [c.year.confidence, c.section.confidence, c.subject.confidence, ocr_conf]
    return round(min(critical), 3)


REVIEW_THRESHOLD = 0.8


def ingest(pdf_path: str, filename: str) -> IngestResult:
    classification = classify_from_filename(filename)
    raw, is_scanned, engine = extract_text(pdf_path)
    classification.is_scanned = is_scanned
    cleaned = clean_text(raw)
    exercises = segment_exercises(cleaned)

    ocr_conf = 0.95 if not is_scanned else 0.7  # refined by the OCR engine later
    conf = overall_confidence(classification, ocr_conf)

    return IngestResult(
        classification=classification,
        cleaned_text=cleaned,
        exercises=exercises,
        overall_confidence=conf,
        needs_review=conf < REVIEW_THRESHOLD,
    )
