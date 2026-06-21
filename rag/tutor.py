"""Source-grounded tutor answer service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

from ai.llm_client import LLMConfigurationError, LLMMessage, get_llm_client
from backend.exam_intelligence.models import AIInteraction, Language
from rag.context_builder import RAGContextBuilder
from rag.text_normalization import normalize_text, normalized_tokens


REFUSAL_TEXT = (
    "Je ne peux pas répondre de manière fiable avec les documents actuellement "
    "retrouvés. Essaie de préciser la matière, le chapitre, ou la question exacte."
)
MOCK_TUTOR_MODEL = "mock-tutor-v1"
SUPPORTED_PROVIDERS = {"mock", "openai", "anthropic"}
QUESTION_STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "cette", "ces", "comment", "dans", "de",
    "des", "du", "donne", "explique", "expliquer", "hors", "la", "le", "les",
    "moi", "mon", "programme", "question", "que", "qui", "quoi", "sur", "un",
    "une", "est", "sont", "pour", "par", "en",
}
GROUP_BY_CONTENT_TYPE = {
    "exercise": "exercise_statements",
    "question": "questions",
    "correction": "corrections",
    "rubric": "rubric_items",
    "mistake": "common_mistakes",
    "combined": "combined_context",
}


@dataclass
class TutorDecision:
    refused: bool
    reason: str | None
    grounding_terms: list[str]
    grounding_overlap: list[str]


def answer_student_question(
    query: str,
    section: str | None = None,
    subject: str | None = None,
    chapter: str | None = None,
    provider: str = "mock",
    top_k: int = 6,
) -> dict:
    """Answer a student query from retrieved Bac context, with audit logging."""
    provider_name = (provider or "mock").strip().lower()
    if provider_name not in SUPPORTED_PROVIDERS:
        raise LLMConfigurationError(
            f"Unknown tutor provider '{provider_name}'. Use mock, openai, or anthropic.")

    if provider_name != "mock":
        # Validate configuration before doing any expensive work. This creates a
        # client object but does not send a request.
        get_llm_client(provider_name)

    clean_query = (query or "").strip()
    context = RAGContextBuilder().build(
        query=clean_query,
        section=section,
        subject=subject,
        chapter=chapter,
        top_k=top_k,
    )

    decision = _decide_answerability(clean_query, context)
    warnings = list(context["warnings"])
    if decision.refused and decision.reason and decision.reason not in warnings:
        warnings.append(decision.reason)

    if decision.refused:
        answer_context = _empty_answer_context(context)
        answer = REFUSAL_TEXT
        model_name = MOCK_TUTOR_MODEL if provider_name == "mock" else ""
    else:
        answer_context = _grounded_context(context, decision.grounding_terms)
        if not answer_context["citations"]:
            decision = TutorDecision(
                True,
                "no grounded citation found after filtering retrieved context",
                decision.grounding_terms,
                decision.grounding_overlap,
            )
            warnings.append(decision.reason)
            answer_context = _empty_answer_context(context)
            answer = REFUSAL_TEXT
            model_name = MOCK_TUTOR_MODEL if provider_name == "mock" else ""
        elif provider_name == "mock":
            answer = _mock_grounded_answer(clean_query, answer_context)
            model_name = MOCK_TUTOR_MODEL
        else:
            client = get_llm_client(provider_name)
            response = client.complete(
                system=_system_prompt(),
                messages=[LLMMessage(
                    role="user",
                    content=_real_llm_prompt(clean_query, answer_context),
                )],
                temperature=0.1,
                max_tokens=900,
            )
            answer = response.text.strip()
            model_name = response.model

    if decision.refused and not model_name:
        model_name = MOCK_TUTOR_MODEL if provider_name == "mock" else ""

    answer_language = _detect_language(clean_query or answer)
    used_chunks = _used_chunks(answer_context)
    citations = answer_context["citations"]

    package = {
        "query": clean_query,
        "answer": answer,
        "answer_language": answer_language,
        "retrieval_mode": context["retrieval_mode"],
        "confidence": context["confidence"],
        "used_chunks": used_chunks,
        "citations": citations,
        "warnings": _dedupe(warnings),
        "refused": decision.refused,
        "refusal_reason": decision.reason,
        "provider": provider_name,
        "model_name": model_name,
        "diagnostics": {
            **context["diagnostics"],
            "selected_chunk_count": context["selected_chunk_count"],
            "used_chunk_count": len(used_chunks),
            "grounding_terms": decision.grounding_terms,
            "grounding_overlap": decision.grounding_overlap,
            "context_warnings": context["warnings"],
        },
    }
    package["interaction_id"] = _log_interaction(package)
    return package


def _decide_answerability(query: str, context: dict) -> TutorDecision:
    terms = _meaningful_terms(query)
    overlap = _grounding_overlap(terms, context)

    if not query:
        return TutorDecision(True, "missing query", terms, overlap)
    if context["selected_chunk_count"] == 0:
        return TutorDecision(True, "no chunks retrieved", terms, overlap)
    if not context["citations"]:
        return TutorDecision(True, "no relevant citation found", terms, overlap)
    if context["diagnostics"].get("keyword_candidates", 0) == 0:
        return TutorDecision(
            True,
            "question outside loaded Bac content or unsupported by retrieved context",
            terms,
            overlap,
        )
    if terms and not overlap:
        return TutorDecision(
            True,
            "question not supported by the retrieved context",
            terms,
            overlap,
        )
    return TutorDecision(False, None, terms, overlap)


def _meaningful_terms(query: str) -> list[str]:
    return [
        token for token in normalized_tokens(query)
        if len(token) > 2 and token not in QUESTION_STOPWORDS
    ]


def _grounding_overlap(terms: list[str], context: dict) -> list[str]:
    if not terms:
        return []
    chunk_text = " ".join(chunk.get("content", "") for chunk in context.get("selected_chunks", []))
    citation_text = " ".join(
        " ".join(str(citation.get(key) or "") for key in [
            "source_object_type", "section", "subject", "chapter", "year",
        ])
        for citation in context.get("citations", [])
    )
    haystack = normalize_text(f"{chunk_text} {citation_text}")
    return [term for term in terms if term in haystack]


def _grounded_context(context: dict, terms: list[str]) -> dict:
    if not terms:
        return context

    citations_by_id = {citation["chunk_id"]: citation for citation in context["citations"]}
    direct_ids: set[int] = set()
    matched_chapters: set[str] = set()

    for chunk in context["selected_chunks"]:
        citation = citations_by_id.get(chunk["chunk_id"], {})
        searchable = normalize_text(" ".join([
            chunk.get("content", ""),
            str(citation.get("source_object_type") or ""),
            str(citation.get("section") or ""),
            str(citation.get("subject") or ""),
            str(citation.get("chapter") or ""),
            str(citation.get("year") or ""),
        ]))
        if any(term in searchable for term in terms):
            direct_ids.add(chunk["chunk_id"])
            if citation.get("chapter"):
                matched_chapters.add(citation["chapter"])

    if matched_chapters:
        allowed_ids = {
            citation["chunk_id"]
            for citation in context["citations"]
            if citation.get("chapter") in matched_chapters
        }
    else:
        allowed_ids = direct_ids

    selected_chunks = [
        chunk for chunk in context["selected_chunks"]
        if chunk["chunk_id"] in allowed_ids
    ]
    citations = [
        citation for citation in context["citations"]
        if citation["chunk_id"] in allowed_ids
    ]
    grouped = _group_selected_chunks(selected_chunks)

    return {
        **context,
        "selected_chunks": selected_chunks,
        "selected_chunk_count": len(selected_chunks),
        "grouped_context": grouped,
        "citations": citations,
    }


def _group_selected_chunks(chunks: list[dict]) -> dict:
    grouped = {
        "exercise_statements": [],
        "questions": [],
        "corrections": [],
        "rubric_items": [],
        "common_mistakes": [],
        "combined_context": [],
        "assembled_context": "",
    }
    for chunk in chunks:
        key = GROUP_BY_CONTENT_TYPE.get(chunk["content_type"])
        if key:
            grouped[key].append(chunk)
    grouped["assembled_context"] = "\n\n".join(
        f"[chunk#{chunk['chunk_id']} {chunk['content_type']}]\n{chunk['content']}"
        for chunk in chunks
    )
    return grouped


def _empty_answer_context(context: dict) -> dict:
    return {
        **context,
        "selected_chunks": [],
        "selected_chunk_count": 0,
        "grouped_context": _group_selected_chunks([]),
        "citations": [],
    }


def _mock_grounded_answer(query: str, context: dict) -> str:
    grouped = context["grouped_context"]
    parts = [
        "Réponse mock fondée uniquement sur les documents retrouvés.",
        f"Question: {query}",
    ]

    lead = _first_content(grouped, ["corrections", "combined_context", "questions", "exercise_statements"])
    if lead:
        parts.append(f"Idée utile: {lead}")

    steps = _contents(grouped, "rubric_items", limit=3)
    if steps:
        parts.append("Étapes à suivre:\n" + "\n".join(f"- {step}" for step in steps))

    questions = _contents(grouped, "questions", limit=2)
    if questions:
        parts.append("Questions liées dans les annales:\n" + "\n".join(f"- {item}" for item in questions))

    mistakes = _contents(grouped, "common_mistakes", limit=2)
    if mistakes:
        parts.append("À éviter:\n" + "\n".join(f"- {mistake}" for mistake in mistakes))

    source_ids = ", ".join(f"chunk#{c['chunk_id']}" for c in context["citations"][:4])
    parts.append(f"Sources utilisées: {source_ids}.")
    return "\n\n".join(parts)


def _first_content(grouped: dict, keys: list[str]) -> str:
    for key in keys:
        values = grouped.get(key, [])
        if values:
            return _shorten(values[0].get("content", ""))
    return ""


def _contents(grouped: dict, key: str, limit: int) -> list[str]:
    return [_shorten(item.get("content", "")) for item in grouped.get(key, [])[:limit]]


def _shorten(text: str, limit: int = 420) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _system_prompt() -> str:
    return (
        "Tu es un tuteur Bac tunisien. Réponds uniquement avec les éléments du "
        "contexte fourni. Si le contexte ne suffit pas, refuse clairement. Cite les chunk ids."
    )


def _real_llm_prompt(query: str, context: dict) -> str:
    return (
        f"Question élève:\n{query}\n\n"
        f"Contexte assemblé:\n{context['grouped_context']['assembled_context']}\n\n"
        f"Citations JSON:\n{json.dumps(context['citations'], ensure_ascii=False)}\n\n"
        "Produis une réponse pédagogique courte, structurée et sourcée."
    )


def _used_chunks(context: dict) -> list[dict]:
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "content_type": chunk["content_type"],
            "score": chunk["score"],
            "citation": chunk["citation"],
        }
        for chunk in context["selected_chunks"]
    ]


def _detect_language(text: str) -> str:
    if not text:
        return "unknown"
    has_arabic = any("\u0600" <= char <= "\u06ff" for char in text)
    has_latin = any(("a" <= char.lower() <= "z") or char in "éèêàâîïôùûç" for char in text)
    if has_arabic and has_latin:
        return "mixed"
    if has_arabic:
        return "ar"
    if has_latin:
        return "fr"
    return "unknown"


def _log_interaction(package: dict) -> int:
    interaction = AIInteraction.objects.create(
        mode="tutor_answer",
        language=_language_for_db(package["answer_language"]),
        query=package["query"],
        retrieved_chunk_ids=[chunk["chunk_id"] for chunk in package["used_chunks"]],
        response=package["answer"],
        citations=package["citations"],
        used_sources=bool(package["citations"]) and not package["refused"],
        provider=package["provider"],
        model_name=package["model_name"],
        retrieval_mode=package["retrieval_mode"],
        refused=package["refused"],
        refusal_reason=package["refusal_reason"] or "",
        warnings=package["warnings"],
        tokens=0,
        confidence=Decimal(str(round(float(package["confidence"] or 0.0), 3))),
    )
    return interaction.id


def _language_for_db(language: str) -> str:
    if language in {Language.FR, Language.AR, Language.DARJA}:
        return language
    return Language.FR


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out
