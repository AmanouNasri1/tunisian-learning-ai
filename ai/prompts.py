"""
Prompt templates for the AI tutor and correction engine.

Layering:
  - SYSTEM_PROMPT: fixed identity. Never varies.
  - build_developer_prompt(): per-request rules + retrieved context + student state.
  - USER_TEMPLATES: the 10 tutor modes.

Hard rule everywhere: ground answers in provided sources, cite them, and if
sources are weak/missing, SAY SO. Never fabricate a source, year, or correction.
"""

from __future__ import annotations

from dataclasses import dataclass


SYSTEM_PROMPT = """\
Tu es un professeur particulier sérieux et rigoureux pour le Baccalauréat tunisien.
Tu enseignes selon les conventions et le niveau du Bac tunisien ACTUEL.

Règles non négociables :
1. Tu ne fabriques JAMAIS un exercice, une correction, une année, une session ou un fait.
2. Tu fondes chaque affirmation factuelle sur les SOURCES fournies et tu les cites.
3. Si les sources sont absentes ou faibles, tu le dis clairement et tu ne prétends pas
   t'appuyer sur un document officiel.
4. Tu es précis avec la notation mathématique et scientifique. Tu montres les étapes,
   pas seulement le résultat.
5. Tu distingues « la correction officielle indique X » de « une méthode générale est Y ».
6. Si une source est marquée hors programme / ancienne, tu préviens l'élève.
7. Tu restes dans le cadre de la préparation au Bac. Pas de conseils hors sujet.
8. Tu es encourageant mais jamais flatteur : comme un bon prof tunisien qui veut que
   l'élève réussisse réellement.

Langue : tu réponds dans la langue demandée (Français / العربية / Darija tunisienne).
Les termes mathématiques et scientifiques gardent leur forme standard quelle que soit la langue.
"""


# Per-mode user instructions. {question} and {context_summary} are filled by the caller.
USER_TEMPLATES = {
    "explain_fr": "Explique en français, de façon pédagogique, la notion/exercice suivant :\n\n{question}",
    "explain_ar": "اشرح بالعربية الفصحى المفهوم/التمرين التالي بطريقة تربوية:\n\n{question}",
    "explain_darja": "Fhem el talmidh bel darja tounsia (mais garde les termes scientifiques standards) :\n\n{question}",
    "hint_only": "Donne UN SEUL indice pour la prochaine étape. Ne révèle PAS la solution complète. Maximum 2 phrases.\n\n{question}",
    "full_correction": "Donne la correction complète et rédigée, étape par étape, en t'appuyant sur la correction officielle si elle est fournie.\n\n{question}",
    "bac_style_answer": "Rédige une réponse au format attendu le jour du Bac tunisien (rigueur, justifications, notation propre).\n\n{question}",
    "identify_mistakes": "Voici la réponse de l'élève. Identifie précisément les erreurs et les étapes manquantes, sans tout réécrire.\n\n{question}",
    "generate_similar": "Génère un exercice SIMILAIRE (même chapitre/concept, même niveau Bac) avec sa correction. Ne copie pas un exercice existant à l'identique.\n\n{question}",
    "revision_summary": "Crée une fiche de révision concise (définitions, formules clés, méthode, pièges) pour :\n\n{question}",
    "explain_points_lost": "Explique à l'élève POURQUOI il a perdu des points, en te basant sur le barème fourni, et comment les récupérer.\n\n{question}",
}


@dataclass
class RetrievedContext:
    """The grounded bundle returned by the RAG layer."""
    lesson_summary: str = ""
    similar_exercises: list[dict] = None      # [{id, text, citation}]
    correction: str = ""
    rubric: list[dict] = None                 # [{description, points}]
    common_mistakes: list[str] = None
    prerequisite_concepts: list[str] = None
    student_past_mistakes: list[str] = None
    retrieval_confidence: float = 0.0
    citations: list[str] = None               # human-readable source labels

    def is_weak(self, threshold: float) -> bool:
        return self.retrieval_confidence < threshold


def build_developer_prompt(ctx: RetrievedContext, language: str, threshold: float) -> str:
    """Inject retrieval state + student state + the strict output contract."""
    weak = ctx.is_weak(threshold)
    weak_note = "FAIBLE — préviens l'élève et n'invente pas de source" if weak else "suffisante"
    lines = [
        f"LANGUE DE RÉPONSE: {language}",
        f"CONFIANCE DE RECHERCHE: {ctx.retrieval_confidence:.2f} ({weak_note})",
        "",
        "=== SOURCES RÉCUPÉRÉES ===",
    ]
    if ctx.lesson_summary:
        lines.append(f"[Cours] {ctx.lesson_summary}")
    for ex in (ctx.similar_exercises or []):
        lines.append(f"[Exercice similaire] ({ex.get('citation','?')}) {ex.get('text','')}")
    if ctx.correction:
        lines.append(f"[Correction] {ctx.correction}")
    for r in (ctx.rubric or []):
        lines.append(f"[Barème] {r.get('description','')} — {r.get('points','?')} pts")
    for m in (ctx.common_mistakes or []):
        lines.append(f"[Erreur fréquente] {m}")
    for p in (ctx.prerequisite_concepts or []):
        lines.append(f"[Prérequis] {p}")
    for pm in (ctx.student_past_mistakes or []):
        lines.append(f"[Erreur passée de l'élève] {pm}")
    if not (ctx.lesson_summary or ctx.similar_exercises or ctx.correction):
        lines.append("(Aucune source pertinente trouvée.)")

    lines += [
        "",
        "=== RÈGLES DE SORTIE ===",
        "- Cite tes sources de façon inline, ex: (Bac 2019, session principale, Sciences exp., ex.2).",
        "- Ne cite QUE des sources présentes ci-dessus. Toute autre citation est interdite.",
        "- Si la confiance est faible: commence par « ⚠️ Réponse non basée sur un document Bac officiel. »",
        "- Termine ta réponse par un bloc JSON: "
        '{"used_sources": bool, "citations": [...], "confidence": 0.0-1.0}.',
    ]
    return "\n".join(lines)


def build_user_prompt(mode: str, question: str) -> str:
    template = USER_TEMPLATES.get(mode)
    if template is None:
        raise ValueError(f"Unknown tutor mode: {mode}")
    return template.format(question=question)


# --- Correction engine prompt (Part 10) -------------------------------------- #

CORRECTION_PROMPT = """\
Tu corriges la réponse d'un élève au Bac tunisien en t'appuyant STRICTEMENT sur le
barème fourni. Pour chaque item du barème, décide: acquis / partiellement / non acquis,
avec une justification courte. Identifie les erreurs et les étapes manquantes. Estime
le score total (toujours présenté comme une ESTIMATION). Produis enfin une solution
rédigée au format Bac.

QUESTION:
{question}

BARÈME (chaque item: description, points, mots-clés attendus):
{rubric}

CORRECTION DE RÉFÉRENCE:
{reference_correction}

RÉPONSE DE L'ÉLÈVE:
{student_answer}

Réponds UNIQUEMENT en JSON valide selon ce schéma:
{{
  "estimated_score": number, "max_score": number, "score_is_estimate": true,
  "correct_parts": [string],
  "mistakes": [{{"description": string, "concept": string, "rubric_item": string, "lost_points": number}}],
  "missing_steps": [string],
  "bac_style_solution": string,
  "recommended_revision": [string],
  "citations": [string]
}}
"""
