"""
Load processed-exercise JSON (seed_data/examples/*.json) into the structured DB.

Maps the ProcessedExercise JSON format -> SourceDocument / Exam / Chapter /
Concept / ExamExercise / ExamQuestion / Correction / RubricItem / CommonMistake /
ExerciseTag.

Idempotent: every record is upserted by a natural key, so re-running does not
duplicate rows. Requires reference data (sections, subjects, eras) to be loaded
first; chapters and concepts referenced by the JSON are created on demand.

Usage:
    python manage.py load_example_exercises seed_data/examples
    python manage.py load_example_exercises seed_data/examples/bac_math_mathematics.json
"""

import json
import re
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from backend.exam_intelligence.models import (
    BacSection, Chapter, CommonMistake, Concept, Correction, CurriculumEra,
    Exam, ExamExercise, ExamQuestion, ExerciseTag, RubricItem, SourceDocument,
    Subject,
)

_EX_NUM_RE = re.compile(r"ex(\d+)\b", re.IGNORECASE)


class Command(BaseCommand):
    help = "Load processed-exercise JSON examples into the structured database."

    def add_arguments(self, parser):
        parser.add_argument(
            "path",
            nargs="?",
            default="seed_data/examples",
            help="A directory of *.json files or a single JSON file.",
        )

    def handle(self, *args, **options):
        path = Path(options["path"])
        files = self._collect_files(path)
        if not files:
            raise CommandError(f"No JSON files found at: {path}")

        loaded, skipped, warnings = 0, 0, []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                warnings.append(f"{f.name}: invalid JSON ({exc}); skipped")
                skipped += 1
                continue

            ok, file_warnings = self._load_one(f, data)
            warnings.extend(f"{f.name}: {w}" for w in file_warnings)
            if ok:
                loaded += 1
            else:
                skipped += 1

        self.stdout.write(self.style.SUCCESS(
            f"Done. files_loaded={loaded} files_skipped={skipped} warnings={len(warnings)}"
        ))
        for w in warnings:
            self.stdout.write(self.style.WARNING(f"  ! {w}"))

    # ------------------------------------------------------------------ #

    @staticmethod
    def _collect_files(path: Path) -> list[Path]:
        if path.is_dir():
            return sorted(path.glob("*.json"))
        if path.is_file():
            return [path]
        return []

    @transaction.atomic
    def _load_one(self, file: Path, data: dict) -> tuple[bool, list[str]]:
        warnings: list[str] = []

        # --- required top-level fields ---
        for field in ("id", "year", "session", "section", "subject", "curriculum_era"):
            if field not in data:
                return False, [f"missing required field '{field}'"]

        section = self._get_ref(BacSection, data["section"]["code"], warnings, "section")
        subject = self._get_ref(Subject, data["subject"]["code"], warnings, "subject")
        era = self._get_ref(CurriculumEra, data["curriculum_era"]["code"], warnings, "era")
        if not (section and subject and era):
            return False, warnings + ["reference data missing — run load_reference_data first"]

        # --- source document ---
        src = data.get("source_document", {})
        source_doc, _ = SourceDocument.objects.get_or_create(
            original_filename=src.get("file", file.name),
            defaults={
                "file": src.get("file", file.name),
                "doc_type": "exam",
                "ocr_engine": "manual_seed",
                "confidence_score": data.get("ocr_confidence"),
                "review_status": "approved",
                "detected_year": data["year"],
                "detected_section": section,
                "detected_subject": subject,
            },
        )

        # --- exam ---
        exam, _ = Exam.objects.update_or_create(
            section=section, subject=subject, year=data["year"], session=data["session"],
            defaults={"era": era, "source_document": source_doc},
        )

        # --- chapter (created on demand) ---
        chapter = None
        if data.get("chapter", {}).get("code"):
            ch = data["chapter"]
            chapter, _ = Chapter.objects.get_or_create(
                subject=subject, era=era, code=ch["code"],
                defaults={"name_fr": ch.get("name_fr", ch["code"])},
            )

        # --- exercise ---
        number = self._exercise_number(data["id"])
        relevance = data.get("current_curriculum_relevance", {})
        exercise, _ = ExamExercise.objects.update_or_create(
            exam=exam, number=number,
            defaults={
                "title": data.get("chapter", {}).get("name_fr", ""),
                "intro_text": data.get("exercise_text", ""),
                "difficulty": data.get("difficulty"),
                "estimated_minutes": data.get("estimated_minutes"),
                "relevance_status": relevance.get("status", "unreviewed"),
                "relevance_weight": data["curriculum_era"].get("relevance_weight"),
                "validated_by_teacher": bool(relevance.get("validated_by_teacher", False)),
            },
        )
        if chapter:
            exercise.chapters.add(chapter)

        # concepts at exercise level (created on demand under the chapter)
        ex_concepts = self._resolve_concepts(data.get("concepts", []), chapter)
        if ex_concepts:
            exercise.concepts.add(*ex_concepts)

        # --- questions (+ subquestions) ---
        question_by_number: dict[str, ExamQuestion] = {}
        for order, q in enumerate(data.get("questions", [])):
            question = self._upsert_question(exercise, q, order, parent=None, chapter=chapter)
            question_by_number[str(q["number"])] = question
            for sub_order, sub in enumerate(q.get("subquestions", [])):
                sub_q = self._upsert_question(exercise, sub, sub_order, parent=question, chapter=chapter)
                question_by_number[str(sub["number"])] = sub_q

        # --- corrections (per question) ---
        corr = data.get("correction", {})
        correction_by_question: dict[str, Correction] = {}
        for item in corr.get("items", []):
            qnum = str(item.get("question", ""))
            question = question_by_number.get(qnum)
            if not question:
                warnings.append(f"correction references unknown question '{qnum}'")
                continue
            text = item.get("text", "")
            if item.get("steps"):
                text += "\n\nÉtapes :\n" + "\n".join(f"- {s}" for s in item["steps"])
            correction, _ = Correction.objects.update_or_create(
                question=question, author_type=corr.get("author_type", "official"),
                defaults={
                    "text": text,
                    "is_official": bool(corr.get("is_official", False)),
                    "reliability": corr.get("reliability", "high"),
                    "source_document": source_doc,
                },
            )
            correction_by_question[qnum] = correction

        # --- rubric items (attach to the correction of their question) ---
        rubric_order: dict[str, int] = {}
        for item in data.get("rubric", []):
            qnum = str(item.get("question", ""))
            correction = correction_by_question.get(qnum)
            if not correction:
                question = question_by_number.get(qnum)
                if not question:
                    warnings.append(f"rubric references unknown question '{qnum}'")
                    continue
                correction, _ = Correction.objects.get_or_create(
                    question=question, author_type=corr.get("author_type", "official"),
                    defaults={"text": "", "is_official": bool(corr.get("is_official", False)),
                              "reliability": corr.get("reliability", "high"),
                              "source_document": source_doc},
                )
                correction_by_question[qnum] = correction
            idx = rubric_order.get(qnum, 0)
            rubric_order[qnum] = idx + 1
            RubricItem.objects.update_or_create(
                correction=correction, order=idx,
                defaults={
                    "description": item.get("description", ""),
                    "points": item.get("points", 0),
                    "keywords": item.get("keywords", []),
                },
            )

        # --- common mistakes (exercise-level) ---
        for cm in data.get("common_mistakes", []):
            CommonMistake.objects.update_or_create(
                exercise=exercise, description_fr=cm.get("description_fr", ""),
                defaults={
                    "description_ar": cm.get("description_ar", ""),
                    "correct_approach": cm.get("correct_approach", ""),
                    "frequency": cm.get("frequency", "common"),
                },
            )

        # --- tags (preserve difficulty + usage recommendations) ---
        if data.get("difficulty"):
            ExerciseTag.objects.get_or_create(
                exercise=exercise, tag_type="difficulty", value=data["difficulty"],
                defaults={"source": "reviewed"},
            )
        for rec in data.get("usage_recommendation", []):
            ExerciseTag.objects.get_or_create(
                exercise=exercise, tag_type="other", value=rec,
                defaults={"source": "reviewed"},
            )

        return True, warnings

    # ------------------------------------------------------------------ #

    def _upsert_question(self, exercise, q: dict, order: int, parent, chapter) -> ExamQuestion:
        question, _ = ExamQuestion.objects.update_or_create(
            exercise=exercise, number=str(q["number"]),
            defaults={"text": q.get("text", ""), "points": q.get("points"),
                      "order": order, "parent": parent},
        )
        concepts = self._resolve_concepts(q.get("concepts", []), chapter)
        if concepts:
            question.concepts.add(*concepts)
        return question

    @staticmethod
    def _resolve_concepts(codes: list[str], chapter) -> list[Concept]:
        if not chapter:
            return []
        resolved = []
        for code in codes:
            concept, _ = Concept.objects.get_or_create(
                chapter=chapter, code=code, defaults={"name_fr": code.replace("_", " ")},
            )
            resolved.append(concept)
        return resolved

    @staticmethod
    def _exercise_number(exercise_id: str) -> int:
        m = _EX_NUM_RE.search(exercise_id)
        return int(m.group(1)) if m else 1

    @staticmethod
    def _get_ref(model, code: str, warnings: list[str], label: str):
        obj = model.objects.filter(code=code).first()
        if not obj:
            warnings.append(f"{label} code '{code}' not found in reference data")
        return obj
