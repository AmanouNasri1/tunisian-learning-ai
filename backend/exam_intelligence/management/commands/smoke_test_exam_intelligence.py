"""
Human-readable smoke test for the exam-intelligence backend.

Checks (each prints PASS / FAIL / WARN):
  1. Database connection
  2. pgvector extension present (postgres only)
  3. Core tables exist
  4. Reference data loaded
  5. Example exercises loaded
  6. Readiness score service runs
  7. Admin modules import

Exit code is non-zero if any hard check FAILs, so it is usable in CI later.

Usage:
    python manage.py smoke_test_exam_intelligence
"""

from django.core.management.base import BaseCommand
from django.db import connection

from backend.exam_intelligence.models import (
    BacSection, Correction, CurriculumEra, Exam, ExamExercise, ExamQuestion,
    RubricItem, Subject,
)
from backend.exam_intelligence.services.readiness import (
    ReadinessComponents, overall_readiness, subject_readiness,
)


class Command(BaseCommand):
    help = "Run a human-readable smoke test of the exam-intelligence backend."

    def handle(self, *args, **options):
        self.failed = False
        self.stdout.write("BacPilot AI — exam-intelligence smoke test")
        self.stdout.write("=" * 50)

        self._check_db_connection()
        self._check_pgvector()
        self._check_core_tables()
        self._check_reference_data()
        self._check_examples()
        self._check_readiness()
        self._check_admin_import()

        self.stdout.write("=" * 50)
        if self.failed:
            self.stdout.write(self.style.ERROR("SMOKE TEST: FAIL"))
            raise SystemExit(1)
        self.stdout.write(self.style.SUCCESS("SMOKE TEST: PASS"))

    # --- helpers --------------------------------------------------------- #

    def _ok(self, msg):
        self.stdout.write(self.style.SUCCESS(f"[PASS] {msg}"))

    def _warn(self, msg):
        self.stdout.write(self.style.WARNING(f"[WARN] {msg}"))

    def _fail(self, msg):
        self.failed = True
        self.stdout.write(self.style.ERROR(f"[FAIL] {msg}"))

    # --- checks ---------------------------------------------------------- #

    def _check_db_connection(self):
        try:
            with connection.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            self._ok(f"Database connection ({connection.vendor})")
        except Exception as exc:
            self._fail(f"Database connection: {exc}")

    def _check_pgvector(self):
        if connection.vendor != "postgresql":
            self._warn(f"pgvector check skipped (DB vendor is '{connection.vendor}', not postgresql)")
            return
        try:
            with connection.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                if cur.fetchone():
                    self._ok("pgvector extension present")
                else:
                    self._fail("pgvector extension NOT installed (run: CREATE EXTENSION vector;)")
        except Exception as exc:
            self._fail(f"pgvector check error: {exc}")

    def _check_core_tables(self):
        existing = set(connection.introspection.table_names())
        required = [
            ExamExercise._meta.db_table, ExamQuestion._meta.db_table,
            Correction._meta.db_table, BacSection._meta.db_table, Subject._meta.db_table,
        ]
        missing = [t for t in required if t not in existing]
        if missing:
            self._fail(f"Missing tables: {missing} (did you run migrate?)")
        else:
            self._ok(f"Core tables exist ({len(required)} checked)")

    def _check_reference_data(self):
        try:
            sections = BacSection.objects.count()
            subjects = Subject.objects.count()
            eras = CurriculumEra.objects.count()
        except Exception as exc:
            self._fail(f"Reference data query failed: {exc}")
            return
        if sections and subjects and eras:
            self._ok(f"Reference data present (sections={sections}, subjects={subjects}, eras={eras})")
        else:
            self._warn(f"Reference data incomplete (sections={sections}, subjects={subjects}, "
                       f"eras={eras}) — run load_reference_data")

    def _check_examples(self):
        try:
            exercises = ExamExercise.objects.count()
            rubric = RubricItem.objects.count()
        except Exception as exc:
            self._fail(f"Example query failed: {exc}")
            return
        if exercises:
            self._ok(f"Example exercises loaded (exercises={exercises}, rubric_items={rubric})")
        else:
            self._warn("No exercises loaded — run load_example_exercises (optional)")

    def _check_readiness(self):
        try:
            maths = ReadinessComponents(diagnostic=0.6, accuracy=0.7, mastery=0.65,
                                        mock=0.5, recency=1.0, recurring_unresolved_mistakes=2)
            score = subject_readiness(maths)
            overall = overall_readiness([(score, 3), (58.0, 4), (66.0, 4)])
            assert 0 <= score <= 100 and 0 <= overall <= 100
            self._ok(f"Readiness service runs (subject={score}, overall={overall})")
        except Exception as exc:
            self._fail(f"Readiness service error: {exc}")

    def _check_admin_import(self):
        try:
            import backend.exam_intelligence.admin  # noqa: F401
            from django.contrib import admin
            registered = len(admin.site._registry)
            self._ok(f"Admin modules import ({registered} models registered)")
        except Exception as exc:
            self._fail(f"Admin import error: {exc}")
