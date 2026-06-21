"""
Load curriculum reference data (sections, subjects, coefficients, eras).

`seed_data/reference/01_reference.json` is a VALID Django fixture (list of
{model, pk, fields}). This command wraps `loaddata` so the documented command in
the README works AND prints a clear created/updated summary.

Idempotent: `loaddata` upserts by primary key, so running it twice does not
create duplicate rows.

Usage:
    python manage.py load_reference_data seed_data/reference/01_reference.json
"""

import json
from collections import Counter
from pathlib import Path

from django.apps import apps
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Load curriculum reference data from a Django fixture JSON file."

    def add_arguments(self, parser):
        parser.add_argument(
            "path",
            nargs="?",
            default="seed_data/reference/01_reference.json",
            help="Path to the reference fixture JSON.",
        )

    def handle(self, *args, **options):
        path = Path(options["path"])
        if not path.exists():
            raise CommandError(f"Reference file not found: {path}")

        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CommandError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(records, list):
            raise CommandError("Reference fixture must be a JSON list of objects.")

        # Count rows per model before/after to derive created vs updated.
        model_labels = sorted({r["model"] for r in records})
        before = {label: self._count(label) for label in model_labels}

        call_command("loaddata", str(path), verbosity=0)

        after = {label: self._count(label) for label in model_labels}
        loaded_per_model = Counter(r["model"] for r in records)

        total_created = total_updated = 0
        self.stdout.write("Reference data loaded:")
        for label in model_labels:
            created = max(0, after[label] - before[label])
            updated = loaded_per_model[label] - created
            total_created += created
            total_updated += updated
            self.stdout.write(
                f"  {label:40s} created={created:<3d} updated={updated:<3d} "
                f"(total rows now {after[label]})"
            )

        self.stdout.write(self.style.SUCCESS(
            f"Done. created={total_created} updated={total_updated} skipped=0"
        ))

    @staticmethod
    def _count(model_label: str) -> int:
        try:
            model = apps.get_model(model_label)
        except LookupError:
            return 0
        return model.objects.count()
