"""
Smoke-test the read-only API using Django's in-process test client.

Does NOT require a running server and works on SQLite. Checks status codes and
basic JSON response shape for each endpoint.

Output style:
    [OK]   /api/sections/ returned 200 (2 items)
    [WARN] /api/exercises/ returned empty list
    Result: PASS_WITH_WARNINGS

Usage:
    python manage.py smoke_test_api
"""

from django.core.management.base import BaseCommand
from django.test import Client

ENDPOINTS = [
    "/api/sections/",
    "/api/subjects/",
    "/api/chapters/",
    "/api/concepts/",
    "/api/exams/",
    "/api/exercises/",
]


class Command(BaseCommand):
    help = "Smoke-test the read-only API endpoints with the Django test client."

    def handle(self, *args, **options):
        # Use a host that is in ALLOWED_HOSTS (the test client defaults to
        # 'testserver', which a standalone command — unlike the test runner —
        # does not auto-allow).
        client = Client(HTTP_HOST="localhost")
        had_fail = False
        had_warn = False

        for url in ENDPOINTS:
            try:
                resp = client.get(url)
            except Exception as exc:
                had_fail = True
                self.stdout.write(self.style.ERROR(f"[FAIL] {url} raised {exc}"))
                continue

            if resp.status_code != 200:
                had_fail = True
                self.stdout.write(self.style.ERROR(f"[FAIL] {url} returned {resp.status_code}"))
                continue

            items = self._extract_items(resp)
            if items is None:
                had_fail = True
                self.stdout.write(self.style.ERROR(
                    f"[FAIL] {url} returned 200 but unexpected JSON shape"))
            elif len(items) == 0:
                had_warn = True
                self.stdout.write(self.style.WARNING(f"[WARN] {url} returned empty list"))
            else:
                self.stdout.write(self.style.SUCCESS(
                    f"[OK] {url} returned 200 ({len(items)} items)"))

        self.stdout.write("-" * 50)
        if had_fail:
            self.stdout.write(self.style.ERROR("Result: FAIL"))
            raise SystemExit(1)
        if had_warn:
            self.stdout.write(self.style.WARNING("Result: PASS_WITH_WARNINGS"))
        else:
            self.stdout.write(self.style.SUCCESS("Result: PASS"))

    @staticmethod
    def _extract_items(resp):
        """Return the list of items from a DRF response, or None if shape is wrong."""
        try:
            data = resp.json()
        except ValueError:
            return None
        if isinstance(data, dict) and "results" in data:   # paginated
            return data["results"] if isinstance(data["results"], list) else None
        if isinstance(data, list):                          # unpaginated
            return data
        return None
