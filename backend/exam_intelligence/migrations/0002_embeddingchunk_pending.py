"""Make EmbeddingChunk.embedding nullable and add embedding lifecycle metadata.

Additive and migration-safe: 0001 is untouched, no data loss. Lets chunks be
created with text + metadata first (status 'pending') and embedded later.

Order matters: the HNSW index is first removed from Django's migration *state*
(a no-op at the database level — it is kept on Postgres, created by 0001, and was
never created on SQLite). This must happen before the AlterField on `embedding`,
because on SQLite an AlterField rebuilds the table and would otherwise try to
recreate the HNSW index from state, which SQLite cannot parse.
"""

import pgvector.django.vector
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("exam_intelligence", "0001_initial"),
    ]

    operations = [
        # Remove HNSW from state only; do NOT touch the database (keep it on Postgres).
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.RemoveIndex(model_name="embeddingchunk", name="embeddingchunk_hnsw"),
            ],
            database_operations=[],
        ),
        migrations.AlterField(
            model_name="embeddingchunk",
            name="content_type",
            field=models.CharField(
                choices=[
                    ("lesson", "Résumé de cours"),
                    ("exercise", "Énoncé d'exercice"),
                    ("question", "Question"),
                    ("correction", "Correction"),
                    ("rubric", "Barème"),
                    ("mistake", "Erreur fréquente"),
                    ("combined", "Contexte combiné de l'exercice"),
                ],
                max_length=16,
            ),
        ),
        migrations.AlterField(
            model_name="embeddingchunk",
            name="embedding",
            field=pgvector.django.vector.VectorField(blank=True, dimensions=1536, null=True),
        ),
        migrations.AlterField(
            model_name="embeddingchunk",
            name="model_name",
            field=models.CharField(blank=True, max_length=64),
        ),
        migrations.AddField(
            model_name="embeddingchunk",
            name="embedding_status",
            field=models.CharField(
                choices=[("pending", "En attente"), ("mock", "Mock déterministe"), ("ready", "Prêt")],
                db_index=True,
                default="pending",
                max_length=12,
            ),
        ),
        migrations.AddField(
            model_name="embeddingchunk",
            name="language",
            field=models.CharField(
                blank=True,
                choices=[("fr", "Français"), ("ar", "العربية"), ("darja", "Darija tunisienne")],
                default="",
                max_length=8,
            ),
        ),
    ]
