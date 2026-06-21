from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("exam_intelligence", "0002_embeddingchunk_pending"),
    ]

    operations = [
        migrations.AlterField(
            model_name="aiinteraction",
            name="mode",
            field=models.CharField(
                choices=[
                    ("explain_fr", "Explication FR"),
                    ("explain_ar", "Explication AR"),
                    ("explain_darja", "Explication Darja"),
                    ("hint_only", "Indice"),
                    ("full_correction", "Correction complète"),
                    ("bac_style_answer", "Réponse type Bac"),
                    ("identify_mistakes", "Repérage d'erreurs"),
                    ("generate_similar", "Exercice similaire"),
                    ("revision_summary", "Fiche de révision"),
                    ("explain_points_lost", "Points perdus"),
                    ("tutor_answer", "Réponse tuteur"),
                ],
                max_length=24,
            ),
        ),
        migrations.AddField(
            model_name="aiinteraction",
            name="provider",
            field=models.CharField(blank=True, max_length=32),
        ),
        migrations.AddField(
            model_name="aiinteraction",
            name="retrieval_mode",
            field=models.CharField(blank=True, max_length=48),
        ),
        migrations.AddField(
            model_name="aiinteraction",
            name="refused",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="aiinteraction",
            name="refusal_reason",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="aiinteraction",
            name="warnings",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
