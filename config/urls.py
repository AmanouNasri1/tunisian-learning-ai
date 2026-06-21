"""Root URL configuration. Admin + read-only reference API."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("backend.exam_intelligence.api.urls")),
]
