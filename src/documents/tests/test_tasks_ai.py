from unittest import mock

import pytest

from documents.models import Document
from documents.models import DocumentAITag
from documents.models import Tag
from documents.tasks_ai import process_document_ai


@pytest.mark.django_db
@mock.patch("documents.tasks_ai.OpenAIService")
def test_process_document_ai_adds_generated_tags(mock_openai_service) -> None:
    document = Document.objects.create(title="Test", content="invoice content")

    mock_openai_service.return_value.enrich_document.return_value = {
        "summary": "Invoice summary",
        "category": "Invoice",
        "confidence": 0.9,
        "embedding": None,
        "tags": [
            {"tag": "invoice", "confidence": 0.95},
            {"tag": "tax", "confidence": 0.81},
        ],
    }

    process_document_ai.run(document.pk)

    document.refresh_from_db()
    assert document.ai_status == "completed"
    assert set(document.tags.values_list("name", flat=True)) == {"invoice", "tax"}
    assert set(
        DocumentAITag.objects.filter(document=document).values_list(
            "tag",
            "approved",
        ),
    ) == {("invoice", True), ("tax", True)}


@pytest.mark.django_db
@mock.patch("documents.tasks_ai.OpenAIService")
def test_process_document_ai_reuses_existing_tag_case_insensitively(
    mock_openai_service,
) -> None:
    existing = Tag.objects.create(name="Invoice")
    document = Document.objects.create(title="Test", content="invoice content")

    mock_openai_service.return_value.enrich_document.return_value = {
        "tags": [{"tag": "invoice", "confidence": 0.95}],
    }

    process_document_ai.run(document.pk)

    assert list(document.tags.values_list("pk", flat=True)) == [existing.pk]
    assert Tag.objects.filter(name__iexact="invoice").count() == 1
