import logging

from celery import shared_task
from django.db import transaction

from ai.services.openai_service import OpenAIService
from documents.models import Document
from documents.models import DocumentAITag

logger = logging.getLogger("paperless.ai_tasks")


def _get_tag_value(tag_data: dict | str, key: str, default=None):
    if isinstance(tag_data, dict):
        return tag_data.get(key, default)
    if key == "tag":
        return tag_data
    return default


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 5},
)
def process_document_ai(self, document_id: int) -> str:
    document = Document.objects.get(pk=document_id)
    document.ai_status = "processing"
    document.save(update_fields=["ai_status"])

    try:
        result = OpenAIService().enrich_document(document.content or "")
        tags = result.get("tags", [])

        with transaction.atomic():
            Document.objects.filter(pk=document.pk).update(
                ai_status="completed",
                ai_summary=result.get("summary"),
                ai_category=result.get("category"),
                ai_confidence=result.get("confidence"),
                embedding=result.get("embedding"),
            )

            document.ai_tags.all().delete()
            DocumentAITag.objects.bulk_create(
                [
                    DocumentAITag(
                        document=document,
                        tag=_get_tag_value(tag_data, "tag", ""),
                        confidence_score=(
                            _get_tag_value(tag_data, "confidence", 0)
                            or _get_tag_value(tag_data, "confidence_score", 0)
                        ),
                    )
                    for tag_data in tags
                    if _get_tag_value(tag_data, "tag")
                ],
            )

    except Exception:
        Document.objects.filter(pk=document.pk).update(ai_status="failed")
        logger.exception("AI processing failed for document %s", document.pk)
        raise

    return f"AI processing completed for document {document.pk}"
