import logging

from celery import shared_task
from django.db import transaction
from openai import RateLimitError

from ai.services.openai_service import OpenAIService
from documents.models import Document
from documents.models import DocumentAITag
from documents.models import Tag

logger = logging.getLogger("paperless.ai_tasks")


def _is_insufficient_quota(exc: RateLimitError) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        return error.get("code") == "insufficient_quota"
    return "insufficient_quota" in str(exc)


def _get_tag_value(tag_data: dict | str, key: str, default=None):
    if isinstance(tag_data, dict):
        return tag_data.get(key, default)
    if key == "tag":
        return tag_data
    return default


def _clean_ai_tag_names(tags: list[dict | str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    for tag_data in tags:
        raw_name = _get_tag_value(tag_data, "tag", "")
        if not isinstance(raw_name, str):
            continue

        name = raw_name.strip()
        key = name.lower()
        if not name or key in seen:
            continue

        seen.add(key)
        names.append(name)

    return names


def _get_or_create_tags_by_name(tag_names: list[str]) -> list[Tag]:
    tags: list[Tag] = []

    for name in tag_names:
        tag = Tag.objects.filter(name__iexact=name).first()
        if tag is None:
            tag = Tag.objects.create(name=name)
        tags.append(tag)

    return tags


def _tag_name_key(tag_data: dict | str) -> str:
    raw_name = _get_tag_value(tag_data, "tag", "")
    return raw_name.strip().lower() if isinstance(raw_name, str) else ""


def _apply_ai_tags(document: Document, tag_names: list[str]) -> set[str]:
    if not tag_names:
        return set()

    tags = _get_or_create_tags_by_name(tag_names)
    tag_ids: set[int] = set()

    for tag in tags:
        tag_ids.add(tag.pk)
        tag_ids.update(tag.get_ancestors_pks())

    document.tags.add(*tag_ids)
    document.save(update_fields=["modified"])

    return {tag.name.lower() for tag in tags}


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
        tag_names = _clean_ai_tag_names(tags)

        with transaction.atomic():
            applied_tag_names = _apply_ai_tags(document, tag_names)

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
                        approved=_tag_name_key(tag_data) in applied_tag_names,
                    )
                    for tag_data in tags
                    if _get_tag_value(tag_data, "tag")
                ],
            )

    except RateLimitError as exc:
        Document.objects.filter(pk=document.pk).update(ai_status="failed")
        if _is_insufficient_quota(exc):
            logger.error(
                "AI processing failed for document %s: OpenAI quota exceeded",
                document.pk,
            )
            return f"AI processing failed for document {document.pk}: quota exceeded"
        logger.exception("AI processing failed for document %s", document.pk)
        raise
    except Exception:
        Document.objects.filter(pk=document.pk).update(ai_status="failed")
        logger.exception("AI processing failed for document %s", document.pk)
        raise

    return f"AI processing completed for document {document.pk}"
