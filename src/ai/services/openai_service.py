import json
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


CLASSIFICATION_PROMPT_TEMPLATE = """
You are an AI document classifier.

Analyze this OCR text and return:
- category
- tags
- confidence score
- summary

Return ONLY valid JSON:

{{
  "category": "category_name",
  "tags": [
    {{
      "tag": "short_tag",
      "confidence": 0.95
    }}
  ],
  "confidence": 0.95,
  "summary": "Concise document summary."
}}

Document text:
{ocr_text}
""".strip()


class OpenAIService:
    def __init__(self):
        self.client = OpenAI()

        self.chat_model = "gpt-4.1-mini"

        self.embedding_model = "text-embedding-3-small"

    def generate_summary(self, text: str) -> str:
        """
        Generate a concise summary for the document.
        """

        prompt = f"""
        You are an AI document assistant.

        Generate a concise summary for the following document.

        Keep the summary under 150 words.

        Document:
        {text}
        """

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You summarize documents.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    def classify_document(self, text: str) -> dict:
        """
        Predict category, tags, confidence score, and summary.
        """

        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(ocr_text=text)

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You classify documents.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        return json.loads(content)

    def generate_tags(self, text: str) -> list:
        """
        Generate AI tags for the document.
        """

        prompt = f"""
        You are an AI tagging system.

        Generate 5-10 useful tags for this document.

        Rules:
        - lowercase
        - short tags
        - no duplicates
        - relevant only

        Return ONLY valid JSON:

        {{
          "tags": [
            {{
              "tag": "invoice",
              "confidence": 0.95
            }}
          ]
        }}

        Document:
        {text}
        """

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You generate document tags.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        parsed = json.loads(content)

        return parsed.get("tags", [])

    def generate_embedding(self, text: str) -> list:
        """
        Generate vector embedding for semantic search.
        """

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )

        return response.data[0].embedding

    def enrich_document(self, text: str) -> dict:
        """
        Full AI enrichment pipeline.
        """

        logger.info("Starting AI enrichment")

        classification = self.classify_document(text)

        embedding = self.generate_embedding(text)

        result = {
            "summary": classification.get("summary"),
            "category": classification.get("category"),
            "confidence": classification.get("confidence"),
            "tags": classification.get("tags", []),
            "embedding": embedding,
        }

        logger.info("AI enrichment completed")

        return result
