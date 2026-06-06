import json
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


CLASSIFICATION_PROMPT_TEMPLATE = """
You are an AI document classifier.

Analyze this OCR text and return:
- category
- tags
- confidence score
- summary

Return ONLY valid JSON with these keys:
- "category": a concrete document category such as "Invoice", "Receipt",
  "Contract", "Medical", "Legal", "Tax", "Identification", or "Other"
- "tags": 3 to 8 lowercase tags, each as an object with "tag" and "confidence"
- "confidence": a number from 0 to 1 for the overall classification
- "summary": a concise summary of the actual document text

Do not return placeholder values. Do not copy these instructions into the
answer.

Document text:
{ocr_text}
""".strip()


class OpenAIService:
    def __init__(self):
        self.provider = os.getenv("AI_PROVIDER", "openai").lower()
        if self.provider == "ollama":
            self.client = OpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1"),
                api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            )
            self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
            self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL")
        else:
            self.client = OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                api_key=os.getenv("OPENAI_API_KEY") or None,
            )
            self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
            self.embedding_model = os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                "text-embedding-3-small",
            )

    def _load_json_response(self, content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(content[start : end + 1])

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

        kwargs = {
            "model": self.chat_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You classify documents.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0,
        }
        if self.provider != "ollama":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content

        return self._load_json_response(content)

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

        if not self.embedding_model:
            return None

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
