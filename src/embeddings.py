"""
This module provides a custom LangChain Embeddings class for LM Studio.
"""

from typing import List, Optional
import httpx
from langchain_core.embeddings import Embeddings
from src.config import settings


class LMStudioEmbeddings(Embeddings):
    """
    Custom LangChain Embeddings class for LM Studio.
    Wraps the OpenAI-compatible /embeddings endpoint.
    """

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or settings.embedding_model
        # Ensure base_url points to the v1 endpoint if not specified
        self.base_url = (base_url or settings.api_url).rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        payload = {"input": texts, "model": self.model}

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                # LM Studio follows OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
                return [item["embedding"] for item in data.get("data", [])]
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to connect to LM Studio at {url}: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"An error occurred during embedding generation: {e}"
            ) from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        result = self._get_embeddings([text])
        return result[0] if result else []
