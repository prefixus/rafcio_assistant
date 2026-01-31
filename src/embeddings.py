"""
This module provides a custom LangChain Embeddings class for LM Studio.
"""

from typing import Any, List, Optional

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
        all_embeddings = []
        batch_size = 16  # Smaller batches are safer for local models

        try:
            with httpx.Client(timeout=300.0) as client:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    payload = {"input": batch, "model": self.model}
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    # LM Studio follows OpenAI format:
                    # {"data": [{"embedding": [...], "index": 0}, ...]}
                    batch_embeddings = [
                        item["embedding"] for item in data.get("data", [])
                    ]
                    all_embeddings.extend(batch_embeddings)
                return all_embeddings
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

    def embed_query(self, text: Any = "", **kwargs) -> Any:
        """
        Embed a single query or multiple queries.
        Supports:
        - LangChain style: embed_query(text="...") -> List[float]
        - ChromaDB style: embed_query(input=["..."]) -> List[List[float]]
        """
        if "input" in kwargs:
            input_val = kwargs["input"]
            if isinstance(input_val, list):
                return self.embed_documents(input_val)
            return self._get_embeddings([input_val])[0]

        if isinstance(text, list):
            return self.embed_documents(text)

        result = self._get_embeddings([text])
        return result[0] if result else []

    def __call__(self, input: List[str]) -> List[List[float]]:  # pylint: disable=redefined-builtin
        """
        Make the class compatible with ChromaDB's EmbeddingFunction protocol.
        """
        return self.embed_documents(input)

    def name(self) -> str:
        """Name of the embedding function."""
        return "lm_studio"
