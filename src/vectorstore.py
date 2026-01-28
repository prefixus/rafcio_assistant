"""
Vector store abstraction layer for the RAG Assistant.
Supports multiple backends via an adapter pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStoreAdapter(ABC):
    """
    Abstract base class for vector store adapters.
    """

    @abstractmethod
    def add_documents(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the vector store."""

    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform a similarity search."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the current collection."""


class VectorStoreManager:  # pylint: disable=too-few-public-methods
    """
    Manager to handle vector store initialization and adapter selection.
    """

    def __init__(self, provider: str, **kwargs):
        self.provider = provider.lower()
        self.kwargs = kwargs
        self._adapter: Optional[VectorStoreAdapter] = None

    def get_adapter(self) -> VectorStoreAdapter:
        """
        Returns the appropriate adapter based on the configured provider.
        """
        if self._adapter:
            return self._adapter

        if self.provider == "chroma":
            # Chroma adapter will be implemented in Step 5
            # For now, it will raise NotImplementedError until Step 5
            raise NotImplementedError("ChromaVectorStoreAdapter not yet implemented.")

        if self.provider == "mock":
            # Mock adapter for Step 3
            # pylint: disable=import-outside-toplevel
            from tests.mock_vectorstore import MockVectorStoreAdapter

            self._adapter = MockVectorStoreAdapter(**self.kwargs)
            return self._adapter

        raise ValueError(f"Unsupported vector store provider: {self.provider}")
