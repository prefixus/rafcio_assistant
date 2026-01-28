"""
Vector store abstraction layer for the RAG Assistant.
Supports multiple backends via an adapter pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, cast


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


class ChromaVectorStoreAdapter(VectorStoreAdapter):
    """
    Adapter for ChromaDB vector store.
    """

    def __init__(self, persist_directory: str, collection_name: str):
        # pylint: disable=import-outside-toplevel
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.provider = "chroma"

    def add_documents(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the Chroma collection."""
        # In a real app, IDs should be UUIDs or hashed
        ids = [f"id_{i}" for i in range(len(texts))]
        # Note: Embedding generation should ideally happen here or be passed in.
        self.collection.add(documents=texts, metadatas=cast(Any, metadatas), ids=ids)

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform a similarity search in Chroma."""
        results = self.collection.query(query_texts=[query], n_results=k)

        # Format results to match the expected output
        formatted_results = []
        if results["documents"] and results["metadatas"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                formatted_results.append({"text": doc, "metadata": meta})
        return formatted_results

    def delete_collection(self) -> None:
        """Delete and recreate the Chroma collection."""
        name = self.collection.name
        self.client.delete_collection(name=name)
        self.collection = self.client.get_or_create_collection(name=name)


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
            return ChromaVectorStoreAdapter(
                persist_directory=self.kwargs.get(
                    "persist_directory", "./data/chroma_db"
                ),
                collection_name=self.kwargs.get("collection_name", "rafcio_assistant"),
            )

        if self.provider == "mock":
            # Mock adapter for Step 3
            # pylint: disable=import-outside-toplevel
            from tests.mock_vectorstore import MockVectorStoreAdapter

            self._adapter = MockVectorStoreAdapter(**self.kwargs)
            return self._adapter

        raise ValueError(f"Unsupported vector store provider: {self.provider}")
