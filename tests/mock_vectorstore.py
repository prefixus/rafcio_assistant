"""
Mock vector store adapter for testing purposes.
"""

from typing import Any, Dict, List, Optional

from src.vectorstore import VectorStoreAdapter


class MockVectorStoreAdapter(VectorStoreAdapter):
    """
    In-memory mock vector store.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.collection: List[Dict[str, Any]] = []
        self.kwargs = kwargs
        self.provider = "mock"

    def add_documents(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the mock store."""
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            self.collection.append({"text": text, "metadata": metadata})

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform a mock similarity search (returns first k documents)."""
        return self.collection[:k]

    def delete_collection(self) -> None:
        """Clear the mock store."""
        self.collection = []

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents from the mock store."""
        return self.collection
