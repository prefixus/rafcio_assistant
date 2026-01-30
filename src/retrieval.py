"""
Hybrid retrieval engine combining vector similarity and keyword search.
Implemented using ChromaDB (vector) and Rank-BM25 (keyword).
"""

from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi  # type: ignore

from src.vectorstore import VectorStoreAdapter


class HybridRetriever:  # pylint: disable=too-few-public-methods
    """
    Retriever that combines semantic vector search with BM25 keyword search.
    """

    def __init__(
        self,
        vector_adapter: VectorStoreAdapter,
        top_k: int = 5,
        rrf_k: int = 60,
    ):
        self.vector_adapter = vector_adapter
        self.top_k = top_k
        self.rrf_k = rrf_k
        self._bm25: Optional[BM25Okapi] = None
        self._all_docs: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace/lowercase tokenizer for BM25."""
        return text.lower().split()

    def _initialize_bm25(self) -> None:
        """Fetch all documents from vector store and initialize BM25 index."""
        self._all_docs = self.vector_adapter.get_all_documents()
        if not self._all_docs:
            return

        tokenized_corpus = [self._tokenize(doc["text"]) for doc in self._all_docs]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Vector similarity and BM25.
        Results are merged using Reciprocal Rank Fusion (RRF).
        """
        k = top_k or self.top_k

        # 1. Vector Search
        vector_results = self.vector_adapter.similarity_search(query, k=k * 2)

        # 2. BM25 Search
        # Lazy initialize BM25 index
        if self._bm25 is None:
            self._initialize_bm25()

        bm25_results = []
        if self._bm25 and self._all_docs:
            tokenized_query = self._tokenize(query)
            # BM25 scores for all docs
            scores = self._bm25.get_scores(tokenized_query)
            # Pair scores with docs and sort
            doc_scores = sorted(
                zip(self._all_docs, scores), key=lambda x: x[1], reverse=True
            )
            # Take top 2*k
            bm25_results = [doc for doc, score in doc_scores[: k * 2]]

        # 3. Reciprocal Rank Fusion (RRF)
        return self._reciprocal_rank_fusion(vector_results, bm25_results, k=k)

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        """
        Merges two lists of results according to their rank.
        Formula: 1 / (rank + k_rrf)
        """
        rrf_scores: Dict[str, float] = {}

        # Process vector results
        for rank, doc in enumerate(vector_results):
            doc_id = doc["text"]  # Using text as ID for simplicity in merging
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                rank + 1 + self.rrf_k
            )

        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            doc_id = doc["text"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                rank + 1 + self.rrf_k
            )

        # Convert back to documents and sort by RRF score
        # Re-map results to get full metadata
        doc_map = {doc["text"]: doc for doc in vector_results + bm25_results}

        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        final_results = [doc_map[doc_id] for doc_id in sorted_ids[:k]]
        return final_results
