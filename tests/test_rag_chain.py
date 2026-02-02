"""
Tests for the RAG chain.
Includes unit tests (mocked) and functional tests (requiring LM Studio).
"""

import os
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.rag_chain import RAGChain, format_docs


class SimpleFakeChatModel(BaseChatModel):
    """A simple fake chat model for testing purposes."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="This is a mocked answer."))
            ]
        )

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"


def test_format_docs():
    """Unit test for document formatting."""
    docs = [
        {"text": "Rafcio Assistant is cool.", "metadata": {"source": "docs"}},
        {"text": "It uses LangChain.", "metadata": {"source": "web"}},
    ]
    formatted = format_docs(docs)
    assert "Content: Rafcio Assistant is cool." in formatted
    assert "Source: docs" in formatted
    assert "Content: It uses LangChain." in formatted
    assert "Source: web" in formatted
    assert "---" in formatted


@patch("src.rag_chain.ChatOpenAI", new=SimpleFakeChatModel)
def test_rag_chain_unit_invoke():
    """Unit test for the RAG chain using mocked components."""
    # Setup mock retriever
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {"text": "Relevant context.", "metadata": {"source": "test_src"}}
    ]

    # Initialize chain
    chain = RAGChain(retriever=mock_retriever)

    # Invoke
    answer = chain.invoke("How does it work?")

    # Assertions
    assert answer == "This is a mocked answer."
    mock_retriever.search.assert_called_once_with("How does it work?")


# --- Functional Tests ---

REQUIRES_LM_STUDIO = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping local service test in GitHub Actions",
)


@REQUIRES_LM_STUDIO
def test_rag_chain_functional():
    """
    Functional test for RAG chain against a live LM Studio instance.
    Uses a mock retriever but real LLM to verify LCEL connectivity and prompt.
    """
    # Use real retriever interface but controlled mock data
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {
            "text": "The secret code for this test is 'JUPYTER-123'.",
            "metadata": {"source": "test_env"},
        }
    ]

    # Initialize chain with real LLM (assuming LM Studio is running)
    # If LM Studio is not running, this will fail or timeout - expected for functional tests.
    chain = RAGChain(retriever=mock_retriever)

    try:
        answer = chain.invoke("What is the secret code for this test?")

        assert "JUPYTER-123" in answer
        print(f"\n[Functional Test Output] Answer: {answer}")
    except Exception as e:  # pylint: disable=broad-except
        pytest.fail(f"Functional test failed due to connectivity or LLM error: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
