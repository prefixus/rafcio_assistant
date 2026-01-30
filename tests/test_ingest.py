"""
Unit tests for the ingestion script.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from scripts.ingest import get_loader_map, ingest_documents


def test_get_loader_map():
    """Verify that the loader map contains the expected extensions."""
    loader_map = get_loader_map()
    assert ".pdf" in loader_map
    assert ".txt" in loader_map
    assert ".md" in loader_map


@patch("scripts.ingest.os.path.exists")
@patch("scripts.ingest.DirectoryLoader")
@patch("scripts.ingest.RecursiveCharacterTextSplitter")
@patch("scripts.ingest.LMStudioEmbeddings")
@patch("scripts.ingest.VectorStoreManager")
def test_ingest_documents_success(
    mock_manager_cls,
    mock_embeddings_cls,
    mock_splitter_cls,
    mock_loader_cls,
    mock_exists,
):
    """Test the full ingestion flow with mocked components."""
    # 1. Setup mocks
    mock_exists.return_value = True

    # Mock Loader
    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        Document(page_content="test content", metadata={"source": "test.txt"})
    ]
    mock_loader_cls.return_value = mock_loader

    # Mock Splitter
    mock_splitter = MagicMock()
    mock_splitter.split_documents.return_value = [
        Document(page_content="chunk1", metadata={"source": "test.txt"}),
        Document(page_content="chunk2", metadata={"source": "test.txt"}),
    ]
    mock_splitter_cls.return_value = mock_splitter

    # Mock Vector Store Manager and Adapter
    mock_adapter = MagicMock()
    mock_manager = MagicMock()
    mock_manager.get_adapter.return_value = mock_adapter
    mock_manager_cls.return_value = mock_manager

    # 2. Execute
    ingest_documents()

    # 3. Assertions
    # Check if DirectoryLoader was called (at least once for each extension)
    assert mock_loader_cls.call_count == len(get_loader_map())

    # Check if splitter was called
    mock_splitter.split_documents.assert_called_once()

    # Check if embeddings were initialized
    mock_embeddings_cls.assert_called_once()

    # Check if vector store was initialized and documents were added
    mock_manager_cls.assert_called_once()
    mock_adapter.add_documents.assert_called_once_with(
        texts=["chunk1", "chunk2"],
        metadatas=[{"source": "test.txt"}, {"source": "test.txt"}],
    )


@patch("scripts.ingest.os.path.exists")
def test_ingest_documents_no_directory(mock_exists):
    """Test behavior when knowledge base directory doesn't exist."""
    mock_exists.return_value = False

    with patch("builtins.print") as mock_print:
        ingest_documents()
        mock_print.assert_any_call(
            "Error: Knowledge base directory './knowledge_base' not found."
        )


@patch("scripts.ingest.os.path.exists")
@patch("scripts.ingest.DirectoryLoader")
def test_ingest_documents_no_files(mock_loader_cls, mock_exists):
    """Test behavior when no files are found."""
    mock_exists.return_value = True

    mock_loader = MagicMock()
    mock_loader.load.return_value = []
    mock_loader_cls.return_value = mock_loader

    with patch("builtins.print") as mock_print:
        ingest_documents()
        mock_print.assert_any_call("No documents found to ingest.")
