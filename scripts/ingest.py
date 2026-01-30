"""
Script to ingest documents from the knowledge base into ChromaDB.
Supports .txt, .md, and .pdf files.
"""

import os
from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.embeddings import LMStudioEmbeddings
from src.vectorstore import VectorStoreManager


def get_loader_map():
    """Returns a mapping of file extensions to their respective loaders."""
    return {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,  # Simplified for now, or use UnstructuredMarkdownLoader
    }


def ingest_documents():
    """Main function to load, split, and store documents."""
    print(f"--- Starting Ingestion from: {settings.knowledge_base_dir} ---")

    # 1. Load Documents
    documents: List[Document] = []

    # Check if directory exists
    if not os.path.exists(settings.knowledge_base_dir):
        print(
            f"Error: Knowledge base directory '{settings.knowledge_base_dir}' not found."
        )
        return

    # Using individual loaders for better control or DirectoryLoader with mapping
    for ext, loader_cls in get_loader_map().items():
        loader = DirectoryLoader(
            settings.knowledge_base_dir,
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            show_progress=True,
            use_multithreading=True,
        )
        loaded_docs = loader.load()
        print(f"Loaded {len(loaded_docs)} documents with extension {ext}")
        documents.extend(loaded_docs)

    if not documents:
        print("No documents found to ingest.")
        return

    # 2. Split Documents
    print(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks.")

    # 3. Initialize Vector Store
    print(f"Initializing Vector Store ({settings.vector_store_provider})...")
    embeddings = LMStudioEmbeddings()

    manager = VectorStoreManager(
        provider=settings.vector_store_provider,
        embedding_model=embeddings,
        persist_directory=settings.chroma_db_dir,
        collection_name=settings.vector_store_collection,
    )
    adapter = manager.get_adapter()

    # 4. Store in Vector Store
    # We need to extract text and metadata from LangChain Document objects
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    print(f"Storing {len(chunks)} chunks in {settings.vector_store_provider}...")
    adapter.add_documents(texts=texts, metadatas=metadatas)

    print("--- Ingestion Completed Successfully ---")


if __name__ == "__main__":
    ingest_documents()
