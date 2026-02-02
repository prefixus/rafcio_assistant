# Implementation Plan: Personal RAG Assistant

This document outlines the step-by-step implementation plan for the Personal RAG Assistant, based on the `technical_specification.md` and `core_functionality.md`.

## Phase 1: Foundation (1 hour)
**Goal:** Set up the development environment and core infrastructure.

1.  **[ ] Environment Setup**
    *   Initialize project using `uv`: `uv init` (if not already done).
    *   Install core dependencies: `langchain`, `langgraph`, `chromadb`, `rank-bm25`, `pydantic`, `python-dotenv`, `pytest`, `ruff`, `mypy`.
    *   Create `.env` file based on the template in `technical_specification.md`.
    *   Configure `.gitignore` to exclude `.venv`, `__pycache__`, and data directories.

2.  **[ ] LM Studio Configuration**
    *   Ensure LM Studio is running.
    *   Load models: `gpt-oss:20b` (for chat) and `gemma-embedding` (for embeddings).
    *   Verify API connectivity to `http://localhost:1234/v1`.

3.  **[ ] Core Infrastructure Classes**
    *   `src/config.py`: Implement settings management using `pydantic-settings` or `dotenv`.
    *   `src/embeddings.py`: Implement a custom LangChain embedding class that wraps the LM Studio `/embeddings` endpoint.
    *   `src/vectorstore.py`: Implement `VectorStoreManager` to handle ChromaDB initialization, persistence, and collection management.

4.  **[ ] Initial Validation**
    *   Create a script `tests/test_foundation.py` to verify that embeddings can be generated and a simple document can be stored/retrieved from ChromaDB.

---

## Phase 2: RAG Pipeline (1.5 hours)
**Goal:** Implement document ingestion and hybrid search.

1.  **[x] Document Ingestion Path**
    *   Create `./knowledge_base/` and `./workspace/` directories.
    *   `scripts/ingest.py`: Implement the ingestion pipeline.
        *   Load `.txt`, `.md`, and `.pdf` files.
        *   Use `RecursiveCharacterTextSplitter` (512 chars, 128 overlap as per config).
        *   Batch embed and store in ChromaDB with metadata.

2.  **[x] Hybrid Retrieval Engine**
    *   `src/retrieval.py`: Implement the `HybridRetriever`.
        *   Vector search via ChromaDB.
        *   Keyword search via `rank-bm25`.
        *   Reciprocal Rank Fusion (RRF) to combine results.
        *   Configurable `top_k`.

3.  **[x] Basic RAG Chain**
    *   `src/rag_chain.py`: Create a standard LangChain Expression Language (LCEL) chain for basic RAG (Retrieve -> Augment -> Generate).
    *   `tests/test_rag_chain.py`: Unit and functional tests implemented and verified.

4.  **[ ] Validation**
    *   Verify that the assistant can answer questions based on a sample document in `./knowledge_base/`.

---

## Phase 3: Agentic Layer & MCP Tools (2 hours)
**Goal:** Implement the agentic workflow and functional tools.

1.  **[ ] MCP Server & Tools Implementation**
    *   `src/tools/mcp_server.py`: Define tools using the MCP pattern (Model Context Protocol).
    *   Implement the recommended tool subset:
        *   `search_knowledge_base(query: str)`: Interface to FR2 hybrid search.
        *   `explain_concept(term: str, context: str)`: Specialized RAG-augmented generation.
        *   `code_snippet_manager`: Tools to save and search code patterns.
        *   `kubernetes_command_generator`: Natural language to `kubectl`.
    *   `src/tools/tool_schemas.py`: Define JSON schemas for all tools to ensure clear LLM reasoning.

2.  **[ ] LangGraph Agent Orchestration**
    *   `src/agent.py`: Implement the ReAct agent pattern using LangGraph.
    *   Define `MessagesState` and the graph with nodes: `agent`, `tools`.
    *   Implement conditional edges for routing based on tool calls.
    *   Integrate `ConversationBufferWindowMemory` (FR4).

3.  **[ ] Interactive CLI**
    *   `src/cli.py`: Build a premium CLI interface using `rich` for formatting.
    *   Support streaming output for both "thinking/tool calls" and final answers.

4.  **[ ] Validation**
    *   Test multi-step queries (e.g., "Explain what RAG is and then save a code snippet for a ChromaDB retriever").

---

## Phase 4: Quality & Testing (1 hour)
**Goal:** Quantify performance and ensure reliability.

1.  **[ ] DeepEval Integration**
    *   `tests/test_rag_quality.py`: Implement the DeepEval test suite.
    *   Create 5-10 "Golden Q&A" pairs.
    *   Configure `LMStudioModel` as the custom judge.

2.  **[ ] Metric Evaluation**
    *   Run tests for Answer Relevancy, Faithfulness, and Context Recall.
    *   Target scores: > 0.7.

3.  **[ ] Conversation Persistence**
    *   Verify chat history is correctly stored in `./data/conversations.db` (SQLite).

---

## Phase 5: Final Polish (30 mins)
**Goal:** Clean up and prepare for presentation.

1.  **[ ] Linter & Quality Check**
    *   Run `ruff`, `mypy`, and `pylint` on the codebase.
    *   Ensure all tests pass.

2.  **[ ] Documentation**
    *   Update `README.md` with setup instructions and a brief architecture overview.
    *   Prepare "Interview Talking Points" based on the implementation (e.g., why hybrid search, chunking trade-offs).

---

## Technical Success Checklist
- [ ] LLM & Embeddings running via LM Studio
- [ ] ChromaDB persists to disk
- [ ] Hybrid Search (Vector + BM25) implemented
- [ ] LangGraph agent handles tool routing correctly
- [ ] Streaming CLI shows "thinking" process
- [ ] DeepEval scores > 0.7 for core tests
