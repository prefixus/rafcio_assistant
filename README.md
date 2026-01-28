# Personal RAG Assistant (Rafcio Assistant)

A modular Python-based agentic assistant designed for RAG (Retrieval-Augmented Generation), tool calling, and LLM evaluation. This project serves as a hands-on technical demonstration of building local-first AI agents.

## 🚀 Project Goals
- **Local-First AI:** Powered by [LM Studio](https://lmstudio.ai/) for full privacy and control.
- **Advanced RAG:** Implementation of hybrid search (Vector + BM25) and reciprocal rank fusion.
- **Agentic Workflows:** Multi-step reasoning using LangGraph and MCP (Model Context Protocol).
- **Quality Focused:** Automated evaluation using DeepEval for answer relevancy and faithfulness.

## 🏗️ Technology Stack
- **Runtime:** Python 3.11+ with `uv`
- **Orchestration:** LangChain & LangGraph
- **Vector Store:** ChromaDB (Embedded)
- **LLM Provider:** LM Studio (OpenAI-compatible API)
- **Evaluation:** DeepEval
- **Linters:** Ruff, MyPy, PyLint (via Pre-commit)

## 🛠️ Setup & Configuration

### 1. Prerequisites
- Install [uv](https://docs.astral.sh/uv/)
- Install [LM Studio](https://lmstudio.ai/)

### 2. Model Setup
Download and load the following models in LM Studio:
- **LLM:** `gpt-oss:20b` (or similar for chat/reasoning)
- **Embeddings:** `text-embedding-bge-m3` (or `gemma-embedding`)

Ensure the local server is running at `http://localhost:1234`.

### 3. Installation
```bash
# Clone the repository
# (Assuming you are already in the project directory)

# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### 4. Configuration
Create a `.env` file (see template in `technical_specification.md` or copy an existing one):
```env
API_URL=http://127.0.0.1:1234
LLM_MODEL=openai/gpt-oss-20b
EMB_MODEL=text-embedding-bge-m3

# Feature Flags
ENABLE_HYBRID_SEARCH=true
ENABLE_MCP_TOOLS=true
```

## 📂 Project Structure
- `src/`: Core logic (config, embeddings, vectorstore, agents)
- `knowledge_base/`: Directory for documents to ingest (.txt, .md, .pdf)
- `workspace/`: Working area for agent file operations
- `tests/`: Quality and connectivity tests
- `scripts/`: Utility scripts (ingestion, linting)

## 🧪 Testing & Quality
Run the linting suite:
```bash
./scripts/lint.sh
```

Run specific tests:
```bash
# Test settings
uv run python tests/test_config.py

# Test embeddings connectivity
uv run pytest tests/test_embeddings.py -s
```

## 📜 License
MIT
