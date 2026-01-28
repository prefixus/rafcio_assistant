# Technical Specification: Personal RAG Assistant

## Project Overview
A modular Python-based agentic assistant using local LLMs via LM Studio, designed as a 4-6 hour hands-on training project covering RAG, tool calling, and LLM evaluation for your N-iX interview preparation.

## Core Architecture

### Technology Stack
- **Runtime:** Python 3.11+ with `uv` package manager
- **LLM Framework:** LangChain + LangGraph (agentic workflow orchestration)
- **LLM Provider:** LM Studio (OpenAI-compatible API) serving `gpt-oss:20b` and `gemma-embedding`
- **Vector Database:** ChromaDB (embedded mode - zero configuration, persistent storage)
- **Evaluation:** DeepEval (answer relevancy, faithfulness metrics)
- **Tool System:** MCP (Model Context Protocol) server for file operations and calculations
- **Configuration:** `.env` with feature flags
- **Code Linting:** `ruff`, `mypy`, `pylint`, `black` with `pre-commit` hooks
- **Testing:** `pytest` with `coverage` and `deepdiff`
- **CI/CD:** GitHub Actions with `pre-commit`, `pytest`, `coverage`, `black`, `ruff`, `mypy`, `pylint`

### Why ChromaDB?
ChromaDB runs embedded (no container needed), persists to disk automatically, and has native LangChain integration - ideal for rapid prototyping. You can upgrade to Qdrant later if needed. [justjoin](https://justjoin.it/job-offer/n-ix-senior-genai-engineer-warszawa-ai)

## Functional Requirements

### FR1: Document Ingestion Pipeline
**Purpose:** Learn RAG data preparation and chunking strategies

**Implementation:**
- Monitor a `./knowledge_base/` directory for `.txt`, `.md`, `.pdf` files
- Chunk documents using RecursiveCharacterTextSplitter (500 chars, 100 overlap)
- Generate embeddings via LM Studio `/embeddings` endpoint (gemma-embedding model)
- Store in ChromaDB collection with metadata (filename, chunk_id, timestamp)

**Key Interview Concepts:**
- Chunking trade-offs (size vs context preservation)
- Metadata filtering for hybrid search
- Embedding model selection criteria

### FR2: Hybrid Search & Retrieval
**Purpose:** Demonstrate advanced RAG beyond basic vector similarity

**Implementation:**
- Combine vector similarity (ChromaDB) with keyword search (BM25 via `rank-bm25` library)
- Rerank top 10 results using reciprocal rank fusion (RRF)
- Return top 3 chunks to LLM context

**Key Interview Concepts:**
- Why hybrid search improves recall
- Reranking vs single-pass retrieval
- Context window management

### FR3: Agentic Tool Calling via MCP
**Purpose:** Master function calling and multi-step reasoning

**Implementation:**
- Create MCP server with 3 tools:
  - `search_knowledge_base(query: str)` - triggers FR2 retrieval
  - `read_file(filepath: str)` - reads files from `./workspace/`
  - `calculate(expression: str)` - evaluates Python math expressions safely
- Use LangGraph ReAct agent pattern to route user queries to appropriate tools
- Stream responses with thought process visible

**Key Interview Concepts:**
- Tool schema definition (JSON Schema for LLM consumption)
- Agent decision loops (ReAct: Reason + Act)
- Error handling in tool execution

### FR4: Conversation Memory
**Purpose:** Maintain stateful multi-turn interactions

**Implementation:**
- Use LangChain's `ConversationBufferWindowMemory` (last 5 messages)
- Store chat history in SQLite (`./data/conversations.db`)
- Include conversation ID in all DeepEval test cases

**Key Interview Concepts:**
- Memory strategies (buffer vs summary vs entity)
- Context injection for personalization
- Token budget management

### FR5: Automated Testing with DeepEval
**Purpose:** Quantify LLM system performance

**Implementation:**
- Create test suite in `tests/test_rag_quality.py` with 5 golden Q&A pairs
- Metrics to evaluate:
  - Answer Relevancy (does answer address question?)
  - Faithfulness (is answer grounded in retrieved context?)
  - Context Recall (are all relevant docs retrieved?)
- Run tests via `pytest` with DeepEval integration

**Key Interview Concepts:**
- LLM-as-judge evaluation pattern
- Difference between retrieval metrics vs generation metrics
- When to use reference-free vs reference-based evaluation

## Project Structure
```
rag-assistant/
├── .env                          # Model configs + feature flags
├── pyproject.toml                # uv dependencies
├── knowledge_base/               # Documents to ingest
│   └── sample_docs.md
├── workspace/                    # Files agent can access
├── data/
│   ├── chroma_db/               # Vector store persistence
│   └── conversations.db         # Chat history
├── src/
│   ├── config.py                # Load .env, feature flags
│   ├── embeddings.py            # LM Studio embedding client
│   ├── vectorstore.py           # ChromaDB operations
│   ├── retrieval.py             # Hybrid search logic (FR2)
│   ├── tools/
│   │   ├── mcp_server.py       # MCP tool definitions
│   │   └── tool_schemas.py     # JSON schemas for LLM
│   ├── agent.py                 # LangGraph ReAct agent (FR3)
│   └── cli.py                   # Interactive chat interface
├── tests/
│   └── test_rag_quality.py      # DeepEval test cases
└── scripts/
    └── ingest.py                # One-time document loader
```

## Environment Configuration (.env)
```env
# LM Studio API
LM_STUDIO_BASE_URL=http://localhost:1234/v1
CHAT_MODEL=gpt-oss:20b
EMBEDDING_MODEL=gemma-embedding

# Feature Flags
ENABLE_HYBRID_SEARCH=true       # Toggle FR2
ENABLE_MCP_TOOLS=true           # Toggle FR3
ENABLE_CONVERSATION_MEMORY=true # Toggle FR4
ENABLE_DEEPEVAL=false           # Disable during development

# Retrieval Config
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K_RESULTS=3
```

## Implementation Phases

### Phase 1: Foundation (1 hour)
1. Initialize project: `uv init` and install dependencies
2. Configure LM Studio with both models loaded
3. Implement `embeddings.py` and `vectorstore.py`
4. Test embedding generation with simple script

**Validation:** Successfully store and retrieve a test document

### Phase 2: RAG Pipeline (1.5 hours)
1. Implement document ingestion (`scripts/ingest.py`)
2. Build hybrid search in `retrieval.py`
3. Create basic LangChain RAG chain (no agents yet)

**Validation:** Ask questions about ingested documents, verify source citation

### Phase 3: Agentic Layer (2 hours)
1. Define MCP tool schemas
2. Implement LangGraph ReAct agent
3. Build CLI with streaming output

**Validation:** Agent successfully routes queries to tools and retrieval

### Phase 4: Testing (1 hour)
1. Create 5 test cases covering different question types
2. Run DeepEval metrics
3. Document results and iterate on prompts

**Validation:** Achieve >0.7 on Answer Relevancy and Faithfulness

## Key Implementation Reminders

### LM Studio API Quirks
- LM Studio uses OpenAI format but may not support all parameters (e.g., `response_format` for JSON mode) [careers.n-ix](https://careers.n-ix.com/the-story-of-how-we-almost-hired-chatgpt-as-a-net-engineer/)
- Always set `temperature=0` for DeepEval runs to ensure reproducibility
- Embeddings endpoint: `POST /embeddings` with `{"input": "text", "model": "gemma-embedding"}`

### LangGraph Agent Pattern
```python
# Critical: Define state schema for agent memory
from langgraph.graph import StateGraph, MessagesState

# Agent must have access to:
# 1. Tools list
# 2. Chat history
# 3. Retrieval results
# Use conditional edges to route tool calls vs final answer
```

### ChromaDB Gotchas
- Collection names must be alphanumeric + underscores only [justjoin](https://justjoin.it/job-offer/n-ix-senior-genai-engineer-warszawa-ai)
- Persist directory must exist before client initialization
- Use `get_or_create_collection()` to avoid errors on restart

### DeepEval Configuration
```python
# Use LM Studio as judge model
from deepeval.models import DeepEvalBaseLLM

class LMStudioModel(DeepEvalBaseLLM):
    def load_model(self):
        return None  # No local loading needed

    def generate(self, prompt: str):
        # Call LM Studio API
        pass

# Set globally before running tests
set_global_custom_llm(LMStudioModel())
```

### MCP Tool Calling
- Tool schemas must include clear descriptions for LLM reasoning [linkedin](https://www.linkedin.com/posts/bisal-show_top-100-gen-ai-engineer-interview-questions-activity-7413609633233154048-kodt)
- Always return structured responses: `{"result": ..., "error": None}`
- Log all tool calls for debugging agent behavior

## Success Criteria
By end of implementation, you should be able to:
1. **Explain chunking strategy:** Why 500 chars? What happens with 2000?
2. **Demonstrate hybrid search:** Show BM25 catching keyword matches vector search misses
3. **Debug tool calling:** Agent chooses wrong tool - how to fix via prompt engineering?
4. **Interpret DeepEval scores:** Why did Faithfulness drop to 0.4? How to improve?
5. **Discuss production scaling:** What breaks when moving from ChromaDB to Qdrant cluster?

## Estimated Timeline
- **Setup + Phase 1:** 1 hour
- **Phase 2 (RAG):** 1.5 hours
- **Phase 3 (Agents):** 2 hours
- **Phase 4 (Testing):** 1 hour
- **Buffer for debugging:** 30 mins

**Total:** 6 hours with agentic IDE support

This specification focuses on **interview-relevant depth** over breadth - you'll touch every topic N-iX will ask about (RAG architecture, tool calling, evaluation, production considerations) while keeping scope manageable for same-day implementation.
