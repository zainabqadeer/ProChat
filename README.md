# ✦ ProChat: Advanced RAG Ecosystem

A production-grade **Retrieval-Augmented Generation** (RAG) ecosystem featuring an interactive Streamlit UI and a high-performance FastAPI backend. Built for precision, transparency, and multi-provider flexibility.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/Powered%20by-LangChain-green.svg)](https://langchain.com/)

---

## 🏗️ The TechLab RAG Pipeline

This project implements an **Advanced RAG** pattern that solves the limitations of "naive" vector search through a multi-stage intelligent retrieval process.

```mermaid
graph TD
    A[User Question] --> B[Multi-Query Generation]
    subgraph "Retrieval Engine"
    B --> C[Query 1]
    B --> D[Query 2]
    B --> E[Query 3]
    B --> F[Original]
    C & D & E & F --> G[pgvector Search]
    G --> H[Deduplication]
    end
    subgraph "Precision Layer"
    H --> I[FlashRank Reranking]
    I --> J[Contextual Compression]
    end
    J --> K[LLM Generation]
    K --> L[Structured Answer + Sources]
    
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#00d2ff,stroke:#333,stroke-width:2px
    style J fill:#00d2ff,stroke:#333,stroke-width:2px
```

### ✨ Advanced Features
- **Multi-Query Expansion**: LLM generates 3 variations of the question to ensure diverse retrieval.
- **Hybrid Embedding Support**: Auto-detects and uses **Google Gemini** (`embedding-002`) or **OpenAI** (`text-embedding-3-small`).
- **Cross-Encoder Reranking**: Uses **FlashRank** to re-score hits for high precision.
- **Contextual Compression**: Extracts only relevant sentences, reducing noise and token costs.
- **Streaming Support**: Real-time token generation via both UI and API (SSE).

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Orchestration** | LangChain |
| **Vector Database** | NeonDB (PostgreSQL + pgvector) |
| **LLMs** | Google Gemini 1.5 Pro/Flash & OpenAI GPT-4o-mini |
| **UI** | Streamlit (Glassmorphism design) |
| **Backend API** | FastAPI + Uvicorn |
| **Package Manager**| `uv` (Faster, reliable Python tooling) |

---

## 📁 Project Structure

```bash
demo-chatbot/
├── app.py              # Streamlit Web Interface (Glassmorphism)
├── api.py              # FastAPI Backend (REST & SSE Streaming)
├── chain.py            # Core RAG logic & Pipeline definitions
├── ingest.py           # Data ingestion to NeonDB
├── docs/               # 📂 Source documents (PDF/TXT)
├── .env.example        # Environment template
└── pyproject.toml      # Project dependencies and config
```

---

## 🚀 Quick Start

### 1. Prerequisites
- [uv](https://docs.astral.sh/uv/) installed.
- A [NeonDB](https://neon.tech) project with `pgvector` enabled.

### 2. Configuration
```bash
cp .env.example .env
```
Fill in your credentials:
```env
OPENAI_API_KEY=sk-...
# OR
GOOGLE_API_KEY=AIza...

NEON_DATABASE_URL=postgresql://user:pass@ep-host.region.aws.neon.tech/neondb
```

### 3. Ingestion
Place your documents in `docs/`, then run:
```bash
uv run python ingest.py
```

### 4. Running the Ecosystem

#### **Option A: Streamlit UI (Frontend)**
```bash
uv run streamlit run app.py
```
> Access at `http://localhost:8501`

#### **Option B: FastAPI (Backend)**
```bash
uv run uvicorn api:app --reload --port 8000
```
> Explore API docs at `http://localhost:8000/docs`

---

## 🔌 API Documentation

The FastAPI backend provides several endpoints for integration:

- `GET /api/health`: Check system status and environment configuration.
- `GET /api/provider`: Returns the currently active LLM/Embedding provider.
- `POST /api/chat`: Synchronous RAG response.
- `POST /api/chat/stream`: Server-Sent Events (SSE) for streaming answers and pipeline stage updates.

**Example Request:**
```json
{
  "question": "How does the reranking work?",
  "system_prompt": "You are a TechLab expert.",
  "chat_history": []
}
```

---

## 🗄️ NeonDB Management

You can inspect your vector embeddings directly via the Neon SQL Console:

```sql
-- Count total document chunks
SELECT COUNT(*) FROM rag_documents;

-- View top chunks for a specific document
SELECT document, cmetadata FROM langchain_pg_embedding LIMIT 5;
```

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
