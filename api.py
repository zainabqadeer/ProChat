"""
api.py — FastAPI Backend for the TechLab RAG Chatbot
=====================================================
Endpoints:
  POST /api/chat          — Full pipeline (non-streaming)
  POST /api/chat/stream   — Streaming SSE (token-by-token + stage updates)
  GET  /api/health        — Health check
  GET  /api/provider      — Active LLM provider

Usage:
    uvicorn api:app --reload --port 8000
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from chain import (
    ask as rag_ask,
    ask_stream as rag_ask_stream,
    DEFAULT_SYSTEM_PROMPT,
    get_provider_name,
)

# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="TechLab RAG Chatbot API",
    description="Advanced RAG pipeline: Multi-query → Retrieval → Reranking → Compression → Streaming Generation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., description="The user's question")
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT, description="System prompt for the LLM")
    chat_history: list[ChatMessage] = Field(default=[], description="Previous conversation messages")


class ChunkResponse(BaseModel):
    content: str
    source: str
    rerank_score: float | None = None
    compressed_content: str | None = None


class ChatResponse(BaseModel):
    answer: str
    provider: str
    original_query: str
    generated_queries: list[str]
    total_before_dedup: int
    chunks_retrieved: list[ChunkResponse]
    chunks_after_rerank: list[ChunkResponse]
    chunks_compressed: list[ChunkResponse]


class HealthResponse(BaseModel):
    status: str
    google_api_key: bool
    openai_api_key: bool
    neon_database_url: bool


# ── Helper ───────────────────────────────────────────────────────────────────

def _validate_keys():
    """Raise HTTPException if no API keys are configured."""
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="No API key configured. Set GOOGLE_API_KEY or OPENAI_API_KEY.")
    if not os.getenv("NEON_DATABASE_URL"):
        raise HTTPException(status_code=400, detail="NEON_DATABASE_URL not configured.")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check if all required environment variables are set."""
    return HealthResponse(
        status="ok",
        google_api_key=bool(os.getenv("GOOGLE_API_KEY")),
        openai_api_key=bool(os.getenv("OPENAI_API_KEY")),
        neon_database_url=bool(os.getenv("NEON_DATABASE_URL")),
    )


@app.get("/api/provider")
async def get_provider():
    """Return which LLM/embedding provider is currently active."""
    return {"provider": get_provider_name()}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming: full pipeline, returns complete result."""
    _validate_keys()
    try:
        history = [{"role": m.role, "content": m.content} for m in request.chat_history]
        result = rag_ask(
            question=request.question,
            system_prompt=request.system_prompt,
            chat_history=history,
        )
        return ChatResponse(
            answer=result.answer,
            provider=result.provider,
            original_query=result.original_query,
            generated_queries=result.generated_queries,
            total_before_dedup=result.total_before_dedup,
            chunks_retrieved=[
                ChunkResponse(content=c.content, source=c.source)
                for c in result.chunks_retrieved
            ],
            chunks_after_rerank=[
                ChunkResponse(content=c.content, source=c.source, rerank_score=c.rerank_score)
                for c in result.chunks_after_rerank
            ],
            chunks_compressed=[
                ChunkResponse(
                    content=c.content, source=c.source,
                    rerank_score=c.rerank_score, compressed_content=c.compressed_content,
                )
                for c in result.chunks_compressed
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming SSE endpoint. Returns Server-Sent Events:
      data: {"type": "stage", "stage": "multi_query", "data": {...}}
      data: {"type": "token", "content": "Hello"}
      data: {"type": "done", "pipeline": {...}}
    """
    _validate_keys()

    history = [{"role": m.role, "content": m.content} for m in request.chat_history]

    def event_generator():
        try:
            for event in rag_ask_stream(
                question=request.question,
                system_prompt=request.system_prompt,
                chat_history=history,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
