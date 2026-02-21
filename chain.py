"""
chain.py — Advanced RAG Pipeline
===================================
Pipeline stages:
  1. Multi-Query Generation   — LLM creates 3 query variations for better recall
  2. Retrieval (pgvector)     — Semantic search across all query variations
  3. Deduplication             — Remove duplicate chunks from multi-query results
  4. Reranking (FlashRank)    — Cross-encoder rescoring for precision
  5. Contextual Compression   — Extract only relevant sentences from top chunks
  6. Generation (LLM)         — Final answer grounded in compressed context

Tries Gemini first, falls back to OpenAI if Gemini fails.
Everything is exposed via the PipelineResult dataclass for UI transparency.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
COLLECTION_NAME = "rag_documents"
LLM_TEMPERATURE = 0.3
RETRIEVER_K = 4            # per-query retrieval count
RERANK_TOP_N = 4            # chunks kept after reranking
NUM_QUERY_VARIATIONS = 2    # multi-query count

# ── Default System Prompt ────────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = """\
You are a professional, polite, and knowledgeable AI assistant. \
Your role is to answer user questions **strictly based on the provided context documents**.

## Core Rules — NEVER violate these:

1. **ONLY use the provided context.** Do NOT use your training data or prior knowledge. \
If the context does not contain enough information to answer, say: \
"I don't have enough information in the provided documents to answer this accurately."
2. **NEVER fabricate, guess, or hallucinate information.** If you are unsure, say so honestly.
3. **Always cite your sources.** Reference the document name when stating facts \
(e.g., "According to [document_name]…").

## Response Guidelines:

4. **Be professional and polite.** Use a warm, helpful tone. Address the user respectfully.
5. **Be concise but thorough.** Provide complete answers without unnecessary filler.
6. **Structure your responses well.** Use markdown: headers, bullet points, bold text \
for key terms, and code blocks where appropriate.
7. **Acknowledge limitations.** If a question is partially answerable, answer what you can \
and clearly state what information is missing.
8. **Stay on topic.** If a question is unrelated to the documents, politely redirect: \
"This question falls outside the scope of the documents I have access to."

## Confidence Calibration:

- If the answer is clearly supported by context → Answer confidently with citations.
- If the answer is partially supported → Answer what you can, flag uncertainty.
- If the answer is NOT in the context → Do NOT guess. Say you don't have the information.
"""


# ── Provider Helper — Gemini first, OpenAI fallback ──────────────────────────

_provider_cache = {}

def get_llm():
    """Get chat LLM — tries Gemini first, falls back to OpenAI."""
    if "llm" in _provider_cache:
        return _provider_cache["llm"]

    # Try Gemini first
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=LLM_TEMPERATURE)
            llm.invoke("hi")  # quick test
            _provider_cache["llm"] = llm
            _provider_cache["provider"] = "Gemini"
            return llm
        except Exception:
            pass

    # Fall back to OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=LLM_TEMPERATURE)
            _provider_cache["llm"] = llm
            _provider_cache["provider"] = "OpenAI"
            return llm
        except Exception:
            pass

    raise ValueError("No working LLM provider. Add GOOGLE_API_KEY or OPENAI_API_KEY to .env")


def get_embeddings():
    """Get embeddings — tries Gemini first, falls back to OpenAI."""
    if "embeddings" in _provider_cache:
        return _provider_cache["embeddings"]

    # Try Gemini first
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            emb = GoogleGenerativeAIEmbeddings(model="models/embedding-002")
            emb.embed_query("test")
            _provider_cache["embeddings"] = emb
            return emb
        except Exception:
            pass

    # Fall back to OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings
            emb = OpenAIEmbeddings(model="text-embedding-3-small")
            _provider_cache["embeddings"] = emb
            return emb
        except Exception:
            pass

    raise ValueError("No working embedding provider. Add GOOGLE_API_KEY or OPENAI_API_KEY to .env")


def get_provider_name():
    """Return the name of the active provider."""
    return _provider_cache.get("provider", "Unknown")


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ChunkInfo:
    """A single chunk with metadata."""
    content: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)
    rerank_score: float | None = None
    compressed_content: str | None = None

@dataclass
class PipelineResult:
    """Full pipeline output with transparency into every stage."""
    # Stage 1: Multi-query
    original_query: str = ""
    generated_queries: list[str] = field(default_factory=list)

    # Stage 2 & 3: Retrieval + dedup
    chunks_retrieved: list[ChunkInfo] = field(default_factory=list)
    total_before_dedup: int = 0

    # Stage 4: Reranking
    chunks_after_rerank: list[ChunkInfo] = field(default_factory=list)

    # Stage 5: Compression
    chunks_compressed: list[ChunkInfo] = field(default_factory=list)

    # Stage 6: Answer
    answer: str = ""
    provider: str = ""


# ── Multi-Query Generator ───────────────────────────────────────────────────

def generate_query_variations(question: str, llm, n: int = NUM_QUERY_VARIATIONS) -> list[str]:
    """
    Use the LLM to generate N variations of the user's question.
    This improves recall by searching from multiple angles.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that generates alternative search queries. "
         "Given a user question, generate {n} different versions of that question "
         "that would help retrieve relevant documents. Each variation should approach "
         "the topic from a slightly different angle or use different keywords.\n"
         "Return ONLY the queries, one per line, numbered 1-{n}. No explanations."),
        ("human", "{question}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"question": question, "n": n})

    # Parse numbered lines
    queries = []
    for line in response.content.strip().split("\n"):
        line = line.strip()
        if line:
            # Remove numbering like "1. " or "1) "
            cleaned = line.lstrip("0123456789.)- ").strip()
            if cleaned:
                queries.append(cleaned)

    return queries[:n]


# ── Reranker ─────────────────────────────────────────────────────────────────

_ranker = None

def get_ranker():
    """Lazy-load the FlashRank cross-encoder model."""
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir="./flashrank_cache")
    return _ranker


def rerank_documents(query: str, chunks: list[ChunkInfo], top_n: int = RERANK_TOP_N) -> list[ChunkInfo]:
    """Rerank chunks using FlashRank cross-encoder. Falls back to top-N if FlashRank unavailable."""
    if not FLASHRANK_AVAILABLE:
        # Fallback: just keep top-N chunks in original order
        return [
            ChunkInfo(
                content=ch.content,
                source=ch.source,
                metadata=ch.metadata,
                rerank_score=round(1.0 - i * 0.1, 4),
            )
            for i, ch in enumerate(chunks[:top_n])
        ]

    ranker = get_ranker()

    passages = [
        {"id": i, "text": ch.content, "meta": ch.metadata}
        for i, ch in enumerate(chunks)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    reranked = []
    for r in results[:top_n]:
        idx = r["id"]
        chunk = ChunkInfo(
            content=chunks[idx].content,
            source=chunks[idx].source,
            metadata=chunks[idx].metadata,
            rerank_score=round(r["score"], 4),
        )
        reranked.append(chunk)

    return reranked


# ── Contextual Compression ──────────────────────────────────────────────────

def compress_chunks(question: str, chunks: list[ChunkInfo], llm) -> list[ChunkInfo]:
    """
    Extract only the sentences relevant to the question from ALL chunks
    in a single batched LLM call for speed.
    """
    # Build a single prompt with all chunks numbered
    chunks_text = ""
    for i, ch in enumerate(chunks):
        chunks_text += f"\n--- CHUNK {i+1} (Source: {ch.source}) ---\n{ch.content}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precision extraction assistant. Given a user question and "
         "multiple document chunks, extract ONLY the sentences that are directly "
         "relevant to answering the question from EACH chunk.\n\n"
         "Format your response as:\n"
         "CHUNK 1: [extracted sentences or NO RELEVANT CONTENT]\n"
         "CHUNK 2: [extracted sentences or NO RELEVANT CONTENT]\n"
         "...and so on. Do not add commentary."),
        ("human",
         "Question: {question}\n\nDocument chunks:{chunks}"),
    ])
    chain = prompt | llm

    try:
        response = chain.invoke({"question": question, "chunks": chunks_text})
        lines = response.content.strip()

        # Parse batch response
        compressed = []
        current_chunk_idx = -1
        current_text = []

        for line in lines.split("\n"):
            line = line.strip()
            # Check if this starts a new chunk section
            chunk_match = False
            for i in range(len(chunks)):
                if line.upper().startswith(f"CHUNK {i+1}"):
                    # Save previous chunk
                    if current_chunk_idx >= 0 and current_text:
                        text = " ".join(current_text)
                        if "NO RELEVANT CONTENT" not in text.upper():
                            compressed.append(ChunkInfo(
                                content=chunks[current_chunk_idx].content,
                                source=chunks[current_chunk_idx].source,
                                metadata=chunks[current_chunk_idx].metadata,
                                rerank_score=chunks[current_chunk_idx].rerank_score,
                                compressed_content=text,
                            ))
                    current_chunk_idx = i
                    # Extract text after "CHUNK N:" prefix
                    after_colon = line.split(":", 1)[1].strip() if ":" in line else ""
                    current_text = [after_colon] if after_colon else []
                    chunk_match = True
                    break
            if not chunk_match and line:
                current_text.append(line)

        # Don't forget the last chunk
        if current_chunk_idx >= 0 and current_text:
            text = " ".join(current_text)
            if "NO RELEVANT CONTENT" not in text.upper():
                compressed.append(ChunkInfo(
                    content=chunks[current_chunk_idx].content,
                    source=chunks[current_chunk_idx].source,
                    metadata=chunks[current_chunk_idx].metadata,
                    rerank_score=chunks[current_chunk_idx].rerank_score,
                    compressed_content=text,
                ))

        return compressed if compressed else chunks

    except Exception:
        return chunks  # fallback to uncompressed


# ── Vector Store ─────────────────────────────────────────────────────────────

_vectorstore = None

def get_vectorstore():
    """Lazy-load the pgvector store from NeonDB."""
    global _vectorstore
    if _vectorstore is None:
        conn_str = os.getenv("NEON_DATABASE_URL")
        if not conn_str:
            raise ValueError("NEON_DATABASE_URL not set. Check your .env file.")
        # Use psycopg v3 driver instead of psycopg2
        if conn_str.startswith("postgresql://"):
            conn_str = conn_str.replace("postgresql://", "postgresql+psycopg://", 1)

        embeddings = get_embeddings()
        _vectorstore = PGVector(
            collection_name=COLLECTION_NAME,
            connection=conn_str,
            embeddings=embeddings,
        )
    return _vectorstore


# ── Smart Router — skip pipeline for casual messages ─────────────────────────

CASUAL_PATTERNS = {
    "hi", "hey", "hello", "hii", "hiii", "yo", "sup", "hola",
    "thanks", "thank you", "thankyou", "thx", "ty",
    "bye", "goodbye", "see you", "cya",
    "ok", "okay", "sure", "cool", "nice", "great",
    "good morning", "good evening", "good night", "good afternoon",
    "how are you", "whats up", "what's up", "wassup",
}

CASUAL_RESPONSES = {
    "greeting": "Hello! 👋 I'm the **TechLab** assistant. Ask me anything about TechLab and I'll find the answer for you from our knowledge base!",
    "thanks": "You're welcome! 😊 Feel free to ask more questions about TechLab anytime.",
    "bye": "Goodbye! 👋 Come back anytime you need help with TechLab info.",
    "general": "I'm here to help you with questions about **TechLab**. Go ahead and ask something specific! 📄",
}


def is_casual_message(question: str) -> str | None:
    """Detect casual/greeting messages. Returns response category or None."""
    q = question.strip().lower().rstrip("!?.,:;")
    
    if q in CASUAL_PATTERNS or len(q) <= 3:
        if any(w in q for w in ("hi", "hey", "hello", "hii", "yo", "sup", "hola")):
            return "greeting"
        elif any(w in q for w in ("thank", "thx", "ty")):
            return "thanks"
        elif any(w in q for w in ("bye", "goodbye", "cya", "see you")):
            return "bye"
        return "general"
    
    # Also catch short phrases
    if q in ("how are you", "whats up", "what's up", "wassup"):
        return "greeting"
    
    return None


# ── Main Advanced RAG Function ───────────────────────────────────────────────

def ask(question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chat_history: list | None = None) -> PipelineResult:
    """
    Advanced RAG pipeline with smart routing:
      - Casual messages (hi, hey, thanks) → instant response, no pipeline
      - Document questions → full pipeline

    Returns a PipelineResult with full stage-by-stage transparency.
    """
    result = PipelineResult(original_query=question)

    llm = get_llm()
    result.provider = get_provider_name()

    # ── Smart Route: skip pipeline for casual messages ───────────────────
    casual_type = is_casual_message(question)
    if casual_type:
        result.answer = CASUAL_RESPONSES.get(casual_type, CASUAL_RESPONSES["general"])
        return result

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    # ── Stage 1: Multi-Query Generation ──────────────────────────────────
    query_variations = generate_query_variations(question, llm)
    result.generated_queries = query_variations
    all_queries = [question] + query_variations  # original + variations

    # ── Stage 2: Retrieval across all queries ────────────────────────────
    all_docs = []
    seen_contents = set()
    for q in all_queries:
        docs = retriever.invoke(q)
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_docs.append(doc)

    result.total_before_dedup = sum(RETRIEVER_K for _ in all_queries)
    result.chunks_retrieved = [
        ChunkInfo(
            content=doc.page_content,
            source=os.path.basename(doc.metadata.get("source", "Unknown")),
            metadata=doc.metadata,
        )
        for doc in all_docs
    ]

    # ── Stage 3: Reranking ───────────────────────────────────────────────
    reranked = rerank_documents(question, result.chunks_retrieved, top_n=RERANK_TOP_N)
    result.chunks_after_rerank = reranked

    # ── Stage 4: Contextual Compression ──────────────────────────────────
    compressed = compress_chunks(question, reranked, llm)
    result.chunks_compressed = compressed

    # ── Stage 5: Generation ──────────────────────────────────────────────
    # Build context from compressed chunks
    context_parts = []
    for ch in compressed:
        text = ch.compressed_content if ch.compressed_content else ch.content
        context_parts.append(f"[Source: {ch.source}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        ("system", system_prompt + "\n\nContext:\n{context}"),
    ]

    # Add chat history
    if chat_history:
        for entry in chat_history:
            messages.append((entry["role"], entry["content"]))

    messages.append(("human", "{question}"))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    result.answer = response.content

    # Cache the result
    _cache_result(question, result)

    return result


# ── LRU Cache ────────────────────────────────────────────────────────────────

import time
from collections import OrderedDict

_query_cache: OrderedDict[str, tuple[PipelineResult, float]] = OrderedDict()
CACHE_MAX_SIZE = 50
CACHE_TTL = 300  # 5 minutes


def _cache_key(question: str) -> str:
    """Normalize question for cache lookup."""
    return question.strip().lower()


def _get_cached(question: str) -> PipelineResult | None:
    """Check cache for a recent result."""
    key = _cache_key(question)
    if key in _query_cache:
        result, timestamp = _query_cache[key]
        if time.time() - timestamp < CACHE_TTL:
            _query_cache.move_to_end(key)
            return result
        else:
            del _query_cache[key]
    return None


def _cache_result(question: str, result: PipelineResult):
    """Store a result in the cache."""
    key = _cache_key(question)
    _query_cache[key] = (result, time.time())
    while len(_query_cache) > CACHE_MAX_SIZE:
        _query_cache.popitem(last=False)


# ── Streaming Pipeline ──────────────────────────────────────────────────────

def ask_stream(question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
               chat_history: list | None = None):
    """
    Streaming version of ask(). Yields events as dicts:
      {"type": "stage", "stage": "...", "data": {...}}
      {"type": "token", "content": "..."}
      {"type": "done", "pipeline": {...}}

    This allows the frontend to show real-time progress.
    """
    result = PipelineResult(original_query=question)

    llm = get_llm()
    result.provider = get_provider_name()

    # ── Smart Route: casual messages ─────────────────────────────────────
    casual_type = is_casual_message(question)
    if casual_type:
        answer = CASUAL_RESPONSES.get(casual_type, CASUAL_RESPONSES["general"])
        # Stream casual response word by word
        for word in answer.split(" "):
            yield {"type": "token", "content": word + " "}
        result.answer = answer
        yield {"type": "done", "pipeline": _result_to_dict(result)}
        return

    # ── Check cache ──────────────────────────────────────────────────────
    cached = _get_cached(question)
    if cached:
        yield {"type": "stage", "stage": "cache_hit", "data": {"message": "Found in cache!"}}
        for word in cached.answer.split(" "):
            yield {"type": "token", "content": word + " "}
        yield {"type": "done", "pipeline": _result_to_dict(cached)}
        return

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    # ── Stage 1: Multi-Query ─────────────────────────────────────────────
    yield {"type": "stage", "stage": "multi_query", "data": {"message": "Generating query variations..."}}
    query_variations = generate_query_variations(question, llm)
    result.generated_queries = query_variations
    all_queries = [question] + query_variations
    yield {"type": "stage", "stage": "multi_query_done", "data": {"queries": all_queries}}

    # ── Stage 2: Retrieval ───────────────────────────────────────────────
    yield {"type": "stage", "stage": "retrieval", "data": {"message": f"Searching {len(all_queries)} queries..."}}
    all_docs = []
    seen_contents = set()
    for q in all_queries:
        docs = retriever.invoke(q)
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_docs.append(doc)

    result.total_before_dedup = sum(RETRIEVER_K for _ in all_queries)
    result.chunks_retrieved = [
        ChunkInfo(
            content=doc.page_content,
            source=os.path.basename(doc.metadata.get("source", "Unknown")),
            metadata=doc.metadata,
        )
        for doc in all_docs
    ]
    yield {"type": "stage", "stage": "retrieval_done", "data": {"count": len(all_docs)}}

    # ── Stage 3: Reranking ───────────────────────────────────────────────
    yield {"type": "stage", "stage": "reranking", "data": {"message": "Reranking chunks..."}}
    reranked = rerank_documents(question, result.chunks_retrieved, top_n=RERANK_TOP_N)
    result.chunks_after_rerank = reranked
    yield {"type": "stage", "stage": "reranking_done", "data": {"count": len(reranked)}}

    # ── Stage 4: Compression ─────────────────────────────────────────────
    yield {"type": "stage", "stage": "compressing", "data": {"message": "Extracting relevant info..."}}
    compressed = compress_chunks(question, reranked, llm)
    result.chunks_compressed = compressed
    yield {"type": "stage", "stage": "compressing_done", "data": {"count": len(compressed)}}

    # ── Stage 5: Streaming Generation ────────────────────────────────────
    yield {"type": "stage", "stage": "generating", "data": {"message": "Generating answer..."}}

    context_parts = []
    for ch in compressed:
        text = ch.compressed_content if ch.compressed_content else ch.content
        context_parts.append(f"[Source: {ch.source}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        ("system", system_prompt + "\n\nContext:\n{context}"),
    ]
    if chat_history:
        for entry in chat_history:
            messages.append((entry["role"], entry["content"]))
    messages.append(("human", "{question}"))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm

    # Stream the answer token by token
    full_answer = ""
    for chunk in chain.stream({"context": context, "question": question}):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full_answer += token
            yield {"type": "token", "content": token}

    result.answer = full_answer
    _cache_result(question, result)

    yield {"type": "done", "pipeline": _result_to_dict(result)}


def _result_to_dict(result: PipelineResult) -> dict:
    """Convert PipelineResult to a JSON-serializable dict."""
    return {
        "answer": result.answer,
        "provider": result.provider,
        "original_query": result.original_query,
        "generated_queries": result.generated_queries,
        "total_before_dedup": result.total_before_dedup,
        "chunks_retrieved": [
            {"content": c.content, "source": c.source}
            for c in result.chunks_retrieved
        ],
        "chunks_after_rerank": [
            {"content": c.content, "source": c.source, "rerank_score": c.rerank_score}
            for c in result.chunks_after_rerank
        ],
        "chunks_compressed": [
            {"content": c.content, "source": c.source, "rerank_score": c.rerank_score,
             "compressed_content": c.compressed_content}
            for c in result.chunks_compressed
        ],
    }

