import os
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechLab AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Glassmorphism CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Animated gradient background ─────────────────────── */
    .stApp {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ── Glass card base ──────────────────────────────────── */
    .glass {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 1rem;
    }

    /* ── Hero header ──────────────────────────────────────── */
    .hero-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .hero-header h1 {
        background: linear-gradient(135deg, #a78bfa, #818cf8, #6366f1, #8b5cf6);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
        animation: shimmer 3s linear infinite;
    }
    @keyframes shimmer {
        to { background-position: 200% center; }
    }
    .hero-header .subtitle {
        color: rgba(203, 213, 225, 0.7);
        font-size: 0.95rem;
        font-weight: 300;
        letter-spacing: 0.02em;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: #c4b5fd;
        padding: 0.3rem 0.9rem;
        border-radius: 2rem;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    /* ── Chat messages ────────────────────────────────────── */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 1rem !important;
        padding: 1rem 1.2rem !important;
        margin-bottom: 0.75rem !important;
        transition: all 0.3s ease;
    }
    .stChatMessage:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(139, 92, 246, 0.15) !important;
    }
    .stChatMessage p, .stChatMessage li, .stChatMessage span {
        color: #e2e8f0 !important;
    }
    .stChatMessage h1, .stChatMessage h2, .stChatMessage h3 {
        color: #f1f5f9 !important;
    }
    .stChatMessage strong {
        color: #c4b5fd !important;
    }
    .stChatMessage code {
        background: rgba(139, 92, 246, 0.15) !important;
        color: #c4b5fd !important;
        border-radius: 0.3rem;
        padding: 0.1rem 0.4rem;
    }

    /* ── Pipeline stage cards ─────────────────────────────── */
    .stage-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.6rem;
    }
    .stage-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.6rem;
        height: 1.6rem;
        border-radius: 50%;
        font-size: 0.75rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .stage-1 .stage-number { background: rgba(99, 102, 241, 0.25); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); }
    .stage-2 .stage-number { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .stage-3 .stage-number { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .stage-4 .stage-number { background: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.4); }

    .stage-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .stage-1 .stage-title { color: #818cf8; }
    .stage-2 .stage-title { color: #fbbf24; }
    .stage-3 .stage-title { color: #34d399; }
    .stage-4 .stage-title { color: #f472b6; }

    /* ── Chunk cards ──────────────────────────────────────── */
    .chunk-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 0.7rem;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
        line-height: 1.55;
        color: #cbd5e1;
        transition: all 0.2s ease;
    }
    .chunk-card:hover {
        background: rgba(255, 255, 255, 0.07);
        transform: translateY(-1px);
    }
    .chunk-card .chunk-source {
        font-weight: 600;
        font-size: 0.75rem;
        margin-bottom: 0.3rem;
    }
    .chunk-card.stage-2-card .chunk-source { color: #fbbf24; }
    .chunk-card.stage-3-card .chunk-source { color: #34d399; }
    .chunk-card.stage-4-card .chunk-source { color: #f472b6; }

    .chunk-card.stage-2-card { border-left: 3px solid rgba(245, 158, 11, 0.5); }
    .chunk-card.stage-3-card { border-left: 3px solid rgba(16, 185, 129, 0.5); }
    .chunk-card.stage-4-card { border-left: 3px solid rgba(236, 72, 153, 0.5); }

    .score-badge {
        display: inline-block;
        padding: 0.1rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.68rem;
        font-weight: 700;
        margin-left: 0.4rem;
    }
    .score-rerank {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* ── Query variation pills ────────────────────────────── */
    .query-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        padding: 0.35rem 0.8rem;
        border-radius: 2rem;
        font-size: 0.78rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        transition: all 0.2s ease;
    }
    .query-pill:hover {
        background: rgba(99, 102, 241, 0.2);
        transform: scale(1.02);
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] * {
        color: #c4b5fd !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e0e7ff !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(139, 92, 246, 0.12) !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
        color: #c4b5fd !important;
        border-radius: 0.6rem;
        font-weight: 500;
        transition: all 0.25s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(139, 92, 246, 0.25) !important;
        border-color: rgba(139, 92, 246, 0.4) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.15);
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 0.5rem !important;
        color: #e0e7ff !important;
    }
    section[data-testid="stSidebar"] input:focus,
    section[data-testid="stSidebar"] textarea:focus {
        border-color: rgba(139, 92, 246, 0.4) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1) !important;
    }

    /* ── Chat input ──────────────────────────────────────── */
    .stChatInput {
        background: transparent !important;
    }
    .stChatInput > div {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 1rem !important;
        transition: all 0.3s ease;
    }
    .stChatInput > div:focus-within {
        border-color: rgba(139, 92, 246, 0.4) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1), 0 8px 30px rgba(0,0,0,0.3) !important;
    }
    .stChatInput textarea {
        color: #e2e8f0 !important;
    }

    /* ── Expanders ────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 0.6rem !important;
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-top: none !important;
        border-radius: 0 0 0.6rem 0.6rem !important;
    }

    /* ── Dividers ─────────────────────────────────────────── */
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin: 1rem 0;
    }

    /* ── Hide default streamlit elements ──────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Spinner ──────────────────────────────────────────── */
    .stSpinner > div > div {
        border-top-color: #8b5cf6 !important;
    }

    /* ── Pipeline flow connector ─────────────────────────── */
    .pipeline-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.3rem;
        padding: 0.8rem 0;
        flex-wrap: wrap;
    }
    .flow-step {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0.3rem 0.7rem;
        border-radius: 2rem;
        font-size: 0.72rem;
        font-weight: 500;
        color: #94a3b8;
    }
    .flow-arrow {
        color: rgba(139, 92, 246, 0.4);
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

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
- If the answer is NOT in the context → Do NOT guess. Say you don't have the information.\
"""

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧪 TechLab AI")
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # API Keys
    st.markdown("#### 🔐 API Keys")
    gemini_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Free from aistudio.google.com/apikey",
    )
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key

    openai_key = st.text_input(
        "OpenAI API Key (fallback)",
        type="password",
        placeholder="sk-...",
        help="Used if Gemini fails",
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    neon_url = st.text_input(
        "Neon Database URL",
        type="password",
        placeholder="postgresql://...",
        help="Overrides .env",
    )
    if neon_url:
        os.environ["NEON_DATABASE_URL"] = neon_url

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # System Prompt
    st.markdown("#### 📝 System Prompt")
    system_prompt = st.text_area(
        "System prompt for the LLM:",
        value=st.session_state.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        height=200,
        label_visibility="collapsed",
    )
    st.session_state["system_prompt"] = system_prompt

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Debug toggle
    show_pipeline = st.toggle("🔍 Show pipeline stages", value=True,
                              help="Reveal every stage of the RAG pipeline")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pipeline_data = []
        st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("#### 📖 Quick Start")
    st.markdown("""
    1. Add **PDF/TXT** → `docs/`
    2. `uv run python ingest.py`
    3. Chat away! 💬
    """)

    st.markdown(
        '<p style="text-align:center; font-size:0.65rem; opacity:0.35; margin-top:2rem;">'
        'LangChain · pgvector · FlashRank · Gemini / OpenAI</p>',
        unsafe_allow_html=True,
    )


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>✦ TechLab AI Assistant</h1>
    <p class="subtitle">Multi-query retrieval · Reranking · Contextual compression</p>
    <div class="hero-badge">pgvector + FlashRank + Gemini / OpenAI</div>
</div>
""", unsafe_allow_html=True)

# Pipeline flow visualization
st.markdown("""
<div class="pipeline-flow">
    <span class="flow-step">📝 Query</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">🔀 Multi-Query</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">🔍 Retrieve</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">🎯 Rerank</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">✂️ Compress</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">🤖 Answer</span>
</div>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_data" not in st.session_state:
    st.session_state.pipeline_data = []


# ── Render Helpers ───────────────────────────────────────────────────────────

def render_pipeline_stages(data):
    """Render the full pipeline visualization for a single response."""
    if not data:
        return

    # Stage 1: Multi-Query
    with st.expander("🔀  Stage 1 — Multi-Query Generation", expanded=False):
        st.markdown(
            '<div class="stage-header stage-1">'
            '<span class="stage-number">1</span>'
            '<span class="stage-title">Generated Query Variations</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<span class="query-pill">🎯 {data["original_query"]}</span>', unsafe_allow_html=True)
        for q in data.get("generated_queries", []):
            st.markdown(f'<span class="query-pill">🔀 {q}</span>', unsafe_allow_html=True)

    # Stage 2: Retrieved chunks
    with st.expander(
        f"🔍  Stage 2 — Retrieved Chunks ({len(data.get('chunks_retrieved', []))} unique / {data.get('total_before_dedup', 0)} total)",
        expanded=False,
    ):
        st.markdown(
            '<div class="stage-header stage-2">'
            '<span class="stage-number">2</span>'
            '<span class="stage-title">Retrieval + Deduplication</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        for i, ch in enumerate(data.get("chunks_retrieved", []), 1):
            st.markdown(
                f'<div class="chunk-card stage-2-card">'
                f'<div class="chunk-source">#{i} · {ch["source"]}</div>'
                f'{ch["content"][:220]}{"…" if len(ch["content"]) > 220 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Stage 3: Reranked
    with st.expander(
        f"🎯  Stage 3 — After Reranking (top {len(data.get('chunks_after_rerank', []))})",
        expanded=False,
    ):
        st.markdown(
            '<div class="stage-header stage-3">'
            '<span class="stage-number">3</span>'
            '<span class="stage-title">FlashRank Cross-Encoder Reranking</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        for i, ch in enumerate(data.get("chunks_after_rerank", []), 1):
            score = ch.get("rerank_score", 0)
            st.markdown(
                f'<div class="chunk-card stage-3-card">'
                f'<div class="chunk-source">#{i} · {ch["source"]}'
                f'<span class="score-badge score-rerank">{score}</span></div>'
                f'{ch["content"][:220]}{"…" if len(ch["content"]) > 220 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Stage 4: Compressed
    with st.expander(
        f"✂️  Stage 4 — Contextual Compression ({len(data.get('chunks_compressed', []))} chunks)",
        expanded=False,
    ):
        st.markdown(
            '<div class="stage-header stage-4">'
            '<span class="stage-number">4</span>'
            '<span class="stage-title">Extracted Relevant Sentences</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        for i, ch in enumerate(data.get("chunks_compressed", []), 1):
            compressed = ch.get("compressed_content") or ch.get("content", "")
            st.markdown(
                f'<div class="chunk-card stage-4-card">'
                f'<div class="chunk-source">#{i} · {ch["source"]}'
                f'<span class="score-badge score-rerank">{ch.get("rerank_score", "")}</span></div>'
                f'{compressed[:300]}{"…" if len(compressed) > 300 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Display Chat History ─────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and show_pipeline:
            detail_idx = sum(1 for m in st.session_state.messages[:idx] if m["role"] == "assistant")
            if detail_idx < len(st.session_state.pipeline_data):
                render_pipeline_stages(st.session_state.pipeline_data[detail_idx])


# ── Chat Input ───────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

STAGE_LABELS = {
    "multi_query": "🔀 Generating query variations…",
    "multi_query_done": "🔀 Query variations ready ✓",
    "retrieval": "🔍 Searching knowledge base…",
    "retrieval_done": "🔍 Documents retrieved ✓",
    "reranking": "🎯 Reranking by relevance…",
    "reranking_done": "🎯 Reranking complete ✓",
    "compressing": "✂️ Extracting key information…",
    "compressing_done": "✂️ Compression complete ✓",
    "generating": "🧠 Writing answer…",
    "cache_hit": "⚡ Found in cache!",
}

if prompt := st.chat_input("Ask anything about your documents …"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Stream the response from FastAPI
    try:
        import requests

        with st.chat_message("assistant", avatar="🤖"):
            # Build chat history
            chat_history = []
            for m in st.session_state.messages[-10:]:
                if m["role"] in ("user", "assistant"):
                    chat_history.append({
                        "role": m["role"],
                        "content": m["content"][:500],
                    })

            # Status widget for pipeline stages
            status = st.status("🤖 TechLab AI is thinking…", expanded=True)
            answer_container = st.empty()
            full_answer = ""
            pipeline_data = None

            # Stream from the API
            response = requests.post(
                f"{API_URL}/api/chat/stream",
                json={
                    "question": prompt,
                    "system_prompt": st.session_state.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                    "chat_history": chat_history[:-1],
                },
                stream=True,
                timeout=120,
            )

            if response.status_code != 200:
                raise Exception(response.text)

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                try:
                    event = json.loads(line[6:])  # strip "data: "
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "stage":
                    stage = event.get("stage", "")
                    label = STAGE_LABELS.get(stage, stage)
                    status.update(label=label)
                    if stage.endswith("_done"):
                        data = event.get("data", {})
                        if "count" in data:
                            status.write(f"{label} ({data['count']} items)")
                        elif "queries" in data:
                            status.write(f"{label} ({len(data['queries'])} queries)")
                        else:
                            status.write(label)
                    elif stage == "cache_hit":
                        status.write("⚡ Found in cache — instant response!")

                elif event_type == "token":
                    token = event.get("content", "")
                    full_answer += token
                    answer_container.markdown(full_answer + "▌")

                elif event_type == "done":
                    pipeline_data = event.get("pipeline")

                elif event_type == "error":
                    raise Exception(event.get("detail", "Unknown error"))

            # Finalize
            answer_container.markdown(full_answer)
            status.update(label="✅ Complete", state="complete", expanded=False)

            # Store results
            st.session_state.pipeline_data.append(pipeline_data)
            if show_pipeline and pipeline_data:
                render_pipeline_stages(pipeline_data)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
            })

    except requests.exceptions.ConnectionError:
        error_msg = "❌ **API server not running.** Start it with: `uv run uvicorn api:app --reload --port 8000`"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.session_state.pipeline_data.append(None)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(error_msg)

    except Exception as e:
        error_msg = f"❌ **Error:** {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.session_state.pipeline_data.append(None)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(error_msg)


