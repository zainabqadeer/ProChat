"""
ingest.py — Document Ingestion Pipeline (NeonDB + pgvector)
=============================================================
Loads documents from the `docs/` folder, splits them into chunks,
generates embeddings, and stores them in NeonDB via pgvector.

Tries Gemini embeddings first, falls back to OpenAI if Gemini fails.

Usage:
    python ingest.py
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "rag_documents"


def get_connection_string():
    """Get NeonDB connection string from environment."""
    conn_str = os.getenv("NEON_DATABASE_URL")
    if not conn_str:
        print("❌  Error: NEON_DATABASE_URL not found.")
        print("   Copy .env.example to .env and add your Neon connection string.")
        sys.exit(1)
    # Use psycopg v3 driver instead of psycopg2
    if conn_str.startswith("postgresql://"):
        conn_str = conn_str.replace("postgresql://", "postgresql+psycopg://", 1)
    return conn_str


def get_embeddings():
    """Try Gemini embeddings first, fall back to OpenAI."""
    # Try Gemini first
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-002")
            # Test with a small call
            embeddings.embed_query("test")
            print("   ✅ Using Gemini embeddings")
            return embeddings
        except Exception as e:
            print(f"   ⚠️  Gemini embeddings failed: {e}")
            print("   Falling back to OpenAI...")

    # Fall back to OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            embeddings.embed_query("test")
            print("   ✅ Using OpenAI embeddings")
            return embeddings
        except Exception as e:
            print(f"   ❌ OpenAI embeddings also failed: {e}")

    print("❌  No working embedding provider found.")
    print("   Add GOOGLE_API_KEY or OPENAI_API_KEY to your .env file.")
    sys.exit(1)


# ── Helper Functions ─────────────────────────────────────────────────────────

def load_documents(docs_dir: str):
    """Load PDF and TXT documents from the given directory."""
    documents = []

    # Load PDFs
    pdf_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    documents.extend(pdf_loader.load())

    # Load plain text files
    txt_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents.extend(txt_loader.load())

    return documents


def split_documents(documents):
    """Split documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return text_splitter.split_documents(documents)


def create_vectorstore(chunks):
    """Create/update pgvector store in NeonDB from document chunks."""
    embeddings = get_embeddings()
    connection_str = get_connection_string()

    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=connection_str,
        pre_delete_collection=True,
    )
    return vectorstore


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Verify at least one API key
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌  Error: No API key found.")
        print("   Add GOOGLE_API_KEY or OPENAI_API_KEY to your .env file.")
        sys.exit(1)

    # Verify Neon connection
    get_connection_string()

    # Verify docs directory
    if not os.path.isdir(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)
        print(f"📁  Created empty docs/ folder at: {DOCS_DIR}")
        print("   Add your PDF or TXT files there, then re-run this script.")
        sys.exit(0)

    # 1. Load
    print("📄  Loading documents …")
    documents = load_documents(DOCS_DIR)
    if not documents:
        print("⚠️  No documents found in docs/. Add PDF or TXT files and retry.")
        sys.exit(0)
    print(f"   Loaded {len(documents)} document page(s).")

    # 2. Split
    print("✂️  Splitting into chunks …")
    chunks = split_documents(documents)
    print(f"   Created {len(chunks)} chunks.")

    # 3. Embed & store
    print("🧠  Generating embeddings & storing in NeonDB …")
    create_vectorstore(chunks)
    print(f"✅  Embeddings stored in NeonDB  →  collection: '{COLLECTION_NAME}'")
    print("   Inspect in Neon SQL Editor:")
    print("   SELECT id, document, embedding FROM langchain_pg_embedding LIMIT 5;")


if __name__ == "__main__":
    main()
