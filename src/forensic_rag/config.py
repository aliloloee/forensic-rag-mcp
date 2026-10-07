"""Settings loaded from forensic-rag-mcp/.env (see .env.example)."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]          # .../forensic-rag-mcp
PROMPTS_DIR = PROJECT_DIR / "prompts"
RESULTS_DIR = PROJECT_DIR / "results"
ENRON_DATA = PROJECT_DIR / "data" / "enron_emails.jsonl"

load_dotenv(PROJECT_DIR / ".env")

# ---- LLM (OpenRouter, OpenAI-compatible API) ----
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-5.5")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ---- Embeddings (Voyage AI) ----
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "voyage-3.5")

# ---- Weaviate ----
# Local Docker by default; set WEAVIATE_URL + WEAVIATE_API_KEY to use Weaviate Cloud instead.
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
# One collection holds both full emails (kind="email", no vector) and their searchable chunks
# (kind="chunk", with a vector), so it fits Weaviate Cloud's one-collection free tier.
COLLECTION = "ForensicEmails"
LEGACY_COLLECTIONS = ("EmailChunk", "Email")     # earlier two-collection layout, removed on --reset
VECTOR_NAME = "dense_vector"

# Multi-tenancy: every user's data lives in their own Weaviate tenant.
PUBLIC_TENANT = "public"          # shared, read-only datasets (the Enron emails)
LOCAL_TENANT = "local"            # the user when running without authentication
DEFAULT_DATASET = "enron"

# ---- Hosted (HTTP) mode ----
# PUBLIC_URL: where the server is reachable, e.g. https://forensic-rag.onrender.com
# AUTHKIT_DOMAIN: e.g. your-app.authkit.app. When set, every MCP request needs a WorkOS token.
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000").rstrip("/")
AUTHKIT_DOMAIN = os.getenv("AUTHKIT_DOMAIN", "").removeprefix("https://").rstrip("/")
SECRET_KEY = os.getenv("SECRET_KEY", "")          # signs upload links and report links
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ---- Upload limits (keep LLM/embedding costs bounded) ----
MAX_UPLOAD_EMAILS = int(os.getenv("MAX_UPLOAD_EMAILS", "500"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_DATASETS_PER_USER = int(os.getenv("MAX_DATASETS_PER_USER", "5"))
MAX_EMAIL_CHARS = 20_000
UPLOAD_LINK_MINUTES = 30
REPORT_LINK_DAYS = 7

# ---- Pipeline defaults (mirroring the thesis notebooks) ----
NUM_QUERIES = 10
TOP_K_CHUNKS = 100     # per method (sparse / dense), split across its queries: 100 / 10 = 10 each
RRF_CONSTANT = 60      # RRF only orders the deduplicated chunks; every one of them is enriched
CONTEXT_WINDOW = 1
SPARSE_ALPHA = 0.5     # sparse/keyword queries: BM25 and vectors equally (best recall/precision on H1-H3)
DENSE_ALPHA = 0.75     # dense/sentence queries lean on vectors
MAX_REWRITE_ROUNDS = 2

# ---- Chunking ----
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 30
