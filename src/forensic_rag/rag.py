"""The thesis RAG steps as plain functions (shared by the MCP server and the agent).

  expand_queries  -> stage 5 (sparse keyword + dense sentence queries)
  retrieve        -> stage 6 (per-query hybrid search, RRF fusion across queries)
  enrich          -> stage 6 (group by email, add neighbour chunks, join with <--chunk-->)
  analyze_email   -> stage 7 (prompt4 Cause/Effect evidence analysis)
"""

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from forensic_rag import config, store
from forensic_rag.hypotheses import resolve
from forensic_rag.models import get_llm

CHUNK_DELIMITER = "<--chunk-->"


class QueryList(BaseModel):
    queries: list[str] = Field(description="The generated retrieval queries")


class Evidence(BaseModel):
    evidence_spans: list[str] = Field(description="Exact spans quoted verbatim from the email")
    reason: str = Field(description="How the evidence relates to the hypothesis")
    strength: Literal["low", "medium", "high"]


def _prompt(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8")


def _structured(schema):
    # json_schema, not function_calling: newer Claude models (e.g. Sonnet 5.5) reject the forced
    # tool_choice that function_calling relies on
    return get_llm().with_structured_output(schema, method="json_schema")


# --------------------------------------------------------------------------- #
# Stage 5: query expansion
# --------------------------------------------------------------------------- #

def expand_queries(hypothesis_text: str, num_queries: int = config.NUM_QUERIES) -> dict[str, list[str]]:
    llm = _structured(QueryList)
    out = {}
    for kind in ("sparse", "dense"):
        prompt = _prompt(f"expansion_{kind}.txt").format(hypothesis=hypothesis_text, num_queries=num_queries)
        out[kind] = _dedupe(llm.invoke(prompt).queries)
    return out


def _dedupe(items: list[str]) -> list[str]:
    seen, result = set(), []
    for q in (s.strip() for s in items):
        if q and q.lower() not in seen:
            seen.add(q.lower())
            result.append(q)
    return result


# --------------------------------------------------------------------------- #
# Stage 6: retrieval
# --------------------------------------------------------------------------- #

def search(tenant: str, query: str, dataset: str, k: int, alpha: float) -> list[dict]:
    return store.hybrid_search(tenant, query, dataset, k=k, alpha=alpha)


def rrf_fuse(results_by_query: dict[str, list[dict]], top_k: int | None = None,
             k: int = config.RRF_CONSTANT) -> list[dict]:
    """Deduplicate chunks across queries, ordered by Reciprocal Rank Fusion: sum 1 / (k + rank)."""
    fused: dict[str, dict] = {}
    for query, hits in results_by_query.items():
        for rank, hit in enumerate(hits, 1):
            entry = fused.setdefault(hit["chunk_id"], {**hit, "rrf": 0.0, "queries": {}})
            entry["rrf"] += 1.0 / (k + rank)
            entry["queries"][query] = rank
    ranked = sorted(fused.values(), key=lambda h: h["rrf"], reverse=True)
    for h in ranked:
        h.pop("score", None)
    return ranked[:top_k]


def per_query_k(top_k: int, num_queries: int) -> int:
    """top_k is per method, split across that method's queries (thesis: 100 / 10 = 10)."""
    return max(1, top_k // max(1, num_queries))


def retrieve(tenant: str, dataset: str, sparse: list[str], dense: list[str],
             top_k: int = config.TOP_K_CHUNKS) -> list[dict]:
    """Hybrid search per query (sparse queries lean BM25, dense lean vector), deduplicated, RRF-ordered."""
    results = {}
    for q in sparse:
        results[f"sparse: {q}"] = search(tenant, q, dataset, per_query_k(top_k, len(sparse)), config.SPARSE_ALPHA)
    for q in dense:
        results[f"dense: {q}"] = search(tenant, q, dataset, per_query_k(top_k, len(dense)), config.DENSE_ALPHA)
    return rrf_fuse(results)


# --------------------------------------------------------------------------- #
# Stage 6: context enrichment
# --------------------------------------------------------------------------- #

def email_context(tenant: str, dataset: str, email_id: int, chunk_idxs: list[int],
                  window: int = config.CONTEXT_WINDOW) -> dict:
    """Hit chunks of one email plus +-window neighbours, in document order, with metadata."""
    chunks: dict[int, dict] = {}
    for idx in chunk_idxs:
        for c in store.get_neighbors(tenant, dataset, email_id, idx, window):
            chunks[c["chunk_idx"]] = c
    ordered = [chunks[i] for i in sorted(chunks)]
    meta = store.get_email(tenant, dataset, email_id) or {}
    return {
        "email_id": int(email_id),
        "subject": meta.get("subject", ""),
        "sender": meta.get("sender", ""),
        "date": meta.get("date", ""),
        "chunk_ids": [c["chunk_id"] for c in ordered],
        "text": f"\n{CHUNK_DELIMITER}\n".join(c["text"] for c in ordered),
    }


def enrich(tenant: str, dataset: str, hits: list[dict], window: int = config.CONTEXT_WINDOW) -> dict[int, dict]:
    """Group all hits by email (best email first) and build each email's context."""
    by_email: dict[int, list[int]] = defaultdict(list)
    for h in hits:
        by_email[int(h["email_id"])].append(int(h["chunk_idx"]))
    return {eid: email_context(tenant, dataset, eid, idxs, window) for eid, idxs in by_email.items()}


# --------------------------------------------------------------------------- #
# Stage 7: inference
# --------------------------------------------------------------------------- #

def _inference_prompt(hypothesis: str, email_text: str) -> str:
    # str.format turns the template's {{ }} into literal braces
    return _prompt("inference.txt").format(hypothesis=hypothesis, email=email_text)


def analyze_email(hypothesis: str, email_text: str) -> dict:
    return _structured(Evidence).invoke(_inference_prompt(hypothesis, email_text)).model_dump()


def analyze_many(hypothesis: str, contexts: dict[int, dict], max_concurrency: int = 5,
                 on_progress: Callable[[int, int], None] | None = None) -> dict[int, dict]:
    """Analyse many emails concurrently; on_progress(done, total) is called as they finish."""
    llm = _structured(Evidence)
    results: dict[int, dict] = {}

    def one(eid: int) -> None:
        try:
            results[eid] = llm.invoke(_inference_prompt(hypothesis, contexts[eid]["text"])).model_dump()
        except Exception as exc:  # keep the run going if one email fails
            results[eid] = {"evidence_spans": [], "reason": f"inference failed: {exc}", "strength": "low"}

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = [pool.submit(one, eid) for eid in contexts]
        for done, _ in enumerate(as_completed(futures), 1):
            if on_progress:
                on_progress(done, len(futures))
    return results


# --------------------------------------------------------------------------- #
# Report and the full fixed pipeline (no agentic loop)
# --------------------------------------------------------------------------- #

STRENGTH_ORDER = {"high": 0, "medium": 1, "low": 2}


def build_report(dataset: str, hypothesis: str, queries: dict, contexts: dict[int, dict],
                 analyses: dict[int, dict], **extra) -> dict:
    """Emails ranked high -> medium -> low (retrieval order breaks ties).

    `hypothesis_id` is set only when the hypothesis is one of the annotated thesis
    hypotheses, which is what makes evaluation possible.
    """
    hypothesis_id, hypothesis_text = resolve(hypothesis)
    emails = []
    for rank, (eid, ctx) in enumerate(contexts.items()):
        emails.append({
            "email_id": eid,
            "subject": ctx.get("subject", ""),
            "sender": ctx.get("sender", ""),
            "date": ctx.get("date", ""),
            "chunk_ids": ctx["chunk_ids"],
            "context": ctx["text"],
            "retrieval_rank": rank + 1,
            **analyses[eid],
        })
    emails.sort(key=lambda e: (STRENGTH_ORDER[e["strength"]], e["retrieval_rank"]))
    return {
        "dataset": dataset,
        "hypothesis": hypothesis_text,
        "hypothesis_id": hypothesis_id,
        "model": config.LLM_MODEL,
        "created": datetime.now().isoformat(timespec="seconds"),
        "queries": queries,
        **extra,
        "emails": emails,
    }


def investigate(tenant: str, dataset: str, hypothesis: str, num_queries: int = config.NUM_QUERIES,
                top_k: int = config.TOP_K_CHUNKS, window: int = config.CONTEXT_WINDOW,
                on_progress: Callable[[float, str], None] | None = None) -> dict:
    """expand -> retrieve -> enrich -> analyze. on_progress(fraction 0..1, message)."""
    def progress(fraction: float, message: str) -> None:
        if on_progress:
            on_progress(fraction, message)

    hypothesis_text = resolve(hypothesis)[1]
    progress(0.0, "Expanding the hypothesis into search queries")
    queries = expand_queries(hypothesis_text, num_queries)
    progress(0.1, "Searching the emails")
    hits = retrieve(tenant, dataset, queries["sparse"], queries["dense"], top_k)
    contexts = enrich(tenant, dataset, hits, window)
    progress(0.2, f"Analysing {len(contexts)} candidate emails")
    analyses = analyze_many(
        hypothesis_text, contexts,
        on_progress=lambda done, total: progress(0.2 + 0.8 * done / total, f"Analysed {done}/{total} emails"),
    )
    return build_report(dataset, hypothesis_text, queries, contexts, analyses)
