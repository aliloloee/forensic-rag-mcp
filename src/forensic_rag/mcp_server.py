"""MCP server exposing the forensic email RAG as tools, resources and a prompt.

Two ways to run it:
  Local (stdio, no login; you are the "local" user):
      python -m forensic_rag.mcp_server
      mcp dev src/forensic_rag/mcp_server.py                      # MCP Inspector
      claude mcp add forensic-rag -- <venv python> -m forensic_rag.mcp_server
  Hosted (Streamable HTTP at <PUBLIC_URL>/mcp, plus the upload page and report pages):
      python -m forensic_rag.mcp_server --http
    With AUTHKIT_DOMAIN set, every MCP request needs a WorkOS AuthKit access token, and each
    signed-in user gets a private Weaviate tenant. Without it the HTTP server is open (dev only).

Never print() to stdout here: with the stdio transport stdout carries the MCP protocol.
"""

import functools
import logging
import os
import sys

import anyio
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import Context, FastMCP

from forensic_rag import auth, config, rag, store, web
from forensic_rag.hypotheses import HYPOTHESES, get_hypothesis, resolve
from forensic_rag.limits import OneAtATime, SlidingWindow

HTTP_MODE = "--http" in sys.argv or os.getenv("MCP_TRANSPORT") == "http"
auth.AUTH_ENABLED = HTTP_MODE and bool(config.AUTHKIT_DOMAIN)

_auth_kwargs = {}
if auth.AUTH_ENABLED:
    _resource = f"{config.PUBLIC_URL}/mcp"
    _auth_kwargs = {
        "token_verifier": auth.AuthKitVerifier(config.AUTHKIT_DOMAIN, _resource),
        "auth": AuthSettings(
            issuer_url=f"https://{config.AUTHKIT_DOMAIN}",
            resource_server_url=_resource,
            validate_token_resource=False,     # AuthKitVerifier checks the audience itself
        ),
    }

mcp = FastMCP(
    "forensic-rag",
    instructions=(
        "Hypothesis-driven forensic analysis over email datasets. A dataset is a set of emails "
        "(list_datasets): the shared 'enron' dataset, or the user's own uploads (create_upload_link). "
        "A hypothesis is any free-text claim of possible misconduct, or one of the example ids H1-H3. "
        "Typical flow: investigate for a one-shot run, or expand_queries -> search_chunks -> "
        "analyze_email for a step-by-step one. When presenting results, group emails by strength and, "
        "for each, give the explanation and quote the evidence spans verbatim; share the report_url."
    ),
    log_level="WARNING",
    host=config.HOST,
    port=config.PORT,
    stateless_http=True,
    json_response=False,               # SSE responses, so investigate can stream progress
    **_auth_kwargs,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
if HTTP_MODE:
    web.register_routes(mcp)


async def _in_thread(fn, *args, **kwargs):
    """Run blocking work (Weaviate, Voyage, LLM calls) in a worker thread so the server can
    handle several tool calls concurrently, e.g. the agent's parallel analyze_email calls."""
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


def _resolve_tenant(user_tenant: str, dataset: str) -> str:
    """The tenant holding `dataset`: the caller's own first, then the shared public tenant."""
    if store.dataset_exists(user_tenant, dataset):
        return user_tenant
    if store.dataset_exists(config.PUBLIC_TENANT, dataset):
        return config.PUBLIC_TENANT
    raise ValueError(f"Unknown dataset '{dataset}'. Call list_datasets to see the available ones.")


async def _tenant(dataset: str) -> str:
    # identity is read here, in the request context, never from tool arguments
    return await _in_thread(_resolve_tenant, auth.current_tenant(), dataset)


# Per-user limits on the tools that spend LLM credit. They apply only to signed-in (hosted) users
# and are keyed on the caller's own tenant, never the dataset's (everyone shares "public").
INVESTIGATIONS = SlidingWindow(config.INVESTIGATIONS_PER_DAY, 24 * 3600, "investigations")
INVESTIGATION_SLOTS = OneAtATime("an investigation")
LLM_TOOL_CALLS = SlidingWindow(config.LLM_TOOL_CALLS_PER_HOUR, 3600, "analysis calls")


def _charge_llm_call() -> None:
    if auth.AUTH_ENABLED:
        LLM_TOOL_CALLS.hit(auth.current_tenant())


def _chunk_idxs(chunk_ids: list[str]) -> list[int]:
    return [int(str(c).split(":")[-1]) for c in chunk_ids]


def _text(hypothesis: str) -> str:
    return resolve(hypothesis)[1]


# --------------------------------------------------------------------------- #
# Tools: datasets
# --------------------------------------------------------------------------- #

@mcp.tool()
async def list_datasets() -> dict:
    """List the email datasets you can investigate: your own uploads and the shared ones,
    plus uploads that are still being indexed."""
    tenant = auth.current_tenant()
    own = await _in_thread(store.list_datasets, tenant) if tenant != config.PUBLIC_TENANT else []
    shared = await _in_thread(store.list_datasets, config.PUBLIC_TENANT)
    return {
        "your_datasets": own,
        "shared_datasets": shared,
        "indexing": [{"dataset": j["dataset"], "stage": j["stage"], "progress": j["progress"]}
                     for j in web.jobs_for(tenant)],
    }


@mcp.tool()
async def create_upload_link() -> dict:
    """Create a private, single-use link where you can upload your own emails (.csv, .mbox, .eml,
    .txt, or a .zip of them) as a new dataset. Give the link to the user to open in a browser."""
    if not HTTP_MODE:
        return {"error": "Uploads need the hosted server (python -m forensic_rag.mcp_server --http). "
                         "Locally, use forensic_rag.ingest instead."}
    return {
        "upload_url": web.make_upload_link(auth.current_tenant()),
        "expires_in_minutes": config.UPLOAD_LINK_MINUTES,
        "limits": {"emails": config.MAX_UPLOAD_EMAILS, "megabytes": config.MAX_UPLOAD_MB},
        "next_step": "After indexing finishes, call list_datasets and investigate a hypothesis on the new dataset.",
    }


@mcp.tool()
async def delete_dataset(dataset: str) -> dict:
    """Permanently delete one of your own datasets (shared datasets cannot be deleted)."""
    tenant = auth.current_tenant()
    if not await _in_thread(store.dataset_exists, tenant, dataset):
        raise ValueError(f"You have no dataset named '{dataset}'")
    await _in_thread(store.delete_dataset, tenant, dataset)
    return {"deleted": dataset}


@mcp.tool()
def list_example_hypotheses() -> list[dict]:
    """List the three annotated thesis hypotheses (H1-H3) for the Enron data. Any other
    free-text hypothesis can also be investigated."""
    return [{"id": hid, "text": h["text"]} for hid, h in HYPOTHESES.items()]


# --------------------------------------------------------------------------- #
# Tools: the RAG steps
# --------------------------------------------------------------------------- #

@mcp.tool()
async def expand_queries(hypothesis: str, num_queries: int = config.NUM_QUERIES) -> dict:
    """Generate sparse (keyword, for BM25) and dense (sentence, for semantic search) retrieval
    queries for a hypothesis (free text, or H1/H2/H3). Returns {"sparse": [...], "dense": [...]}."""
    _charge_llm_call()
    return await _in_thread(rag.expand_queries, _text(hypothesis), num_queries)


@mcp.tool()
async def search_chunks(query: str, dataset: str = config.DEFAULT_DATASET, k: int = 10,
                        alpha: float = 0.5) -> list[dict]:
    """Hybrid search over the email chunks of one dataset.

    alpha controls the BM25/vector mix: 0 = keyword only, 1 = semantic only.
    Use ~0.25 for keyword queries and ~0.75 for sentence queries.
    Returns chunks with email_id, chunk_id, chunk_idx, subject, text and score.
    """
    tenant = await _tenant(dataset)
    return await _in_thread(rag.search, tenant, query, dataset, min(k, 50), alpha)


@mcp.tool()
async def get_email_context(email_id: int, chunk_ids: list[str], dataset: str = config.DEFAULT_DATASET,
                            window: int = config.CONTEXT_WINDOW) -> dict:
    """Return the given chunks of an email plus `window` neighbouring chunks on each side,
    in document order and joined with <--chunk-->, together with subject, sender and date."""
    tenant = await _tenant(dataset)
    return await _in_thread(rag.email_context, tenant, dataset, email_id, _chunk_idxs(chunk_ids), window)


@mcp.tool()
async def get_email(email_id: int, dataset: str = config.DEFAULT_DATASET) -> dict:
    """Return the full email (metadata + body)."""
    tenant = await _tenant(dataset)
    email = await _in_thread(store.get_email, tenant, dataset, email_id)
    if email is None:
        raise ValueError(f"Email {email_id} not found in dataset {dataset}")
    return email


@mcp.tool()
async def analyze_email(hypothesis: str, email_id: int, chunk_ids: list[str] | None = None,
                        dataset: str = config.DEFAULT_DATASET, window: int = config.CONTEXT_WINDOW) -> dict:
    """Assess how strongly an email supports the hypothesis (Cause/Effect analysis).

    With chunk_ids, only those chunks (+ neighbours) are analysed; otherwise the whole email.
    Returns {"evidence_spans": [...], "reason": str, "strength": "low"|"medium"|"high"}.
    """
    _charge_llm_call()
    if chunk_ids:
        text = (await get_email_context(email_id, chunk_ids, dataset, window))["text"]
    else:
        text = (await get_email(email_id, dataset))["body"]
    return {"email_id": email_id, **await _in_thread(rag.analyze_email, _text(hypothesis), text)}


@mcp.tool()
async def investigate(hypothesis: str, ctx: Context, dataset: str = config.DEFAULT_DATASET,
                      num_queries: int = config.NUM_QUERIES, top_k: int = config.TOP_K_CHUNKS) -> dict:
    """Run the full pipeline (expand -> retrieve -> enrich -> analyze) for a hypothesis (free text,
    or H1/H2/H3) over a dataset. Returns the high- and medium-relevance emails with evidence spans
    and explanations, and (on the hosted server) a link to a full HTML report. Takes 1-3 minutes.
    On the hosted server each user can run one at a time and a limited number per day."""
    tenant = await _tenant(dataset)            # an unknown dataset fails here, before any limit is charged
    if not auth.AUTH_ENABLED:
        return await _investigate(tenant, hypothesis, ctx, dataset, num_queries, top_k)
    user = auth.current_tenant()
    with INVESTIGATION_SLOTS.slot(user):
        INVESTIGATIONS.hit(user)
        return await _investigate(tenant, hypothesis, ctx, dataset, num_queries, top_k)


async def _investigate(tenant: str, hypothesis: str, ctx: Context, dataset: str, num_queries: int,
                       top_k: int) -> dict:
    def on_progress(fraction: float, message: str) -> None:      # called from the worker thread
        try:
            anyio.from_thread.run(functools.partial(ctx.report_progress, fraction * 100, 100, message))
        except Exception:
            pass                                                   # progress is best-effort

    report = await _in_thread(rag.investigate, tenant, dataset, hypothesis, num_queries, min(top_k, 150),
                              on_progress=on_progress)
    report_url = await _in_thread(web.save_report, tenant, report) if HTTP_MODE else None

    keep = [{k: v for k, v in e.items() if k != "context"}            # keep the response small
            for e in report["emails"] if e["strength"] != "low"]
    return {**{k: v for k, v in report.items() if k != "emails"},
            "emails": keep,
            "low_relevance_count": len(report["emails"]) - len(keep),
            "report_url": report_url}


# --------------------------------------------------------------------------- #
# Resources and prompts
# --------------------------------------------------------------------------- #

@mcp.resource("hypothesis://{hypothesis_id}")
def hypothesis_resource(hypothesis_id: str) -> str:
    """The text of one example thesis hypothesis (H1-H3)."""
    return get_hypothesis(hypothesis_id)["text"]


@mcp.resource("prompt://inference")
def inference_prompt_resource() -> str:
    """The evidence-analysis prompt template used by analyze_email."""
    return (config.PROMPTS_DIR / "inference.txt").read_text(encoding="utf-8")


@mcp.prompt()
def forensic_investigation(hypothesis: str, dataset: str = config.DEFAULT_DATASET) -> str:
    """Guide an assistant through a tool-driven investigation of one hypothesis."""
    return (
        f"Investigate this forensic hypothesis over the email dataset '{dataset}':\n\"{_text(hypothesis)}\"\n\n"
        "1. Split the hypothesis into Cause and Effect.\n"
        "2. Call expand_queries, then search_chunks for each query (alpha 0.25 for sparse, 0.75 for dense).\n"
        "3. If results only cover the Cause or only the Effect, write extra queries for the missing part.\n"
        "4. For the most promising emails call analyze_email with the hit chunk_ids.\n"
        "5. Report emails grouped by strength (high, medium): subject, explanation, and the evidence "
        "spans quoted verbatim."
    )


if __name__ == "__main__":
    if HTTP_MODE:
        from forensic_rag.ingest import log
        store.create_collections()
        log(f"forensic-rag on {config.PUBLIC_URL}/mcp "
            f"(auth: {'WorkOS AuthKit ' + config.AUTHKIT_DOMAIN if auth.AUTH_ENABLED else 'OFF - dev only'})")
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
