"""LangGraph agentic RAG that talks to the forensic-rag MCP server as an MCP client.

    expand -> retrieve -> grade --(sufficient / max rounds)--> enrich -> analyze (fan-out) -> report
                ^            |
                +- rewrite <-+ (missing Cause/Effect aspects -> new queries)

Every data step (query expansion, search, context, inference) is an MCP tool call to
forensic_rag.mcp_server, launched as a stdio subprocess. The grading/rewriting decisions are
made here, on the client side, by the LLM.

Usage:
  python -m forensic_rag.agent --hypothesis H3 --open
  python -m forensic_rag.agent --dataset enron --hypothesis "Employees discussed hiding trading losses from auditors."
      -> results/<dataset>_<H|custom>_<ts>.json and .html (the evidence report)
  python -m forensic_rag.agent --chat "Which emails talk about shredding documents?"
      -> free-form ReAct agent over the MCP tools
"""

import argparse
import asyncio
import json
import operator
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from mcp import ClientSession
from pydantic import BaseModel, Field

from forensic_rag import config, rag
from forensic_rag.hypotheses import resolve
from forensic_rag.report import write_html

MCP_SERVERS = {
    "forensic": {
        "command": sys.executable,
        "args": ["-m", "forensic_rag.mcp_server"],
        "transport": "stdio",
    }
}
GRADE_SNIPPETS = 15
ANALYZE_CONCURRENCY = 5


# --------------------------------------------------------------------------- #
# MCP helper
# --------------------------------------------------------------------------- #

async def call_tool(session: ClientSession, name: str, **args):
    """Call an MCP tool and return its result as Python data."""
    result = await session.call_tool(name, args)
    texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
    if result.isError:
        raise RuntimeError(f"MCP tool {name} failed: {' '.join(texts)}")
    data = result.structuredContent
    if data is not None:
        # FastMCP wraps non-object return values (e.g. lists) as {"result": ...}
        return data["result"] if set(data) == {"result"} else data
    parsed = [json.loads(t) for t in texts]
    return parsed[0] if len(parsed) == 1 else parsed


# --------------------------------------------------------------------------- #
# State and LLM schemas
# --------------------------------------------------------------------------- #

class State(TypedDict, total=False):
    dataset: str
    hypothesis: str                                       # full hypothesis text
    queries: dict                                         # {"sparse": [...], "dense": [...]}, all rounds
    pending: list[tuple[str, str]]                        # (kind, query) still to search
    results_by_query: Annotated[dict, operator.or_]
    hits: list[dict]
    round: int
    grades: Annotated[list[dict], operator.add]
    new_queries: dict
    contexts: dict
    analyses: Annotated[dict, operator.or_]
    report: dict
    html_path: str


class AnalyzeTask(TypedDict):
    dataset: str
    hypothesis: str
    email_id: int
    chunk_ids: list[str]


class Grade(BaseModel):
    cause: str = Field(description="The Cause part of the hypothesis")
    effect: str = Field(description="The Effect part of the hypothesis")
    sufficient: bool = Field(description="True if the snippets cover both the Cause and the Effect well")
    missing_aspects: str = Field(description="What the retrieved snippets fail to cover; empty if sufficient")
    new_sparse: list[str] = Field(default_factory=list, description="Up to 4 new keyword queries (<=6 words)")
    new_dense: list[str] = Field(default_factory=list, description="Up to 4 new neutral, third-person sentences")


GRADE_PROMPT = """You are grading retrieval results for a forensic email investigation.

Hypothesis:
{hypothesis}

Queries already used:
{queries}

Top retrieved email snippets:
{snippets}

Split the hypothesis into Cause and Effect. Decide whether the snippets contain material
covering BOTH the Cause and the Effect. If not, describe the missing aspects and propose new
queries that target them. New queries must not repeat the used ones and must not add
mechanisms, motives or actors that are not in the hypothesis."""


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

def build_graph(session: ClientSession):
    semaphore = asyncio.Semaphore(ANALYZE_CONCURRENCY)

    async def expand(state: State) -> dict:
        q = await call_tool(session, "expand_queries", hypothesis=state["hypothesis"],
                            num_queries=config.NUM_QUERIES)
        print(f"[expand] {len(q['sparse'])} sparse + {len(q['dense'])} dense queries")
        pending = [("sparse", s) for s in q["sparse"]] + [("dense", d) for d in q["dense"]]
        return {"queries": q, "pending": pending, "round": 0}

    async def retrieve(state: State) -> dict:
        # same depth as the first round for rewrite queries too (thesis: 100 / 10 = 10)
        per_query = rag.per_query_k(config.TOP_K_CHUNKS, config.NUM_QUERIES)

        async def one(kind: str, query: str):
            alpha = config.SPARSE_ALPHA if kind == "sparse" else config.DENSE_ALPHA
            hits = await call_tool(session, "search_chunks", query=query,
                                   dataset=state["dataset"], k=per_query, alpha=alpha)
            return f"{kind}: {query}", hits

        new = dict(await asyncio.gather(*(one(k, q) for k, q in state["pending"])))
        all_results = {**state.get("results_by_query", {}), **new}
        hits = rag.rrf_fuse(all_results)
        print(f"[retrieve] round {state['round']}: {len(new)} queries -> {len(hits)} fused chunks, "
              f"{len({h['email_id'] for h in hits})} emails")
        return {"results_by_query": new, "hits": hits, "pending": []}

    async def grade(state: State) -> dict:
        snippets = "\n\n".join(
            f"[email {h['email_id']} | {h['subject']}]\n{h['text'][:400]}" for h in state["hits"][:GRADE_SNIPPETS]
        )
        used = "\n".join(f"- {q}" for q in state["queries"]["sparse"] + state["queries"]["dense"])
        g: Grade = await rag._structured(Grade).ainvoke(
            GRADE_PROMPT.format(hypothesis=state["hypothesis"], queries=used, snippets=snippets)
        )
        print(f"[grade] sufficient={g.sufficient}" + ("" if g.sufficient else f" missing: {g.missing_aspects}"))
        return {"grades": [g.model_dump()], "new_queries": {"sparse": g.new_sparse[:4], "dense": g.new_dense[:4]}}

    def route_after_grade(state: State) -> str:
        last = state["grades"][-1]
        has_new = state["new_queries"]["sparse"] or state["new_queries"]["dense"]
        if last["sufficient"] or not has_new or state["round"] >= config.MAX_REWRITE_ROUNDS:
            return "enrich"
        return "rewrite"

    async def rewrite(state: State) -> dict:
        used = {q.lower() for q in state["queries"]["sparse"] + state["queries"]["dense"]}
        new = {k: [q for q in v if q.lower() not in used] for k, v in state["new_queries"].items()}
        print(f"[rewrite] +{len(new['sparse'])} sparse, +{len(new['dense'])} dense queries")
        return {
            "queries": {k: state["queries"][k] + new[k] for k in ("sparse", "dense")},
            "pending": [("sparse", q) for q in new["sparse"]] + [("dense", q) for q in new["dense"]],
            "round": state["round"] + 1,
        }

    async def enrich(state: State) -> dict:
        hit_ids: dict[int, list[str]] = {}
        for h in state["hits"]:
            hit_ids.setdefault(int(h["email_id"]), []).append(h["chunk_id"])

        async def one(eid: int, chunk_ids: list[str]):
            ctx = await call_tool(session, "get_email_context", dataset=state["dataset"],
                                  email_id=eid, chunk_ids=chunk_ids, window=config.CONTEXT_WINDOW)
            return eid, {**ctx, "hit_chunk_ids": chunk_ids}

        contexts = dict(await asyncio.gather(*(one(e, c) for e, c in hit_ids.items())))
        print(f"[enrich] {len(contexts)} candidate emails")
        return {"contexts": contexts}

    def fan_out(state: State) -> list[Send]:
        return [
            Send("analyze", AnalyzeTask(dataset=state["dataset"], hypothesis=state["hypothesis"], email_id=eid,
                                        chunk_ids=ctx["hit_chunk_ids"]))
            for eid, ctx in state["contexts"].items()
        ]

    async def analyze(task: AnalyzeTask) -> dict:
        async with semaphore:
            try:
                out = await call_tool(session, "analyze_email", hypothesis=task["hypothesis"], dataset=task["dataset"],
                                      email_id=task["email_id"], chunk_ids=task["chunk_ids"],
                                      window=config.CONTEXT_WINDOW)
                out.pop("email_id", None)
            except Exception as exc:  # keep the run going if one email fails
                out = {"evidence_spans": [], "reason": f"inference failed: {exc}", "strength": "low"}
        print(f"[analyze] email {task['email_id']}: {out['strength']}")
        return {"analyses": {task["email_id"]: out}}

    async def report(state: State) -> dict:
        rep = rag.build_report(state["dataset"], state["hypothesis"], state["queries"], state["contexts"],
                               state["analyses"], rewrite_rounds=state["round"], grades=state["grades"])
        config.RESULTS_DIR.mkdir(exist_ok=True)
        name = f"{state['dataset']}_{rep['hypothesis_id'] or 'custom'}_{datetime.now():%Y%m%d_%H%M%S}"
        path = config.RESULTS_DIR / f"{name}.json"
        path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        html_path = write_html(rep, path)
        counts = {s: sum(e["strength"] == s for e in rep["emails"]) for s in ("high", "medium", "low")}
        print(f"[report] {counts} -> {html_path}")
        return {"report": rep, "html_path": str(html_path)}

    g = StateGraph(State)
    g.add_node("expand", expand)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("enrich", enrich)
    g.add_node("analyze", analyze)
    g.add_node("report", report)

    g.add_edge(START, "expand")
    g.add_edge("expand", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, ["rewrite", "enrich"])
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("enrich", fan_out, ["analyze"])
    g.add_edge("analyze", "report")
    g.add_edge("report", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #

async def run_investigation(dataset: str, hypothesis: str) -> dict:
    client = MultiServerMCPClient(MCP_SERVERS)
    async with client.session("forensic") as session:      # one long-lived MCP server process
        graph = build_graph(session)
        final = await graph.ainvoke(
            {"dataset": dataset, "hypothesis": resolve(hypothesis)[1]},
            {"recursion_limit": 100},
        )
    return final


async def run_chat(question: str, dataset: str) -> str:
    """Free-form ReAct agent: the LLM decides which MCP tools to call (via langchain-mcp-adapters)."""
    from langchain.agents import create_agent
    from forensic_rag.models import get_llm

    client = MultiServerMCPClient(MCP_SERVERS)
    async with client.session("forensic") as session:
        tools = await load_mcp_tools(session)
        agent = create_agent(
            get_llm(),
            tools,
            system_prompt=(
                "You are a forensic email analyst. Use the tools to find and assess evidence in the "
                f"dataset '{dataset}'. Quote evidence verbatim, explain it, and cite email ids."
            ),
        )
        result = await agent.ainvoke({"messages": [("user", question)]}, {"recursion_limit": 50})
    return result["messages"][-1].content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=config.DEFAULT_DATASET, help="dataset to search (default: enron)")
    parser.add_argument("--hypothesis", help="free-text hypothesis, or H1/H2/H3")
    parser.add_argument("--chat", help="ask a free-form question instead of running the investigation graph")
    parser.add_argument("--open", action="store_true", help="open the HTML report when done")
    args = parser.parse_args()

    if args.chat:
        print(asyncio.run(run_chat(args.chat, args.dataset)))
    elif args.hypothesis:
        final = asyncio.run(run_investigation(args.dataset, args.hypothesis))
        if args.open:
            webbrowser.open(Path(final["html_path"]).resolve().as_uri())
    else:
        parser.error("give --hypothesis or --chat")


if __name__ == "__main__":
    main()
