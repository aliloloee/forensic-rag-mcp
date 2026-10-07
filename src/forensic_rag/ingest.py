"""Ingestion: emails -> LangChain semantic chunking -> Voyage embeddings -> Weaviate.

`ingest_emails(dataset, emails)` works for any list of emails. The CLI loads the bundled Enron
sample, data/enron_emails.jsonl: 306 emails from the thesis (TREC 2010 Legal Track topics
201/204/205 = H1/H2/H3), already cleaned by the thesis pipeline (stages 1-2).

The Enron data goes to the shared "public" tenant, so every user can search it.

Usage:
  python -m forensic_rag.ingest --reset                              # all 306 emails -> dataset "enron"
  python -m forensic_rag.ingest --source H3 --dataset enron-h3       # one thesis topic as its own dataset
"""

import argparse
import json
import sys
from collections.abc import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from forensic_rag import config, store
from forensic_rag.hypotheses import HYPOTHESES
from forensic_rag.models import get_embeddings

def log(msg: str) -> None:
    # stderr, so this is safe to call from the MCP server (stdout carries the protocol there)
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Enron loader
# --------------------------------------------------------------------------- #

def load_enron(source: str) -> list[dict]:
    """Emails of one thesis topic (H1/H2/H3) as
    [{email_id, subject, sender, recipients, date, docID, body}, ...]."""
    emails = []
    with open(config.ENRON_DATA, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.pop("topic") == source:
                emails.append(record)
    return emails


# --------------------------------------------------------------------------- #
# Generic pipeline
# --------------------------------------------------------------------------- #

def chunk_emails(emails: list[dict], on_progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """Semantic chunking (embedding-based breakpoints) + a size cap safety pass."""
    # imported here: langchain-experimental warns on import, and the MCP server rarely needs it
    from langchain_experimental.text_splitter import SemanticChunker

    semantic = SemanticChunker(get_embeddings(), breakpoint_threshold_type="percentile")
    capper = RecursiveCharacterTextSplitter(chunk_size=config.MAX_CHUNK_CHARS, chunk_overlap=100)

    chunks = []
    for i, email in enumerate(emails, 1):
        pieces = []
        for piece in semantic.split_text(email["body"]):
            pieces.extend(capper.split_text(piece) if len(piece) > config.MAX_CHUNK_CHARS else [piece])

        idx = 0
        for text in pieces:
            text = text.strip()
            if len(text) < config.MIN_CHUNK_CHARS:
                continue
            idx += 1
            chunks.append({
                "email_id": email["email_id"],
                "dataset": email["dataset"],
                "chunk_id": f"{email['email_id']}:{idx}",
                "chunk_idx": idx,
                "text": text,
                "subject": email["subject"],
            })
        if on_progress:
            on_progress(i, len(emails))
        if i % 20 == 0:
            log(f"  chunked {i}/{len(emails)} emails")
    return chunks


def ingest_emails(dataset: str, emails: list[dict], tenant: str = config.PUBLIC_TENANT,
                  on_progress: Callable[[str, float], None] | None = None) -> dict:
    """Chunk, embed and store emails as `dataset` in `tenant` (replacing any previous version).

    Each email needs email_id (int) and body; subject/sender/recipients/date/docID are optional.
    on_progress(stage, fraction 0..1) reports progress, e.g. to the upload page.
    """
    def progress(stage: str, fraction: float) -> None:
        if on_progress:
            on_progress(stage, fraction)

    ids = [e["email_id"] for e in emails]
    if len(ids) != len(set(ids)):
        raise ValueError("email_id values must be unique within a dataset")
    emails = [
        {"subject": "", "sender": "", "recipients": "", "date": "", "docID": "", **e,
         "email_id": int(e["email_id"]), "dataset": dataset}
        for e in emails if e.get("body", "").strip()
    ]

    store.create_collections()
    log(f"[{dataset}] {len(emails)} emails -> semantic chunking")
    chunks = chunk_emails(emails, lambda i, n: progress("chunking", 0.7 * i / n))
    log(f"[{dataset}] {len(chunks)} chunks -> embedding with {config.EMBED_MODEL}")
    progress("embedding", 0.7)
    vectors = get_embeddings().embed_documents([c["text"] for c in chunks])

    progress("storing", 0.9)
    store.delete_dataset(tenant, dataset)
    store.insert_emails(tenant, emails)
    store.insert_chunks(tenant, chunks, vectors)
    stats = {"dataset": dataset,
             "emails": store.count("email", tenant, dataset),
             "chunks": store.count("chunk", tenant, dataset)}
    progress("done", 1.0)
    log(f"[{dataset}] stored: {stats['emails']} emails, {stats['chunks']} chunks")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", nargs="+", default=list(HYPOTHESES),
                        help="thesis topics to load: H1 H2 H3 (default: all)")
    parser.add_argument("--dataset", default=config.DEFAULT_DATASET, help="dataset name (default: enron)")
    parser.add_argument("--tenant", default=config.PUBLIC_TENANT,
                        help="tenant to write to (default: public, readable by every user)")
    parser.add_argument("--reset", action="store_true",
                        help="drop and recreate the collections first (deletes ALL datasets)")
    args = parser.parse_args()

    store.create_collections(reset=args.reset)
    emails = []
    for source in args.source:
        emails.extend(load_enron(source.upper()))
    ingest_emails(args.dataset, emails, tenant=args.tenant)


if __name__ == "__main__":
    main()
