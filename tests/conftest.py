"""Shared fixtures. Everything external is faked: no Weaviate, Voyage or OpenRouter calls, no keys.

The environment is set before any forensic_rag import: config reads .env at import time (existing
variables win over .env), and mcp_server decides on HTTP mode and auth at import time.
"""

import os

os.environ.update({
    "MCP_TRANSPORT": "http",
    "AUTHKIT_DOMAIN": "test.authkit.app",
    "PUBLIC_URL": "http://localhost",
    "SECRET_KEY": "test-secret",
    "OPENROUTER_API_KEY": "",
    "VOYAGE_API_KEY": "",
    "WEAVIATE_URL": "",
})

import time  # noqa: E402

import jwt  # noqa: E402
import pytest  # noqa: E402

from forensic_rag import auth, config, ingest, store, web  # noqa: E402

TOKENS = {"token-alice": "alice", "token-bob": "bob"}


class FakeStore:
    """In-memory stand-in for the Weaviate functions in store.py: {tenant: {dataset: {...}}}."""

    def __init__(self):
        self.data: dict[str, dict[str, dict]] = {}

    def add(self, tenant: str, dataset: str, emails: list[dict]) -> None:
        """One chunk per email (chunk_idx 0); the chunk text is the body."""
        ds = self.data.setdefault(tenant, {})[dataset] = {"emails": {}, "chunks": []}
        for e in emails:
            eid = int(e["email_id"])
            ds["emails"][eid] = {"subject": "", "sender": "", "date": "", **e, "email_id": eid, "dataset": dataset}
            ds["chunks"].append({"email_id": eid, "dataset": dataset, "chunk_id": f"{eid}:0", "chunk_idx": 0,
                                 "text": e["body"], "subject": e.get("subject", "")})

    # the store.py API
    def create_collections(self, reset: bool = False) -> None:
        pass

    def list_datasets(self, tenant):
        return [{"dataset": d, "emails": len(v["emails"])} for d, v in sorted(self.data.get(tenant, {}).items())]

    def dataset_exists(self, tenant, dataset):
        return dataset in self.data.get(tenant, {})

    def delete_dataset(self, tenant, dataset):
        self.data.get(tenant, {}).pop(dataset, None)

    def get_email(self, tenant, dataset, email_id):
        return self.data.get(tenant, {}).get(dataset, {}).get("emails", {}).get(int(email_id))

    def hybrid_search(self, tenant, query, dataset, k=10, alpha=0.5):
        words = set(query.lower().split())
        chunks = self.data.get(tenant, {}).get(dataset, {}).get("chunks", [])
        hits = [{**c, "score": float(len(words & set(c["text"].lower().split())))} for c in chunks]
        return sorted((h for h in hits if h["score"] > 0), key=lambda h: -h["score"])[:k]

    def get_neighbors(self, tenant, dataset, email_id, chunk_idx, window=1):
        chunks = self.data.get(tenant, {}).get(dataset, {}).get("chunks", [])
        return [c for c in chunks if c["email_id"] == int(email_id) and abs(c["chunk_idx"] - chunk_idx) <= window]


@pytest.fixture
def fake_store(monkeypatch, tmp_path):
    fake = FakeStore()
    fake.add(config.PUBLIC_TENANT, "enron", [
        {"email_id": 1, "subject": "Prepay", "body": "The prepay is really a loan kept off the balance sheet."},
    ])
    for name in ("create_collections", "list_datasets", "dataset_exists", "delete_dataset",
                 "get_email", "hybrid_search", "get_neighbors"):
        monkeypatch.setattr(store, name, getattr(fake, name))

    def fake_ingest(dataset, emails, tenant=config.PUBLIC_TENANT, on_progress=None):
        fake.add(tenant, dataset, emails)
        return {"dataset": dataset, "emails": len(emails), "chunks": len(emails)}

    monkeypatch.setattr(web, "ingest_emails", fake_ingest)
    monkeypatch.setattr(ingest, "ingest_emails", fake_ingest)
    monkeypatch.setattr(web, "HOSTED_RESULTS", tmp_path / "hosted")
    return fake


@pytest.fixture
def fake_tokens(monkeypatch):
    """The real AuthKitVerifier runs; only the JWKS signature check is swapped for a lookup."""
    def decode(self, token):
        if token not in TOKENS:
            raise jwt.InvalidTokenError("unknown test token")
        return {"sub": TOKENS[token], "exp": int(time.time()) + 3600, "iss": self.issuer}

    monkeypatch.setattr(auth.AuthKitVerifier, "_decode", decode)
