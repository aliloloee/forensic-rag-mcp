"""The hosted server over real HTTP: MCP auth, signed upload/report links and tenant isolation.

Requests go through the real MCP auth middleware and AuthKitVerifier (see fake_tokens) and the
real tools and routes; Weaviate, ingestion and the LLM steps are faked (see conftest.py).
"""

import json
import time
from urllib.parse import parse_qs, urlparse

import pytest
from starlette.testclient import TestClient

from forensic_rag import rag
from forensic_rag.mcp_server import mcp

ALICE, BOB = "token-alice", "token-bob"
CSV = (b"subject,from,body\n"
       b"Clean-up,alice,Please shred the old prepay files before the auditors arrive next week.\n"
       b"Lunch,alice,Shall we get lunch on Friday after the board meeting finishes?\n")


@pytest.fixture(scope="session")
def client():
    # one app per process: the MCP session manager can only be started once
    with TestClient(mcp.streamable_http_app(), base_url="http://localhost") as c:
        yield c


@pytest.fixture(autouse=True)
def _fakes(fake_store, fake_tokens):
    pass


def call(client, token, tool, **arguments):
    """Call an MCP tool; returns (result, is_error)."""
    response = client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}},
    )
    assert response.status_code == 200, response.text
    messages = [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]
    result = next(m for m in messages if m.get("id") == 1)["result"]
    if result.get("isError"):
        return result["content"][0]["text"], True
    data = result.get("structuredContent")
    if data is None:
        data = json.loads(result["content"][0]["text"])
    return data.get("result", data) if isinstance(data, dict) else data, False


def upload(client, token, dataset="mine"):
    link = call(client, token, "create_upload_link")[0]["upload_url"]
    t = parse_qs(urlparse(link).query)["t"][0]
    response = client.post("/upload", data={"t": t, "dataset": dataset}, files={"files": ("mail.csv", CSV)})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    for _ in range(100):
        status = client.get(f"/upload/status/{job_id}").json()
        if status["status"] != "running":
            break
        time.sleep(0.02)
    assert status["status"] == "done", status
    return t


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_mcp_requires_a_valid_token(client):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    headers = {"Accept": "application/json, text/event-stream"}

    no_token = client.post("/mcp", json=body, headers=headers)
    assert no_token.status_code == 401
    assert "/.well-known/oauth-protected-resource/mcp" in no_token.headers["www-authenticate"]

    bad_token = client.post("/mcp", json=body, headers={**headers, "Authorization": "Bearer forged"})
    assert bad_token.status_code == 401

    metadata = client.get("/.well-known/oauth-protected-resource/mcp").json()
    assert metadata["resource"] == "http://localhost/mcp"
    assert any("test.authkit.app" in s for s in metadata["authorization_servers"])


def test_upload_link_is_single_use(client):
    t = upload(client, ALICE)
    again = client.post("/upload", data={"t": t, "dataset": "mine2"}, files={"files": ("mail.csv", CSV)})
    assert again.status_code == 403


def test_forged_or_bad_uploads_are_rejected(client):
    assert client.get("/upload", params={"t": "forged.link"}).status_code == 403
    assert client.post("/upload", data={"t": "forged.link", "dataset": "x1"},
                       files={"files": ("mail.csv", CSV)}).status_code == 403

    t = parse_qs(urlparse(call(client, ALICE, "create_upload_link")[0]["upload_url"]).query)["t"][0]
    for name, message in (("enron", "shared dataset"), ("Bad Name!", "Dataset name")):
        r = client.post("/upload", data={"t": t, "dataset": name}, files={"files": ("mail.csv", CSV)})
        assert r.status_code == 400 and message in r.json()["error"]


def test_users_cannot_reach_each_others_datasets(client):
    upload(client, ALICE)

    alice = call(client, ALICE, "list_datasets")[0]
    bob = call(client, BOB, "list_datasets")[0]
    assert [d["dataset"] for d in alice["your_datasets"]] == ["mine"]
    assert bob["your_datasets"] == []
    assert [d["dataset"] for d in bob["shared_datasets"]] == ["enron"]

    hits, err = call(client, ALICE, "search_chunks", query="shred prepay files", dataset="mine")
    assert not err and hits[0]["subject"] == "Clean-up"

    for tool, args in (("search_chunks", {"query": "shred", "dataset": "mine"}),
                       ("get_email", {"email_id": 1, "dataset": "mine"}),
                       ("delete_dataset", {"dataset": "mine"})):
        message, err = call(client, BOB, tool, **args)
        assert err, f"bob could call {tool} on alice's dataset"
    assert call(client, ALICE, "list_datasets")[0]["your_datasets"]           # still there

    message, err = call(client, ALICE, "delete_dataset", dataset="enron")     # shared: read-only
    assert err


def test_investigate_returns_a_signed_report_link(client, monkeypatch):
    monkeypatch.setattr(rag, "expand_queries", lambda h, n=10: {"sparse": ["shred files"], "dense": ["auditors"]})
    monkeypatch.setattr(rag, "analyze_many", lambda h, contexts, **kw: {
        eid: {"evidence_spans": ["shred the old prepay files"], "reason": "test", "strength": "high"}
        for eid in contexts})
    upload(client, ALICE)

    report, err = call(client, ALICE, "investigate", hypothesis="Staff destroyed files", dataset="mine")
    assert not err and report["emails"][0]["subject"] == "Clean-up"
    path = urlparse(report["report_url"]).path
    page = client.get(path)
    assert page.status_code == 200 and "shred the old prepay files" in page.text

    token = path.rsplit("/", 1)[1]
    assert client.get(f"/reports/{token[:-3]}abc").status_code == 403         # forged signature

    message, err = call(client, BOB, "investigate", hypothesis="Staff destroyed files", dataset="mine")
    assert err and "Unknown dataset" in message


def test_owner_can_delete(client):
    upload(client, ALICE)
    assert call(client, ALICE, "delete_dataset", dataset="mine") == ({"deleted": "mine"}, False)
    assert call(client, ALICE, "list_datasets")[0]["your_datasets"] == []
