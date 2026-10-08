"""The retrieval arithmetic from the thesis: top_k per method split across its queries, chunks
deduplicated and RRF-ordered, every retrieved chunk enriched."""

from forensic_rag import config, rag


def _hit(email_id, idx=0):
    return {"email_id": email_id, "chunk_idx": idx, "chunk_id": f"{email_id}:{idx}", "text": "", "subject": ""}


def test_ten_queries_per_method_fetch_ten_chunks_each(monkeypatch):
    calls = []
    monkeypatch.setattr(rag, "search", lambda tenant, q, ds, k, alpha: calls.append((q, k, alpha)) or [])
    rag.retrieve("public", "enron", [f"s{i}" for i in range(10)], [f"d{i}" for i in range(10)])

    assert len(calls) == 20
    assert {k for _, k, _ in calls} == {10}                       # not 100 // 20 = 5
    assert {a for q, _, a in calls if q.startswith("s")} == {config.SPARSE_ALPHA}
    assert {a for q, _, a in calls if q.startswith("d")} == {config.DENSE_ALPHA}


def test_per_query_k():
    assert rag.per_query_k(100, 10) == 10
    assert rag.per_query_k(100, 4) == 25
    assert rag.per_query_k(5, 10) == 1
    assert rag.per_query_k(100, 0) == 100


def test_rrf_dedupes_and_keeps_everything():
    results = {
        "q1": [_hit(1), _hit(2), _hit(3)],
        "q2": [_hit(2), _hit(4)],
    }
    fused = rag.rrf_fuse(results)
    assert [h["chunk_id"] for h in fused][0] == "2:0"             # found by both queries
    assert sorted(h["chunk_id"] for h in fused) == ["1:0", "2:0", "3:0", "4:0"]
    many = {f"q{i}": [_hit(i * 100 + j) for j in range(10)] for i in range(20)}
    assert len(rag.rrf_fuse(many)) == 200                         # never truncated


def test_enrich_covers_every_retrieved_email(monkeypatch):
    seen = {}
    monkeypatch.setattr(rag, "email_context",
                        lambda t, ds, eid, idxs, window: seen.setdefault(eid, idxs) and {"email_id": eid})
    hits = [_hit(e, i) for e in range(60) for i in (0, 3)]
    contexts = rag.enrich("public", "enron", hits)
    assert len(contexts) == 60                                    # no top-50 chunk cut
    assert seen[7] == [0, 3]
