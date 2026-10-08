import time

from forensic_rag import auth


def test_signed_link_round_trip():
    token = auth.sign({"k": "report", "t": "u_alice", "r": "abc"}, ttl_seconds=60)
    payload = auth.unsign(token)
    assert payload["k"] == "report" and payload["t"] == "u_alice" and payload["r"] == "abc"


def test_tampered_link_is_rejected():
    body, mac = auth.sign({"k": "report", "t": "u_alice"}, ttl_seconds=60).split(".")
    # someone edits the payload to point at another tenant but cannot re-sign it
    forged_body = auth._b64(auth._unb64(body).replace(b"u_alice", b"u_bob__"))
    assert auth.unsign(f"{forged_body}.{mac}") is None
    assert auth.unsign(f"{body}.{mac[:-2]}xx") is None


def test_expired_link_is_rejected(monkeypatch):
    token = auth.sign({"k": "upload", "t": "u_alice"}, ttl_seconds=60)
    monkeypatch.setattr(time, "time", lambda: 10**12)
    assert auth.unsign(token) is None


def test_garbage_link_is_rejected():
    for token in ("", "no-dot", "a.b.c", "!!!.???"):
        assert auth.unsign(token) is None


def test_tenant_names_are_safe_for_weaviate():
    tenant = auth.tenant_for_subject("user_01ABC|google-oauth2:12345/" + "x" * 100)
    assert tenant.startswith("u_user_01ABC_google-oauth2_12345_")
    assert len(tenant) <= 64
    assert all(c.isalnum() or c in "_-" for c in tenant)
