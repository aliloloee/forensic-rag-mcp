"""Who is calling, and signed links.

* AuthKitVerifier: validates the OAuth access tokens that Claude obtains from WorkOS AuthKit
  (RS256 JWTs: signature via AuthKit's JWKS, issuer, audience = this MCP server's URL).
* current_tenant(): the Weaviate tenant of the signed-in user. Tools never take a user id as an
  argument; identity comes only from the verified token.
* sign()/unsign(): short-lived HMAC-signed links, used for the upload page and report pages so
  a browser can reach them without a second login.
"""

import base64
import hashlib
import hmac
import json
import re
import secrets
import time

import anyio
import jwt
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken

from forensic_rag import config

# Set by mcp_server when it runs over HTTP with AuthKit configured.
AUTH_ENABLED = False


class AuthKitVerifier:
    """mcp TokenVerifier for WorkOS AuthKit access tokens."""

    def __init__(self, authkit_domain: str, resource_url: str):
        self.issuer = f"https://{authkit_domain}"
        self.resource_url = resource_url
        self.jwks = jwt.PyJWKClient(f"{self.issuer}/oauth2/jwks", cache_keys=True)

    def _decode(self, token: str) -> dict:
        key = self.jwks.get_signing_key_from_jwt(token)
        return jwt.decode(token, key.key, algorithms=["RS256"], issuer=self.issuer,
                          audience=self.resource_url, options={"require": ["exp", "sub"]})

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            claims = await anyio.to_thread.run_sync(self._decode, token)   # JWKS fetch is blocking
        except jwt.PyJWTError:
            return None
        return AccessToken(
            token=token,
            client_id=str(claims.get("client_id") or claims.get("azp") or "unknown"),
            scopes=str(claims.get("scope", "")).split(),
            expires_at=claims.get("exp"),
            resource=self.resource_url,
            subject=claims["sub"],
            claims={"iss": claims.get("iss")},
        )


def tenant_for_subject(subject: str) -> str:
    # Weaviate tenant names: letters, digits, '-' and '_', at most 64 characters
    return "u_" + re.sub(r"[^A-Za-z0-9_-]", "_", subject)[:62]


def current_tenant() -> str:
    """Tenant of the user making the current MCP request."""
    if not AUTH_ENABLED:
        return config.LOCAL_TENANT
    token = get_access_token()
    if token is None or not token.subject:
        raise PermissionError("Not signed in")
    return tenant_for_subject(token.subject)


# --------------------------------------------------------------------------- #
# Signed links
# --------------------------------------------------------------------------- #

# Without SECRET_KEY (local runs) links are signed with a per-process key and stop working on restart.
_SECRET = (config.SECRET_KEY or secrets.token_hex(32)).encode()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign(payload: dict, ttl_seconds: int) -> str:
    body = _b64(json.dumps({**payload, "exp": int(time.time()) + ttl_seconds}, separators=(",", ":")).encode())
    mac = _b64(hmac.new(_SECRET, body.encode(), hashlib.sha256).digest())
    return f"{body}.{mac}"


def unsign(token: str) -> dict | None:
    """The payload if the signature is valid and not expired, else None."""
    try:
        body, mac = token.split(".", 1)
        expected = _b64(hmac.new(_SECRET, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(mac, expected):
            return None
        payload = json.loads(_unb64(body))
    except (ValueError, json.JSONDecodeError):
        return None
    return payload if payload.get("exp", 0) >= time.time() else None
