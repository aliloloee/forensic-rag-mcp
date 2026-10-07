"""Browser-facing routes of the hosted server: health check, upload page, report pages.

Uploading 200+ emails does not fit through a chat, so the `create_upload_link` MCP tool hands the
user a signed, single-use link to this upload page. The link carries the user's tenant, so the
page needs no second login, and whatever is uploaded lands in that user's tenant only.
"""

import html
import json
import re
import secrets
import threading
import time

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from forensic_rag import auth, config, store
from forensic_rag.ingest import ingest_emails, log
from forensic_rag.parsers import UploadError, parse_uploads
from forensic_rag.report import render_html

DATASET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,39}$")
HOSTED_RESULTS = config.RESULTS_DIR / "hosted"

# In-memory state; fine for a single small instance (a restart forgets running jobs).
JOBS: dict[str, dict] = {}
_used_upload_nonces: set[str] = set()
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Links (used by the MCP tools)
# --------------------------------------------------------------------------- #

def make_upload_link(tenant: str) -> str:
    token = auth.sign({"k": "upload", "t": tenant, "n": secrets.token_urlsafe(8)},
                      config.UPLOAD_LINK_MINUTES * 60)
    return f"{config.PUBLIC_URL}/upload?t={token}"


def save_report(tenant: str, report: dict) -> str:
    """Store a report for its owner and return a signed link to its HTML page."""
    rid = secrets.token_urlsafe(9)
    folder = HOSTED_RESULTS / tenant
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{rid}.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    token = auth.sign({"k": "report", "t": tenant, "r": rid}, config.REPORT_LINK_DAYS * 86400)
    return f"{config.PUBLIC_URL}/reports/{token}"


def jobs_for(tenant: str) -> list[dict]:
    with _lock:
        return [_public_job(j) for j in JOBS.values() if j["tenant"] == tenant and j["status"] == "running"]


def _public_job(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "tenant"}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

def register_routes(mcp) -> None:
    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    @mcp.custom_route("/upload", methods=["GET"])
    async def upload_page(request: Request) -> Response:
        payload = auth.unsign(request.query_params.get("t", ""))
        if not payload or payload.get("k") != "upload":
            return _message_page("This upload link has expired or is invalid.",
                                 "Ask Claude for a new one (the create_upload_link tool).", 403)
        return HTMLResponse(UPLOAD_PAGE.format(
            token=html.escape(request.query_params["t"]),
            max_emails=config.MAX_UPLOAD_EMAILS, max_mb=config.MAX_UPLOAD_MB,
            minutes=config.UPLOAD_LINK_MINUTES,
        ))

    @mcp.custom_route("/upload", methods=["POST"])
    async def upload(request: Request) -> Response:
        if int(request.headers.get("content-length") or 0) > config.MAX_UPLOAD_MB * 1024 * 1024:
            return _error(f"The upload is larger than {config.MAX_UPLOAD_MB} MB")
        form = await request.form(max_files=2000, max_part_size=1024 * 1024)
        payload = auth.unsign(str(form.get("t", "")))
        if not payload or payload.get("k") != "upload":
            return _error("This upload link has expired. Ask Claude for a new one.", 403)
        tenant = payload["t"]

        dataset = str(form.get("dataset", "")).strip().lower()
        if not DATASET_NAME_RE.match(dataset):
            return _error("Dataset name: 2-40 characters, lowercase letters, digits, '-' or '_'")
        if any(d["dataset"] == dataset for d in store.list_datasets(config.PUBLIC_TENANT)):
            return _error(f"'{dataset}' is the name of a shared dataset; please choose another name")
        existing = {d["dataset"] for d in store.list_datasets(tenant)}
        if dataset not in existing and len(existing) >= config.MAX_DATASETS_PER_USER:
            return _error(f"You already have {len(existing)} datasets (the limit). Delete one first.")

        files = []
        total = 0
        for item in form.getlist("files"):
            if not hasattr(item, "read"):
                continue
            data = await item.read()
            total += len(data)
            if total > config.MAX_UPLOAD_MB * 1024 * 1024:
                return _error(f"The upload is larger than {config.MAX_UPLOAD_MB} MB")
            files.append((item.filename or "upload.txt", data))
        if not files:
            return _error("Choose at least one file")

        try:
            emails = parse_uploads(files)
        except UploadError as exc:
            return _error(str(exc))
        except Exception as exc:
            return _error(f"Could not read the files: {exc}")

        with _lock:   # the link is single-use: consume it only once the upload is accepted
            if payload["n"] in _used_upload_nonces:
                return _error("This upload link was already used. Ask Claude for a new one.", 403)
            _used_upload_nonces.add(payload["n"])
            job_id = secrets.token_urlsafe(12)
            JOBS[job_id] = {"job_id": job_id, "tenant": tenant, "dataset": dataset, "status": "running",
                            "stage": "queued", "progress": 0.0, "emails": len(emails), "error": None,
                            "started": time.time()}
        threading.Thread(target=_run_ingest, args=(job_id, tenant, dataset, emails), daemon=True).start()
        return JSONResponse({"job_id": job_id, "dataset": dataset, "emails": len(emails)})

    @mcp.custom_route("/upload/status/{job_id}", methods=["GET"])
    async def upload_status(request: Request) -> Response:
        with _lock:
            job = JOBS.get(request.path_params["job_id"])
            return JSONResponse(_public_job(job)) if job else _error("Unknown job", 404)

    @mcp.custom_route("/reports/{token}", methods=["GET"])
    async def report_page(request: Request) -> Response:
        payload = auth.unsign(request.path_params["token"])
        if not payload or payload.get("k") != "report":
            return _message_page("This report link has expired or is invalid.", "Run the investigation again.", 403)
        path = HOSTED_RESULTS / payload["t"] / f"{payload['r']}.json"
        if not path.exists():
            return _message_page("This report is no longer available.",
                                 "Reports are kept on the server's disk and are lost when it restarts.", 404)
        return HTMLResponse(render_html(json.loads(path.read_text(encoding="utf-8"))))


def _run_ingest(job_id: str, tenant: str, dataset: str, emails: list[dict]) -> None:
    def progress(stage: str, fraction: float) -> None:
        with _lock:
            JOBS[job_id].update(stage=stage, progress=round(fraction, 3))

    try:
        stats = ingest_emails(dataset, emails, tenant=tenant, on_progress=progress)
        with _lock:
            JOBS[job_id].update(status="done", stage="done", progress=1.0, chunks=stats["chunks"])
    except Exception as exc:
        log(f"[upload {job_id}] failed: {exc!r}")
        with _lock:
            JOBS[job_id].update(status="failed", error=str(exc))


def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _message_page(title: str, detail: str, status: int) -> HTMLResponse:
    return HTMLResponse(PAGE_SHELL.format(body=f"<h1>{html.escape(title)}</h1><p>{html.escape(detail)}</p>"),
                        status_code=status)


PAGE_CSS = """
:root{--bg:#f7f7f5;--surface:#fff;--text:#1d1d1f;--muted:#6b6b70;--border:#e3e3e0;--accent:#2f5bd3;
  --ok:#17803d;--err:#b42318;--track:#ececea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141416;--surface:#1d1d20;
  --text:#ececee;--muted:#9a9aa2;--border:#2e2e33;--accent:#7d9bff;--ok:#5fd38a;--err:#ff8a80;--track:#2a2a2f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:640px;margin:0 auto;padding:40px 16px}
h1{font-size:1.4rem;margin:0 0 8px}
p{color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px;margin-top:20px}
label{display:block;font-weight:600;margin:14px 0 6px}
input[type=text]{width:100%;padding:9px 10px;border:1px solid var(--border);border-radius:6px;
  background:var(--bg);color:var(--text);font:inherit}
.drop{border:2px dashed var(--border);border-radius:8px;padding:22px;text-align:center;color:var(--muted)}
button{margin-top:18px;background:var(--accent);color:#fff;border:0;border-radius:6px;padding:10px 18px;
  font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.hint{font-size:.85rem;color:var(--muted)}
.bar{height:8px;background:var(--track);border-radius:99px;overflow:hidden;margin-top:12px}
.bar>div{height:100%;width:0;background:var(--accent);transition:width .4s}
.status{margin-top:10px}.ok{color:var(--ok)}.err{color:var(--err)}
code{background:var(--track);padding:1px 5px;border-radius:4px}
"""

PAGE_SHELL = ("<!doctype html><html lang=en><head><meta charset=utf-8>"
              "<meta name=viewport content='width=device-width, initial-scale=1'><title>Forensic RAG</title>"
              "<style>" + PAGE_CSS.replace("{", "{{").replace("}", "}}") + "</style></head>"
              "<body><main>{body}</main></body></html>")

UPLOAD_PAGE = PAGE_SHELL.replace("{body}", """
<h1>Upload your emails</h1>
<p>The emails are cleaned, split into semantic chunks, embedded and stored in your private space.
Only you can search them, from Claude. This link works once and expires {minutes} minutes after it was created.</p>
<form class="card" id="f">
  <input type="hidden" name="t" value="{token}">
  <label for="dataset">Dataset name</label>
  <input type="text" id="dataset" name="dataset" placeholder="e.g. my-mailbox" required pattern="[a-z0-9][a-z0-9_-]{{1,39}}">
  <div class="hint">Lowercase letters, digits, - or _. You will use this name in Claude.</div>
  <label for="files">Files</label>
  <div class="drop"><input type="file" id="files" name="files" multiple required
     accept=".csv,.mbox,.eml,.txt,.zip"></div>
  <div class="hint">.csv (a body column, plus optional subject/from/to/date), .mbox, .eml, .txt, or a .zip of them.
  Up to {max_emails} emails and {max_mb} MB.</div>
  <button id="go" type="submit">Upload and index</button>
  <div class="bar" hidden id="bar"><div></div></div>
  <div class="status" id="status"></div>
</form>
<script>
const f = document.getElementById('f'), go = document.getElementById('go'),
      bar = document.getElementById('bar'), status = document.getElementById('status');
const say = (msg, cls) => {{ status.textContent = msg; status.className = 'status ' + (cls || ''); }};
f.addEventListener('submit', async (e) => {{
  e.preventDefault(); go.disabled = true; say('Uploading and reading the files...');
  try {{
    const r = await fetch('upload', {{ method: 'POST', body: new FormData(f) }});
    const d = await r.json();
    if (!r.ok) {{ say(d.error, 'err'); go.disabled = false; return; }}
    bar.hidden = false; say(d.emails + ' emails found. Indexing...');
    const poll = async () => {{
      const s = await (await fetch('upload/status/' + d.job_id)).json();
      bar.firstElementChild.style.width = Math.round((s.progress || 0) * 100) + '%';
      if (s.status === 'done') say('Done: ' + s.emails + ' emails, ' + s.chunks + ' chunks in dataset "' + s.dataset +
          '". Go back to Claude and investigate a hypothesis on it.', 'ok');
      else if (s.status === 'failed') say('Indexing failed: ' + s.error, 'err');
      else {{ say('Indexing: ' + s.stage + '...'); setTimeout(poll, 1500); }}
    }};
    poll();
  }} catch (err) {{ say('Upload failed: ' + err, 'err'); go.disabled = false; }}
}});
</script>""")
