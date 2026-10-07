"""Turn uploaded files into email records: [{email_id, subject, sender, recipients, date, body}].

Supported uploads (several files at once are fine):
  * .csv   one email per row; a body column (body/message/text/content) plus optional
           subject, from/sender, to/recipients, date columns
  * .mbox  a mailbox export (Gmail Takeout, Thunderbird, ...)
  * .eml   one email per file (Outlook, Apple Mail, ...)
  * .txt   one email per file, either raw RFC 822 (headers + body) or just the body
  * .zip   any mix of the above

Bodies are cleaned with deterministic rules ported from the thesis preparation stage
(Thesis codes/1- Preparation/script.ipynb in https://github.com/aliloloee/forensic-analysis). The thesis's LLM-based cleaning is skipped to keep
uploads fast and cheap.
"""

import csv
import email
import io
import mailbox
import os
import re
import tempfile
import zipfile
from email import policy
from email.message import EmailMessage
from html.parser import HTMLParser
from pathlib import PurePath

from forensic_rag import config


class UploadError(ValueError):
    """A problem with the uploaded files that the user should see."""


# --------------------------------------------------------------------------- #
# Cleaning (from the thesis preparation notebook)
# --------------------------------------------------------------------------- #

ILLEGAL_CHAR_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
FORWARD_DIVIDER_RE = re.compile(r"(?im)^\s*-{2,}\s*forwarded by\b.*$")
ORIGINAL_MSG_LINE_RE = re.compile(r"(?im)^\s*-{2,}\s*original message\s*-{2,}\s*$")
ATTACH_RE = re.compile(r"(?im)^\s*<<\s*file:.*?>>\s*$")
REPLY_HDR_BLOCK_RE = re.compile(
    r"(?ims)^\s*from:\s.*?\n\s*sent:\s.*?\n\s*to:\s.*?\n\s*subject:\s.*?(?:\n\s*cc:\s.*)?\s*(?:\n|$)"
)
SHORT_PLEASANTRY_RE = re.compile(r"(?im)^\s*(thanks|thank you|thx|regards|best|sincerely|cheers)\s*[,\.\!]*\s*$")
# additions for modern mailboxes: quoted reply lines and "On <date>, <name> wrote:" markers
QUOTED_LINE_RE = re.compile(r"(?m)^\s*>.*$")
ON_WROTE_RE = re.compile(r"(?im)^\s*on .{5,200} wrote:\s*$")


def clean_body(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = ILLEGAL_CHAR_RE.sub("", text)
    for pattern in (FORWARD_DIVIDER_RE, ORIGINAL_MSG_LINE_RE, ATTACH_RE, REPLY_HDR_BLOCK_RE,
                    QUOTED_LINE_RE, ON_WROTE_RE):
        text = pattern.sub("", text)
    text = "\n".join(ln for ln in text.split("\n") if not SHORT_PLEASANTRY_RE.match(ln))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[: config.MAX_EMAIL_CHARS]


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("br", "p", "div", "li", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    parser = _HTMLText()
    parser.feed(markup)
    return "".join(parser.parts)


def decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


# --------------------------------------------------------------------------- #
# Format parsers -> raw records {subject, sender, recipients, date, body}
# --------------------------------------------------------------------------- #

def _from_message(msg: EmailMessage) -> dict:
    part = msg.get_body(preferencelist=("plain", "html"))
    body = ""
    if part is not None:
        try:
            body = part.get_content()
        except (LookupError, UnicodeError):
            body = decode_bytes(part.get_payload(decode=True) or b"")
        if part.get_content_type() == "text/html":
            body = html_to_text(body)
    return {
        "subject": str(msg.get("subject", "") or ""),
        "sender": str(msg.get("from", "") or ""),
        "recipients": str(msg.get("to", "") or ""),
        "date": str(msg.get("date", "") or ""),
        "body": body,
    }


def parse_eml(raw: bytes) -> list[dict]:
    return [_from_message(email.message_from_bytes(raw, policy=policy.default))]


def parse_mbox(raw: bytes) -> list[dict]:
    # mailbox needs a real file
    fd, path = tempfile.mkstemp(suffix=".mbox")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        box = mailbox.mbox(path, factory=lambda fp: email.message_from_binary_file(fp, policy=policy.default),
                           create=False)
        records = [_from_message(m) for m in box]
        box.close()
        return records
    finally:
        os.remove(path)


HEADER_LINE_RE = re.compile(r"^(from|to|subject|date|sent|cc)\s*:", re.IGNORECASE)


def parse_txt(raw: bytes, filename: str) -> list[dict]:
    text = decode_bytes(raw)
    first_lines = [ln for ln in text.lstrip().splitlines()[:6] if ln.strip()]
    if sum(bool(HEADER_LINE_RE.match(ln)) for ln in first_lines) >= 2:
        return parse_eml(text.encode("utf-8"))           # looks like headers + body
    return [{"subject": PurePath(filename).stem, "sender": "", "recipients": "", "date": "", "body": text}]


CSV_COLUMNS = {
    "body": ("body", "message", "text", "content", "email"),
    "subject": ("subject", "title"),
    "sender": ("from", "sender", "author"),
    "recipients": ("to", "recipients", "recipient"),
    "date": ("date", "sent", "timestamp"),
}


def parse_csv(raw: bytes) -> list[dict]:
    csv.field_size_limit(10_000_000)
    reader = csv.DictReader(io.StringIO(decode_bytes(raw)))
    headers = {h.strip().lower(): h for h in (reader.fieldnames or [])}
    mapping = {}
    for field, candidates in CSV_COLUMNS.items():
        mapping[field] = next((headers[c] for c in candidates if c in headers), None)
    if mapping["body"] is None:
        raise UploadError(f"CSV needs a body column (one of {', '.join(CSV_COLUMNS['body'])}); "
                          f"found: {', '.join(reader.fieldnames or [])}")
    return [{f: (row.get(col) or "") if col else "" for f, col in mapping.items()} for row in reader]


def parse_file(filename: str, raw: bytes) -> list[dict]:
    ext = PurePath(filename).suffix.lower()
    if ext == ".csv":
        return parse_csv(raw)
    if ext == ".mbox":
        return parse_mbox(raw)
    if ext == ".eml":
        return parse_eml(raw)
    if ext == ".txt":
        return parse_txt(raw, filename)
    if ext == ".zip":
        records = []
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for info in zf.infolist():
                name = PurePath(info.filename)
                if info.is_dir() or name.name.startswith(".") or "__MACOSX" in name.parts:
                    continue
                if name.suffix.lower() in (".csv", ".mbox", ".eml", ".txt"):
                    records.extend(parse_file(name.name, zf.read(info)))
        return records
    raise UploadError(f"Unsupported file type: {filename} (use .csv, .mbox, .eml, .txt or .zip)")


def parse_uploads(files: list[tuple[str, bytes]]) -> list[dict]:
    """Parse all uploaded files, clean bodies, drop empties, number emails 1..N."""
    records = []
    for filename, raw in files:
        records.extend(parse_file(filename, raw))

    emails = []
    for r in records:
        body = clean_body(r["body"])
        if len(body) < config.MIN_CHUNK_CHARS:
            continue
        emails.append({**r, "body": body, "subject": r["subject"].strip()[:500]})
        if len(emails) > config.MAX_UPLOAD_EMAILS:
            raise UploadError(f"Too many emails: the limit is {config.MAX_UPLOAD_EMAILS} per dataset")
    if not emails:
        raise UploadError("No emails with text were found in the uploaded files")
    for i, e in enumerate(emails, 1):
        e["email_id"] = i
    return emails
