import io
import zipfile

import pytest

from forensic_rag.parsers import UploadError, parse_uploads

BODY = "Please move the prepay to next quarter so the debt does not show."

EML = (
    "From: alice@example.com\nTo: bob@example.com\nSubject: Prepay timing\n"
    "Date: Tue, 12 Dec 2000 08:59:00 -0800\nContent-Type: text/plain\n\n" + BODY
).encode()


def test_csv_with_header_aliases():
    csv = f"Subject,From,To,Date,Message\nPrepay,alice,bob,2000-12-12,{BODY}\n".encode()
    [email] = parse_uploads([("mail.csv", csv)])
    assert email["email_id"] == 1
    assert email["subject"] == "Prepay" and email["sender"] == "alice" and email["recipients"] == "bob"
    assert BODY in email["body"]


def test_csv_without_body_column_is_rejected():
    with pytest.raises(UploadError, match="body column"):
        parse_uploads([("mail.csv", b"subject,from\nhi,alice\n")])


def test_eml():
    [email] = parse_uploads([("one.eml", EML)])
    assert email["subject"] == "Prepay timing"
    assert BODY in email["body"]


def test_zip_with_mixed_files_is_numbered_in_order():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.eml", EML)
        zf.writestr("notes/b.txt", "Second email: the auditors are coming, keep only the clean version.")
        zf.writestr("__MACOSX/._a.eml", "resource fork junk")
    emails = parse_uploads([("mailbox.zip", buf.getvalue())])
    assert [e["email_id"] for e in emails] == [1, 2]
    assert emails[1]["subject"] == "b"


def test_short_bodies_are_dropped_and_empty_upload_rejected():
    with pytest.raises(UploadError, match="No emails"):
        parse_uploads([("x.txt", b"ok")])


def test_unsupported_file_type():
    with pytest.raises(UploadError, match="Unsupported"):
        parse_uploads([("mail.pdf", b"%PDF")])
