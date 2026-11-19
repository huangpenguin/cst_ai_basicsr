"""Internal HTTP pilot: upload ownership, explicit review, no training endpoint."""

import base64
import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import store

HERE = Path(__file__).parent
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
store.initialize()
if (HERE / "static").exists():
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


async def json_object(request):
    try:
        data = await request.json()
    except (ValueError, UnicodeDecodeError):
        fail(422, "Expected a JSON object")
    if not isinstance(data, dict):
        fail(422, "Expected a JSON object")
    return data


def fail(code, message):
    raise HTTPException(code, message)


def session(request):
    key = store.token_hash(request.cookies.get("portal_session", ""))
    with store.connect() as db:
        row = db.execute("SELECT doc,expires FROM sessions WHERE id=?", (key,)).fetchone()
    if not row or row["expires"] < time.time():
        fail(401, "Open the portal or sign in again")
    return key, json.loads(row["doc"])


def csrf(request, sess):
    if not hmac.compare_digest(request.headers.get("x-csrf-token", ""), sess["csrf_token"]):
        fail(403, "Session check failed; reload the page")
    origin = request.headers.get("origin")
    host = request.headers.get("host")
    if origin and origin.rstrip("/") != f"http://{host}" and origin.rstrip("/") != f"https://{host}":
        fail(403, "Cross-origin request denied")


def access(request, batch_id=None, admin=False, write=False):
    _, sess = session(request)
    if write:
        csrf(request, sess)
    if admin and not sess.get("admin"):
        fail(403, "Reviewer login required")
    if batch_id and not sess.get("admin") and batch_id not in sess.get("batch_ids", []):
        fail(403, "This batch belongs to another session; enter its access code")
    return sess


def batch_read(batch_id):
    with store.connect() as db:
        try:
            return store.get_batch(db, batch_id)
        except KeyError:
            fail(404, "Batch not found")


def mutable(batch):
    if batch["status"] in store.BUSY or batch.get("version"):
        fail(409, "This batch is busy or frozen; create a new batch for new files")


def clean(value, label, limit=160):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or any(ord(c) < 32 for c in value):
        fail(422, f"{label} must contain 1–{limit} printable characters")
    return value.strip()


def throttle(db, key, maximum, seconds=3600):
    count = db.execute(
        "SELECT COUNT(*) FROM attempts WHERE key=? AND created>?", (key, store.now() - seconds)
    ).fetchone()[0]
    if count >= maximum:
        fail(429, "Too many attempts; please wait before retrying")
    db.execute("INSERT INTO attempts VALUES(?,?)", (key, store.now()))
    db.execute("DELETE FROM attempts WHERE created<?", (store.now() - 86400,))


def public_batch(batch):
    value = json.loads(json.dumps(batch))
    for f in value.get("files", []):
        f.pop("path", None)
    return value


@app.exception_handler(HTTPException)
async def http_error(request, exc):
    status = exc.status_code
    if request.url.path == "/internal/upload-auth" and status != 401:
        status = 403
    return JSONResponse({"detail": exc.detail}, status_code=status)


@app.exception_handler(ValueError)
async def bad_value(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.exception_handler(RuntimeError)
async def unavailable(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=503)


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "templates/index.html").read_text()


@app.get("/health")
def health():
    return {"ok": True, "service": "xray-portal"}


@app.get("/api/session")
def get_session(request: Request):
    try:
        _, sess = session(request)
        return sess
    except HTTPException:
        token = secrets.token_urlsafe(32)
        sess = {"admin": False, "batch_ids": [], "csrf_token": secrets.token_urlsafe(32)}
        with store.connect() as db:
            db.execute(
                "INSERT INTO sessions VALUES(?,?,?)",
                (store.token_hash(token), json.dumps(sess), store.now() + 7 * 86400),
            )
        response = JSONResponse(sess)
        response.set_cookie("portal_session", token, httponly=True, samesite="strict", max_age=7 * 86400)
        return response


@app.post("/api/admin/login")
async def login(request: Request):
    key, sess = session(request)
    csrf(request, sess)
    data = await json_object(request)
    with store.connect() as db:
        throttle(db, "login:" + request.client.host, 12, 900)
    expected = os.environ.get("PORTAL_ADMIN_PASSWORD", "")
    if not expected or not hmac.compare_digest(str(data.get("password", "")), expected):
        fail(401, "Invalid reviewer password")
    sess["admin"] = True
    token = secrets.token_urlsafe(32)
    sess["csrf_token"] = secrets.token_urlsafe(32)
    with store.transaction() as db:
        db.execute("DELETE FROM sessions WHERE id=?", (key,))
        db.execute(
            "INSERT INTO sessions VALUES(?,?,?)", (store.token_hash(token), json.dumps(sess), store.now() + 8 * 3600)
        )
    response = JSONResponse(sess)
    response.set_cookie("portal_session", token, httponly=True, samesite="strict", max_age=8 * 3600)
    return response


@app.post("/api/logout")
def logout(request: Request):
    key, sess = session(request)
    csrf(request, sess)
    with store.connect() as db:
        db.execute("DELETE FROM sessions WHERE id=?", (key,))
    response = JSONResponse({"ok": True})
    response.delete_cookie("portal_session")
    return response


@app.post("/api/access")
async def recover(request: Request):
    key, sess = session(request)
    csrf(request, sess)
    data = await json_object(request)
    data["batch_id"] = str(data.get("batch_id", "")).strip().lower()
    with store.transaction() as db:
        throttle(db, "access:" + request.client.host, 30, 900)
        row = db.execute("SELECT access_hash FROM batches WHERE id=?", (data.get("batch_id"),)).fetchone()
    if not row or not hmac.compare_digest(store.token_hash(str(data.get("access_code", ""))), row["access_hash"]):
        fail(403, "Batch ID or access code is incorrect")
    sess["batch_ids"] = sorted(set(sess["batch_ids"] + [data["batch_id"]]))
    with store.connect() as db:
        db.execute("UPDATE sessions SET doc=? WHERE id=?", (json.dumps(sess), key))
    return sess


@app.post("/api/batches")
async def create_batch(request: Request):
    key, sess = session(request)
    csrf(request, sess)
    data = await json_object(request)
    fields = {n: clean(data.get(n), n) for n in ("title", "workpiece", "scan_group", "uploader")}
    store.check_storage()
    batch_id, code = secrets.token_hex(12), secrets.token_urlsafe(24)
    batch = {
        "id": batch_id,
        **fields,
        "status": "draft",
        "created_at": store.now(),
        "files": [],
        "report": None,
        "decisions": None,
        "version": None,
        "error": None,
        "events": [],
    }
    store.event(batch, "Batch created", fields["uploader"])
    with store.transaction() as db:
        throttle(db, "create:" + request.client.host, 20)
        (store.DATA / batch_id / "original").mkdir(parents=True, mode=0o750)
        db.execute("INSERT INTO batches VALUES(?,?,?)", (batch_id, store.token_hash(code), json.dumps(batch)))
        sess["batch_ids"].append(batch_id)
        db.execute("UPDATE sessions SET doc=? WHERE id=?", (json.dumps(sess), key))
    return {"id": batch_id, "access_code": code}


@app.get("/api/batches")
def batches(request: Request):
    sess = access(request)
    with store.connect() as db:
        records = [json.loads(r[0]) for r in db.execute("SELECT doc FROM batches")]
    return [
        {k: b.get(k) for k in ("id", "title", "status", "workpiece", "scan_group", "created_at", "updated_at")}
        for b in records
        if sess["admin"] or b["id"] in sess["batch_ids"]
    ]


@app.get("/api/batches/{batch_id}")
def batch_detail(batch_id: str, request: Request, after: float | None = None):
    access(request, batch_id)
    batch = batch_read(batch_id)
    if after is not None and after == batch.get("updated_at"):
        return {"unchanged": True}
    return public_batch(batch)


@app.post("/api/batches/{batch_id}/files")
async def register_file(batch_id: str, request: Request):
    access(request, batch_id, write=True)
    data = await json_object(request)
    filename = clean(data.get("filename"), "Filename", 255)
    if (
        Path(filename).name != filename
        or "\\" in filename
        or Path(filename).suffix.lower() not in (".zip", ".tif", ".tiff")
    ):
        fail(422, "Choose a TIFF or ZIP with a simple filename")
    size = data.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= store.MAX_FILE:
        fail(422, "File must be between 1 byte and 32 GiB")
    fingerprint = data.get("fingerprint", "")
    if not isinstance(fingerprint, str) or not re.fullmatch("[0-9a-f]{64}", fingerprint):
        fail(422, "Full file SHA256 is required")
    if data.get("role") not in ("GT", "LQ"):
        fail(422, "Select Noisy images or Reference images")
    condition = clean(data.get("condition"), "Capture condition")
    with store.transaction() as db:
        batch = store.get_batch(db, batch_id)
        mutable(batch)
        for f in batch["files"]:
            if f["status"] != "excluded" and (
                f["fingerprint"],
                f["role"],
                f["condition"],
                f["workpiece"],
                f["scan_group"],
            ) == (
                fingerprint,
                data["role"],
                condition,
                data.get("workpiece") or batch["workpiece"],
                data.get("scan_group") or batch["scan_group"],
            ):
                if f["size"] != size:
                    fail(409, "The registered checksum already belongs to a different declared size")
                if (
                    f["status"] == "uploading"
                    and store.now() - f.get("upload_started", 0) > 30
                    and not (store.STATE / "tus" / (f["id"] + ".info")).exists()
                    and not Path(f["path"]).exists()
                ):
                    f["status"] = "registered"
                    store.put_batch(db, batch)
                return {
                    "id": f["id"],
                    "upload_url": "/uploads/" + f["id"] if f["status"] != "registered" else None,
                    "status": f["status"],
                }
        if sum(f["size"] for f in batch["files"]) + size > store.MAX_BATCH or len(batch["files"]) >= 50000:
            fail(422, "Batch exceeds its 64 GiB or 50,000-file limit")
        # Reserve enough for declared originals plus bounded extraction and training copy.
        reserved = sum(
            store.MAX_EXPANDED * 3 + store.MAX_BATCH
            for r in db.execute("SELECT doc FROM batches")
            if json.loads(r[0])["status"] not in ("cloud_ready", "rejected")
        )
        store.check_storage(reserved + size)
        file_id = secrets.token_hex(16)
        f = {
            "id": file_id,
            "filename": filename,
            "size": size,
            "fingerprint": fingerprint,
            "role": data["role"],
            "condition": condition,
            "workpiece": clean(data.get("workpiece") or batch["workpiece"], "Workpiece"),
            "scan_group": clean(data.get("scan_group") or batch["scan_group"], "Scan group"),
            "path": str(store.DATA / batch_id / "original" / file_id),
            "status": "registered",
        }
        batch["files"].append(f)
        batch["status"], batch["report"], batch["decisions"] = "uploading", None, None
        store.put_batch(db, batch)
    return {"id": file_id, "upload_url": None, "status": "registered"}


@app.post("/api/batches/{batch_id}/files/{file_id}/exclude")
async def exclude_file(batch_id: str, file_id: str, request: Request):
    access(request, batch_id, write=True)
    data = await json_object(request)
    reason = clean(data.get("reason"), "Exclusion reason", 2000)
    with store.transaction() as db:
        batch = store.get_batch(db, batch_id)
        mutable(batch)
        f = next((f for f in batch["files"] if f["id"] == file_id), None)
        if not f:
            fail(404, "Upload not found")
        f["status"], f["exclusion_reason"] = "excluded", reason
        batch["report"], batch["decisions"], batch["status"] = None, None, "pending_validation"
        store.event(batch, f"Upload excluded, original retained: {f['filename']}. Reason: {reason}", "uploader")
        store.put_batch(db, batch)
    return {"ok": True}


@app.post("/api/batches/{batch_id}/validate")
def validate(batch_id: str, request: Request):
    access(request, batch_id, write=True)
    with store.transaction() as db:
        batch = store.get_batch(db, batch_id)
        mutable(batch)
        if not batch["files"]:
            fail(409, "Upload files before validation")
        for f in batch["files"]:
            if f["status"] == "excluded":
                continue
            path = store.DATA / batch_id / "original" / f["id"]
            if f["status"] not in ("uploaded", "received", "invalid"):
                # Recover a lost completion hook only when tus and the complete file exist.
                if (
                    f["status"] == "uploading"
                    and (store.STATE / "tus" / (f["id"] + ".info")).is_file()
                    and path.is_file()
                    and path.stat().st_size == f["size"]
                ):
                    f["status"] = "uploaded"
                else:
                    fail(409, "Finish all uploads before validation")
        return {"job_id": store.enqueue(db, batch, "validate")}


@app.post("/api/batches/{batch_id}/review")
async def review(batch_id: str, request: Request):
    access(request, batch_id, admin=True, write=True)
    data = await json_object(request)
    if not isinstance(data.get("pairs"), list) or data.get("split_mode") not in ("group", "spatial"):
        fail(422, "Choose group or spatial split and provide reviewed pairs")
    for pair in data["pairs"]:
        if not isinstance(pair, dict) or not isinstance(pair.get("id"), str):
            fail(422, "Each pair decision must contain its candidate ID")
        if type(pair.get("approved")) is not bool or pair.get("split") not in ("train", "monitor", "test", "exclude"):
            fail(422, "Each pair needs a boolean approval and a valid split")
        if pair.get("gt_id") is not None and not isinstance(pair["gt_id"], str):
            fail(422, "Reference mapping must be an image ID")
    guard = data.get("guard_slices", 16)
    if type(guard) is not int or not 1 <= guard <= 100000:
        fail(422, "Guard band must be a positive integer")
    if type(data.get("acknowledge_issues", False)) is not bool:
        fail(422, "Issue acknowledgement must be a boolean")
    data["rationale"] = clean(data.get("rationale"), "Review notes", 2000)
    data["reviewed_by"] = "reviewer"
    with store.transaction() as db:
        batch = store.get_batch(db, batch_id)
        mutable(batch)
        if not batch.get("report") or batch["status"] not in ("awaiting_review", "reviewed", "freeze_failed"):
            fail(409, "Validate the batch before review")
        if any(f["status"] not in ("received", "excluded") for f in batch["files"]):
            fail(409, "All source transfers must pass checksum verification")
        candidate_ids = {pair["id"] for pair in batch["report"]["candidate_pairs"]}
        decision_ids = [pair["id"] for pair in data["pairs"]]
        if len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != candidate_ids:
            fail(422, "Every candidate must have exactly one decision")
        batch["decisions"], batch["status"] = data, "reviewed"
        store.event(batch, "Pairing and split decisions saved", "reviewer")
        store.put_batch(db, batch)
    return {"ok": True}


@app.post("/api/batches/{batch_id}/freeze")
def freeze(batch_id: str, request: Request):
    access(request, batch_id, admin=True, write=True)
    with store.transaction() as db:
        batch = store.get_batch(db, batch_id)
        if batch["status"] not in ("reviewed", "freeze_failed") or not batch["decisions"]:
            fail(409, "Save the review before freezing")
        batch["pending_version"] = secrets.token_hex(16)
        return {"job_id": store.enqueue(db, batch, "freeze")}


@app.post("/api/batches/{batch_id}/sync")
def sync(batch_id: str, request: Request):
    access(request, batch_id, admin=True, write=True)
    with store.transaction() as db:
        batch = store.get_batch(db, batch_id)
        if batch["status"] == "cloud_ready":
            return {"ok": True, "version": batch["version"]}
        if batch["status"] not in ("frozen", "sync_failed") or not batch.get("version"):
            fail(409, "Freeze an approved version before cloud synchronization")
        return {"job_id": store.enqueue(db, batch, "sync")}


@app.post("/api/batches/{batch_id}/reject")
async def reject(batch_id: str, request: Request):
    access(request, batch_id, admin=True, write=True)
    data = await json_object(request)
    reason = clean(data.get("reason"), "Reason", 2000)
    with store.transaction() as db:
        batch = store.get_batch(db, batch_id)
        mutable(batch)
        batch["status"] = "rejected"
        store.event(batch, reason, "reviewer")
        store.put_batch(db, batch)
    return {"ok": True}


@app.get("/api/batches/{batch_id}/preview/{image_id}")
def preview(batch_id: str, image_id: str, request: Request):
    access(request, batch_id)
    batch = batch_read(batch_id)
    for image in (batch.get("report") or {}).get("images", []):
        if image["id"] == image_id and image.get("preview"):
            path = (store.DATA / batch_id / image["preview"]).resolve()
            if not path.is_relative_to((store.DATA / batch_id / "derived").resolve()):
                fail(403, "Invalid preview path")
            return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})
    fail(404, "Preview unavailable")


@app.get("/api/batches/{batch_id}/image/{image_id}")
def original_image(batch_id: str, image_id: str, request: Request):
    access(request, batch_id)
    batch = batch_read(batch_id)
    for image in (batch.get("report") or {}).get("images", []):
        if image["id"] == image_id:
            path = (store.DATA / batch_id / image["relative_path"]).resolve()
            if not path.is_relative_to((store.DATA / batch_id).resolve()):
                fail(403, "Invalid image path")
            return FileResponse(
                path,
                media_type="image/tiff",
                filename=Path(image["filename"]).name,
                headers={"Cache-Control": "private, no-store"},
            )
    fail(404, "Original image unavailable")


def metadata(value):
    result = {}
    try:
        for item in value.split(","):
            key, encoded = item.strip().split(" ", 1)
            result[key] = base64.b64decode(encoded, validate=True).decode()
    except (ValueError, UnicodeDecodeError):
        fail(400, "Invalid upload metadata")
    return result


def lookup_upload(file_id):
    with store.connect() as db:
        for row in db.execute("SELECT doc FROM batches"):
            b = json.loads(row[0])
            for f in b["files"]:
                if f["id"] == file_id:
                    return b, f
    fail(404, "Register the file before upload")


@app.api_route("/internal/upload-auth", methods=["GET", "POST", "HEAD", "PATCH", "OPTIONS"])
def upload_auth(request: Request):
    method = request.headers.get("x-original-method", "")
    if request.headers.get("upload-concat") or request.headers.get("upload-defer-length"):
        fail(403, "Concatenated or deferred-length uploads are not supported")
    if method not in ("POST", "HEAD", "PATCH", "OPTIONS"):
        fail(403, "Upload operation not supported")
    _, sess = session(request)
    if method in ("POST", "PATCH"):
        csrf(request, sess)
    uri = request.headers.get("x-original-uri", "").split("?")[0]
    if method == "OPTIONS":
        return Response(status_code=204)
    if method == "POST":
        meta = metadata(request.headers.get("upload-metadata", ""))
        batch, f = lookup_upload(meta.get("file_id", ""))
        if meta.get("batch_id") != batch["id"] or meta.get("fingerprint") != f["fingerprint"]:
            fail(403, "Upload metadata does not match the registered source")
        if request.headers.get("upload-length") != str(f["size"]):
            fail(422, "Upload length differs from the registered file")
    else:
        batch, f = lookup_upload(uri.rstrip("/").rsplit("/", 1)[-1])
    if not sess.get("admin") and batch["id"] not in sess["batch_ids"]:
        fail(403, "Upload belongs to another batch")
    if method in ("POST", "PATCH"):
        mutable(batch)
        if f["status"] in ("uploaded", "received", "invalid", "excluded"):
            fail(409, "Upload already completed")
        store.check_storage()
    return Response(status_code=204)


@app.post("/internal/tus")
async def tus_hook(request: Request):
    body = await request.json()
    ev = body.get("Event", {})
    headers = {k.lower(): v for k, v in ev.get("HTTPRequest", {}).get("Header", {}).items()}
    secret = headers.get("x-portal-hook-secret", [""])
    secret = secret[0] if isinstance(secret, list) else secret
    expected = os.environ.get("PORTAL_HOOK_SECRET", "")
    if not expected or not hmac.compare_digest(secret, expected):
        fail(403, "Private hook required")
    upload = ev.get("Upload", {})
    meta = upload.get("MetaData", {})
    kind = body.get("Type")
    batch, f = lookup_upload(meta.get("file_id", upload.get("ID", "")))
    if kind == "pre-create":
        if any(upload.get(k) for k in ("IsPartial", "IsFinal", "SizeIsDeferred")):
            return {
                "RejectUpload": True,
                "HTTPResponse": {"StatusCode": 400, "Body": "Explicit standalone upload required"},
            }
        mutable(batch)
        store.check_storage()
        if (
            meta.get("batch_id") != batch["id"]
            or meta.get("fingerprint") != f["fingerprint"]
            or upload.get("Size") != f["size"]
        ):
            return {"RejectUpload": True, "HTTPResponse": {"StatusCode": 400, "Body": "Registered upload mismatch"}}
        with store.transaction() as db:
            b = store.get_batch(db, batch["id"])
            actual = next(x for x in b["files"] if x["id"] == f["id"])
            if actual["status"] != "registered":
                return {"RejectUpload": True, "HTTPResponse": {"StatusCode": 409, "Body": "Resume the existing upload"}}
            mutable(b)
            actual["status"] = "uploading"
            actual["upload_started"] = store.now()
            store.put_batch(db, b)
        return {"ChangeFileInfo": {"ID": f["id"], "Storage": {"Path": f["path"]}}}
    if kind in ("post-finish", "pre-finish"):
        with store.transaction() as db:
            b = store.get_batch(db, batch["id"])
            actual = next(x for x in b["files"] if x["id"] == f["id"])
            # tus may retry notifications after review, or deliver pre/post finish twice.
            # Only the first completion can advance an uploading file; never reopen a frozen batch.
            if actual["status"] != "uploading" or b.get("version") or b["status"] in store.BUSY:
                return {}
            actual["status"] = "uploaded"
            if b["status"] in ("uploading", "draft", "pending_validation") and all(
                x["status"] in ("uploaded", "received", "excluded") for x in b["files"]
            ):
                b["status"] = "pending_validation"
            store.put_batch(db, b)
    return {}
