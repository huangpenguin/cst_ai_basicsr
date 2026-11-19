"""Portal API authorization and durable review gate tests, using isolated storage."""

import base64
import hashlib
import importlib
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def portal(tmp_path, monkeypatch):
    from apps.xray_portal import store

    monkeypatch.setattr(store, "STATE", tmp_path / "state")
    monkeypatch.setattr(store, "DATA", tmp_path / "data")
    store.DATA.mkdir()
    monkeypatch.setattr(store, "check_storage", lambda required=0: {"free_bytes": 1024**4})
    monkeypatch.setenv("PORTAL_ADMIN_PASSWORD", "test-only-reviewer-password")
    monkeypatch.setenv("PORTAL_HOOK_SECRET", "test-only-private-hook")
    module = importlib.import_module("apps.xray_portal.app")
    store.initialize()
    with TestClient(module.app) as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        yield client, csrf, store, module


def post(client, csrf, path, payload=None, **kwargs):
    return client.post(path, json=payload or {}, headers={"x-csrf-token": csrf, **kwargs})


def make_batch(client, csrf):
    response = post(
        client,
        csrf,
        "/api/batches",
        {"title": "Test scan", "workpiece": "part-a", "scan_group": "scan-1", "uploader": "colleague"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def admin(client, csrf):
    response = post(client, csrf, "/api/admin/login", {"password": "test-only-reviewer-password"})
    assert response.status_code == 200
    return response.json()["csrf_token"]


def register(client, csrf, batch, **changes):
    payload = {
        "filename": "condition.zip",
        "role": "LQ",
        "condition": "10uA",
        "size": 123,
        "fingerprint": "a" * 64,
        **changes,
    }
    response = post(client, csrf, f"/api/batches/{batch}/files", payload)
    return response


def modify_batch(store, batch, **changes):
    with store.transaction() as db:
        record = store.get_batch(db, batch)
        record.update(changes)
        store.put_batch(db, record)
    return record


def metadata(**values):
    return ",".join(f"{name} {base64.b64encode(str(value).encode()).decode()}" for name, value in values.items())


def auth_upload(client, csrf, file_id, batch_id, **changes):
    headers = {
        "x-original-method": "POST",
        "x-original-uri": "/uploads/",
        "x-csrf-token": csrf,
        "upload-length": "123",
        "upload-metadata": metadata(file_id=file_id, batch_id=batch_id, fingerprint="a" * 64),
        **changes,
    }
    return client.get("/internal/upload-auth", headers=headers)


def test_anonymous_and_colleague_cannot_review_freeze_sync(portal):
    client, csrf, _, module = portal
    batch = make_batch(client, csrf)["id"]
    for action in ("review", "freeze", "sync", "reject"):
        assert post(client, csrf, f"/api/batches/{batch}/{action}").status_code == 403
    with TestClient(module.app) as anonymous:
        assert anonymous.get(f"/api/batches/{batch}").status_code == 401
        assert anonymous.post(f"/api/batches/{batch}/sync").status_code == 401


def test_csrf_and_cross_origin_mutations_blocked(portal):
    client, csrf, _, _ = portal
    assert client.post("/api/batches", json={}).status_code == 403
    assert post(client, csrf, "/api/batches", origin="https://other-site.example").status_code == 403
    assert post(client, "incorrect", "/api/admin/login", {"password": "test-only-reviewer-password"}).status_code == 403


def test_other_session_cannot_read_preview_or_resume_upload(portal):
    client, csrf, _, module = portal
    batch = make_batch(client, csrf)["id"]
    file_id = register(client, csrf, batch).json()["id"]
    with TestClient(module.app) as other:
        other_csrf = other.get("/api/session").json()["csrf_token"]
        assert other.get(f"/api/batches/{batch}").status_code == 403
        assert other.get(f"/api/batches/{batch}/preview/image-id").status_code == 403
        assert auth_upload(other, other_csrf, file_id, batch).status_code == 403
        assert (
            other.get(
                "/internal/upload-auth", headers={"x-original-method": "HEAD", "x-original-uri": f"/uploads/{file_id}"}
            ).status_code
            == 403
        )


def test_access_code_recovers_only_requested_batch(portal):
    client, csrf, _, module = portal
    first = make_batch(client, csrf)
    second = make_batch(client, csrf)
    with TestClient(module.app) as other:
        other_csrf = other.get("/api/session").json()["csrf_token"]
        assert (
            post(other, other_csrf, "/api/access", {"batch_id": first["id"], "access_code": "wrong"}).status_code == 403
        )
        assert (
            post(
                other, other_csrf, "/api/access", {"batch_id": first["id"], "access_code": first["access_code"]}
            ).status_code
            == 200
        )
        assert other.get(f"/api/batches/{first['id']}").status_code == 200
        assert other.get(f"/api/batches/{second['id']}").status_code == 403
        assert not other.get("/api/session").json()["admin"]


def test_upload_declared_size_and_full_fingerprint_must_match(portal):
    client, csrf, _, _ = portal
    batch = make_batch(client, csrf)["id"]
    file_id = register(client, csrf, batch).json()["id"]
    assert auth_upload(client, csrf, file_id, batch).status_code == 204
    assert auth_upload(client, csrf, file_id, batch, **{"upload-length": "124"}).status_code == 403
    wrong = metadata(file_id=file_id, batch_id=batch, fingerprint="b" * 64)
    assert auth_upload(client, csrf, file_id, batch, **{"upload-metadata": wrong}).status_code == 403
    assert auth_upload(client, "invalid", file_id, batch).status_code == 403
    assert register(client, csrf, batch, fingerprint="partial-hash").status_code == 422
    assert register(client, csrf, batch, size=True).status_code == 422
    assert register(client, csrf, batch, size=33 * 1024**3).status_code == 422


def test_validate_requires_all_uploads_complete(portal):
    client, csrf, _, _ = portal
    batch = make_batch(client, csrf)["id"]
    assert post(client, csrf, f"/api/batches/{batch}/validate").status_code == 409
    register(client, csrf, batch)
    assert post(client, csrf, f"/api/batches/{batch}/validate").status_code == 409


def test_reviewer_cannot_skip_validation_review_or_freeze(portal):
    client, csrf, store, _ = portal
    batch = make_batch(client, csrf)["id"]
    csrf = admin(client, csrf)
    review = {"pairs": [], "split_mode": "group", "rationale": "Review notes"}
    assert post(client, csrf, f"/api/batches/{batch}/review", review).status_code == 409
    assert post(client, csrf, f"/api/batches/{batch}/freeze").status_code == 409
    assert post(client, csrf, f"/api/batches/{batch}/sync").status_code == 409
    modify_batch(
        store,
        batch,
        status="awaiting_review",
        report={"images": [], "candidate_pairs": [], "issues": []},
        files=[{"status": "uploaded"}],
    )
    assert post(client, csrf, f"/api/batches/{batch}/review", review).status_code == 409
    modify_batch(store, batch, files=[{"status": "received"}])
    assert post(client, csrf, f"/api/batches/{batch}/review", review).status_code == 200
    assert post(client, csrf, f"/api/batches/{batch}/sync").status_code == 409
    frozen = post(client, csrf, f"/api/batches/{batch}/freeze")
    assert frozen.status_code == 200
    assert post(client, csrf, f"/api/batches/{batch}/freeze").status_code == 409
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_frozen_batch_blocks_new_uploads_and_reviews(portal):
    client, csrf, store, _ = portal
    batch = make_batch(client, csrf)["id"]
    file_id = register(client, csrf, batch).json()["id"]
    modify_batch(store, batch, status="frozen", version={"id": "v1"})
    assert register(client, csrf, batch, fingerprint="b" * 64).status_code == 409
    assert auth_upload(client, csrf, file_id, batch).status_code == 403
    csrf = admin(client, csrf)
    assert (
        post(
            client,
            csrf,
            f"/api/batches/{batch}/review",
            {"pairs": [], "split_mode": "group", "rationale": "Already frozen"},
        ).status_code
        == 409
    )


def test_private_hook_authentication_required(portal):
    client, _, _, _ = portal
    assert client.post("/internal/tus", json={"Type": "post-finish", "Event": {}}).status_code == 403


def test_worker_rechecks_bytes_before_marking_received(portal):
    client, csrf, store, _ = portal
    from apps.xray_portal import worker

    batch = make_batch(client, csrf)["id"]
    payload = b"actually-received-original"
    file_id = register(client, csrf, batch, size=len(payload)).json()["id"]
    (store.DATA / batch / "original" / file_id).write_bytes(payload)
    with pytest.raises(ValueError, match="SHA256 differs"):
        worker.run_job({"id": "unit-job", "batch_id": batch, "kind": "validate"})
    state = client.get(f"/api/batches/{batch}").json()
    assert state["files"][0]["status"] == "invalid"
    assert state["report"] is None


def test_worker_recovers_export_completed_before_db_commit(portal, monkeypatch):
    client, csrf, store, _ = portal
    from apps.xray_portal import worker

    batch = make_batch(client, csrf)["id"]
    version = store.DATA / batch / "versions" / "recovered"
    version.mkdir(parents=True)
    payload = b"stand-in-for-reviewed-archive"
    (version / "dataset.tar").write_bytes(payload)
    manifest = {"sha256": hashlib.sha256(payload).hexdigest(), "counts": {"train": 1, "monitor": 1, "test": 0}}
    (version / "manifest.json").write_text(json.dumps(manifest))
    modify_batch(store, batch, status="freezing", pending_version="recovered")
    with store.connect() as db:
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,NULL)", ("freeze-job", batch, "freeze", "running", store.now()))
    monkeypatch.setattr(worker, "export_dataset", lambda *args: pytest.fail("Completed export must be recovered"))
    worker.run_job({"id": "freeze-job", "batch_id": batch, "kind": "freeze"})
    state = client.get(f"/api/batches/{batch}").json()
    assert state["status"] == "frozen"
    assert state["version"]["id"] == "recovered"
    with store.connect() as db:
        assert db.execute("SELECT status FROM jobs WHERE id='freeze-job'").fetchone()[0] == "succeeded"


def test_upload_concat_and_unknown_length_are_denied(portal):
    client, csrf, _, _ = portal
    batch = make_batch(client, csrf)["id"]
    file_id = register(client, csrf, batch).json()["id"]
    assert auth_upload(client, csrf, file_id, batch, **{"upload-concat": "partial"}).status_code == 403
    assert auth_upload(client, csrf, file_id, batch, **{"upload-defer-length": "1"}).status_code == 403


def test_same_bytes_different_workpiece_are_separate_sources(portal):
    client, csrf, _, _ = portal
    batch = make_batch(client, csrf)["id"]
    first = register(client, csrf, batch).json()["id"]
    second = register(client, csrf, batch, workpiece="another-part").json()["id"]
    assert first != second


def test_invalid_upload_can_be_excluded_without_deleting_original(portal):
    client, csrf, store, _ = portal
    batch = make_batch(client, csrf)["id"]
    file_id = register(client, csrf, batch).json()["id"]
    original = store.DATA / batch / "original" / file_id
    original.write_bytes(b"invalid image")
    with store.transaction() as db:
        record = store.get_batch(db, batch)
        record["files"][0]["status"] = "invalid"
        store.put_batch(db, record)
    result = post(client, csrf, f"/api/batches/{batch}/files/{file_id}/exclude", {"reason": "Replacement required"})
    assert result.status_code == 200
    assert original.read_bytes() == b"invalid image"
    state = client.get(f"/api/batches/{batch}").json()
    assert state["files"][0]["status"] == "excluded"
    assert register(client, csrf, batch).json()["id"] != file_id


def test_lost_completion_hook_can_recover_complete_file(portal):
    client, csrf, store, _ = portal
    batch = make_batch(client, csrf)["id"]
    file_id = register(client, csrf, batch).json()["id"]
    (store.DATA / batch / "original" / file_id).write_bytes(b"a" * 123)
    (store.STATE / "tus").mkdir()
    (store.STATE / "tus" / (file_id + ".info")).write_text(json.dumps({"ID": file_id, "Size": 123}))
    with store.transaction() as db:
        record = store.get_batch(db, batch)
        record["files"][0]["status"] = "uploading"
        store.put_batch(db, record)
    assert post(client, csrf, f"/api/batches/{batch}/validate").status_code == 200
    assert client.get(f"/api/batches/{batch}").json()["status"] == "validating"


def test_recovered_export_with_changed_bytes_is_rejected(portal):
    client, csrf, store, _ = portal
    from apps.xray_portal import worker

    batch = make_batch(client, csrf)["id"]
    version = store.DATA / batch / "versions" / "damaged"
    version.mkdir(parents=True)
    (version / "dataset.tar").write_bytes(b"changed")
    (version / "manifest.json").write_text(json.dumps({"sha256": "a" * 64}))
    modify_batch(store, batch, status="freezing", pending_version="damaged")
    with pytest.raises(ValueError, match="checksum failed"):
        worker.run_job({"id": "test-job", "batch_id": batch, "kind": "freeze"})
    assert client.get(f"/api/batches/{batch}").json()["version"] is None


def hook_payload(batch_id, file_id, kind="pre-create"):
    return {
        "Type": kind,
        "Event": {
            "HTTPRequest": {"Header": {"X-Portal-Hook-Secret": ["test-only-private-hook"]}},
            "Upload": {
                "ID": file_id,
                "Size": 123,
                "MetaData": {"batch_id": batch_id, "file_id": file_id, "fingerprint": "a" * 64},
            },
        },
    }


def test_aborted_precreate_registration_can_restart_after_grace_period(portal):
    client, csrf, store, _ = portal
    batch = make_batch(client, csrf)["id"]
    file_id = register(client, csrf, batch).json()["id"]
    created = client.post("/internal/tus", json=hook_payload(batch, file_id))
    assert created.status_code == 200
    assert created.json()["ChangeFileInfo"]["ID"] == file_id
    # A concurrent request must not reset a live creation hook.
    assert register(client, csrf, batch).json()["status"] == "uploading"
    with store.transaction() as db:
        record = store.get_batch(db, batch)
        record["files"][0]["upload_started"] = store.now() - 31
        store.put_batch(db, record)
    recovered = register(client, csrf, batch).json()
    assert recovered == {"id": file_id, "status": "registered", "upload_url": None}
    retried = client.post("/internal/tus", json=hook_payload(batch, file_id)).json()
    assert retried["ChangeFileInfo"]["ID"] == file_id


@pytest.mark.parametrize("existing", ["protocol", "original"])
def test_registration_never_resets_existing_upload_bytes_or_protocol(portal, existing):
    client, csrf, store, _ = portal
    batch = make_batch(client, csrf)["id"]
    file_id = register(client, csrf, batch).json()["id"]
    if existing == "protocol":
        (store.STATE / "tus").mkdir()
        (store.STATE / "tus" / (file_id + ".info")).write_text("{}")
    else:
        (store.DATA / batch / "original" / file_id).write_bytes(b"partial-data")
    with store.transaction() as db:
        record = store.get_batch(db, batch)
        record["files"][0].update(status="uploading", upload_started=store.now() - 60)
        store.put_batch(db, record)
    resumed = register(client, csrf, batch).json()
    assert resumed["status"] == "uploading"
    assert resumed["upload_url"] == "/uploads/" + file_id


def test_excluded_partial_cannot_be_revived_by_late_hook_or_patch(portal):
    client, csrf, store, _ = portal
    batch = make_batch(client, csrf)["id"]
    file_id = register(client, csrf, batch).json()["id"]
    assert client.post("/internal/tus", json=hook_payload(batch, file_id)).status_code == 200
    original = store.DATA / batch / "original" / file_id
    original.write_bytes(b"partial-preserved")
    assert (
        post(
            client, csrf, f"/api/batches/{batch}/files/{file_id}/exclude", {"reason": "Wrong condition uploaded"}
        ).status_code
        == 200
    )
    for kind in ("pre-finish", "post-finish"):
        assert client.post("/internal/tus", json=hook_payload(batch, file_id, kind)).status_code == 200
        assert client.get(f"/api/batches/{batch}").json()["files"][0]["status"] == "excluded"
    patch_auth = client.get(
        "/internal/upload-auth",
        headers={"x-original-method": "PATCH", "x-original-uri": f"/uploads/{file_id}", "x-csrf-token": csrf},
    )
    assert patch_auth.status_code == 403
    assert auth_upload(client, csrf, file_id, batch).status_code == 403
    assert original.read_bytes() == b"partial-preserved"


def test_original_download_requires_batch_ownership_and_confined_path(portal):
    client, csrf, store, module = portal
    batch = make_batch(client, csrf)["id"]
    original = store.DATA / batch / "original" / "slice.tif"
    original.write_bytes(b"original-image-bytes")
    report = {"images": [{"id": "slice1", "relative_path": "original/slice.tif", "filename": "XY/slice.tif"}]}
    modify_batch(store, batch, report=report)
    downloaded = client.get(f"/api/batches/{batch}/image/slice1")
    assert downloaded.status_code == 200
    assert downloaded.content == b"original-image-bytes"
    assert "no-store" in downloaded.headers["cache-control"]
    assert "slice.tif" in downloaded.headers["content-disposition"]
    with TestClient(module.app) as other:
        assert other.get(f"/api/batches/{batch}/image/slice1").status_code == 401
        other.get("/api/session")
        assert other.get(f"/api/batches/{batch}/image/slice1").status_code == 403
    report["images"][0]["relative_path"] = "../../outside.tif"
    modify_batch(store, batch, report=report)
    assert client.get(f"/api/batches/{batch}/image/slice1").status_code == 403


@pytest.mark.parametrize("status", ["awaiting_review", "reviewed", "freezing", "frozen", "syncing", "cloud_ready"])
def test_duplicate_completion_hook_does_not_regress_review_or_frozen_state(portal, status):
    client, csrf, store, _ = portal
    batch = make_batch(client, csrf)["id"]
    file_id = register(client, csrf, batch).json()["id"]
    with store.transaction() as db:
        record = store.get_batch(db, batch)
        record["files"][0]["status"] = "received"
        record["status"] = status
        if status in {"frozen", "syncing", "cloud_ready"}:
            record["version"] = {"id": "immutable-version"}
        store.put_batch(db, record)
    for kind in ("pre-finish", "post-finish"):
        response = client.post("/internal/tus", json=hook_payload(batch, file_id, kind))
        assert response.status_code == 200
        current = client.get(f"/api/batches/{batch}").json()
        assert current["status"] == status
        assert current["files"][0]["status"] == "received"


def test_upload_registration_rejects_conflicting_size_for_same_identity(portal):
    client, csrf, _, _ = portal
    batch = make_batch(client, csrf)["id"]
    register(client, csrf, batch)
    response = register(client, csrf, batch, size=124)
    assert response.status_code in (409, 422)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"pairs": ["not-a-pair"], "split_mode": "group"},
        {"pairs": [{"id": "pair", "approved": "yes", "split": "train"}], "split_mode": "group"},
    ],
)
def test_malformed_review_never_changes_review_state(portal, payload):
    client, csrf, store, module = portal
    batch = make_batch(client, csrf)["id"]
    csrf = admin(client, csrf)
    modify_batch(
        store,
        batch,
        status="awaiting_review",
        report={"images": [], "candidate_pairs": [], "issues": []},
        files=[{"status": "received"}],
    )
    with TestClient(module.app, raise_server_exceptions=False) as browser:
        browser.cookies.update(client.cookies)
        response = browser.post(f"/api/batches/{batch}/review", json=payload, headers={"x-csrf-token": csrf})
    assert response.status_code == 422
    assert client.get(f"/api/batches/{batch}").json()["status"] == "awaiting_review"


def test_late_worker_failure_does_not_overwrite_committed_success(portal):
    client, csrf, store, _ = portal
    from apps.xray_portal import worker

    batch = make_batch(client, csrf)["id"]
    modify_batch(store, batch, status="frozen", version={"id": "completed-version"})
    with store.connect() as db:
        db.execute(
            "INSERT INTO jobs VALUES(?,?,?,?,?,NULL)", ("committed-job", batch, "freeze", "succeeded", store.now())
        )
    worker.mark_failure({"id": "committed-job", "batch_id": batch, "kind": "freeze"}, "Late timeout observation")
    current = client.get(f"/api/batches/{batch}").json()
    assert current["status"] == "frozen"
    assert current["error"] is None
    with store.connect() as db:
        assert db.execute("SELECT status FROM jobs WHERE id='committed-job'").fetchone()[0] == "succeeded"
