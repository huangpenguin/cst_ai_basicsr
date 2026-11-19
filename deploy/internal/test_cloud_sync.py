"""Offline transport checks; never use ADC or make cloud requests."""

import json
from pathlib import Path

import pytest
import requests

from apps.xray_portal import cloud_sync as cloud


class Response:
    def __init__(self, code, data=None, headers=None):
        self.status_code = code
        self._data = data
        self.headers = headers or {}

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, expected, fail_once=False):
        self.expected = expected
        self.calls = []
        self.offset = 0
        self.stored = False
        self.fail_once = fail_once
        self.created = 0

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if method == "GET":
            return Response(200, self.metadata()) if self.stored else Response(404)
        if method == "POST":
            assert kwargs["params"]["ifGenerationMatch"] == "0"
            self.created += 1
            return Response(200, headers={"Location": "https://storage.googleapis.com/upload/session-secret"})
        if kwargs["data"] == b"":
            if self.stored:
                return Response(200, self.metadata())
            return Response(308, headers={"Range": f"bytes=0-{self.offset - 1}"} if self.offset else {})
        if self.fail_once:
            self.fail_once = False
            # Simulate a chunk being durably received but its response lost.
            self.offset += len(kwargs["data"])
            self.stored = self.offset == self.expected["size"]
            raise requests.ConnectionError("SECRET https://storage.googleapis.com/upload/session-secret")
        content_range = kwargs["headers"]["Content-Range"]
        assert content_range.startswith(f"bytes {self.offset}-")
        self.offset += len(kwargs["data"])
        if self.offset == self.expected["size"]:
            self.stored = True
            return Response(200, self.metadata())
        return Response(308, headers={"Range": f"bytes=0-{self.offset - 1}"})

    def metadata(self):
        return {
            "size": str(self.expected["size"]),
            "crc32c": self.expected["crc32c"],
            "metadata": {"sha256": self.expected["sha256"]},
            "generation": "123",
        }


def test_interruption_preserves_private_resume_state_and_redacts_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "CHUNK", 256 * 1024)
    source = tmp_path / "dataset.tar"
    source.write_bytes(b"a" * (700 * 1024))
    state = tmp_path / "state"
    state.mkdir()
    session = FakeSession(cloud._checksums(source), fail_once=True)
    with pytest.raises(RuntimeError) as error:
        cloud._upload(session, source, "datasets/portal/v1/dataset.tar", state)
    assert "session-secret" not in str(error.value)
    resume = next(state.glob("*.json"))
    assert resume.stat().st_mode & 0o777 == 0o600
    assert json.loads(resume.read_text())["session_url"].endswith("session-secret")
    result = cloud._upload(session, source, "datasets/portal/v1/dataset.tar", state)
    assert result["generation"] == "123"
    assert session.created == 1
    assert not list(state.glob("*.json"))


def test_existing_content_must_match_crc_and_sha(tmp_path):
    source = tmp_path / "dataset.tar"
    source.write_bytes(b"small synthetic dataset")
    session = FakeSession(cloud._checksums(source))
    session.stored = True
    assert cloud._upload(session, source, "datasets/portal/v1/dataset.tar", tmp_path)["generation"] == "123"
    session.expected["crc32c"] = "wrong"
    with pytest.raises(RuntimeError, match="different content"):
        cloud._upload(session, source, "datasets/portal/v1/dataset.tar", tmp_path)


def test_changed_file_cannot_reuse_old_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "CHUNK", 256 * 1024)
    source = tmp_path / "dataset.tar"
    source.write_bytes(b"a" * (700 * 1024))
    state = tmp_path / "state"
    state.mkdir()
    session = FakeSession(cloud._checksums(source), fail_once=True)
    with pytest.raises(RuntimeError):
        cloud._upload(session, source, "datasets/portal/v1/dataset.tar", state)
    source.write_bytes(b"b" * (700 * 1024))
    with pytest.raises(RuntimeError, match="does not match"):
        cloud._upload(session, source, "datasets/portal/v1/dataset.tar", state)


def test_fixed_allowlist_and_commit_last(tmp_path, monkeypatch):
    version = tmp_path / "version"
    version.mkdir()
    for name in ("dataset.tar", "manifest.json", "pairs_manifest.json", "normalization.json", "raw.zip"):
        (version / name).write_text("synthetic")
    (version / "manifest.json").write_text(json.dumps({"sha256": cloud._checksums(version / "dataset.tar")["sha256"]}))
    (version / "GT").mkdir()
    (version / "GT" / "test.tif").write_bytes(b"private-test")
    calls = []

    def upload(session, path, name, state):
        calls.append(Path(path).name)
        return {"uri": "gs://fixed/" + name, **cloud._checksums(path), "generation": "123"}

    class Session:
        def close(self):
            pass

    monkeypatch.setattr(cloud.google.auth, "default", lambda **kw: (object(), None))
    monkeypatch.setattr(cloud, "AuthorizedSession", lambda credentials: Session())
    monkeypatch.setattr(cloud, "_upload", upload)
    result = cloud.sync_version(version, "v1", tmp_path / "state")
    assert calls == [
        "dataset.tar",
        "manifest.json",
        "pairs_manifest.json",
        "normalization.json",
        "v1-cloud-manifest.json",
    ]
    assert result["manifest_uri"].endswith("cloud-manifest.json")
    assert "raw.zip" not in result["files"]


def test_invalid_version_and_symlink_rejected_before_auth(tmp_path):
    with pytest.raises(ValueError, match="version ID"):
        cloud.sync_version(tmp_path, "../../escape", tmp_path / "state")
    (tmp_path / "dataset.tar").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symbolic"):
        cloud.sync_version(tmp_path, "v1", tmp_path / "state")


def test_frozen_archive_tampering_rejected_before_auth(tmp_path):
    (tmp_path / "dataset.tar").write_bytes(b"tampered")
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": "0" * 64}))
    with pytest.raises(ValueError, match="reviewed manifest"):
        cloud.sync_version(tmp_path, "v1", tmp_path / "state")
