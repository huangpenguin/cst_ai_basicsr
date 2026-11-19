"""Mount identity and capacity fail-closed checks with isolated filesystem fixtures."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.xray_portal import store

NFS = "192.168.3.14:/mnt/storage/data"
GIB = 1024**3


def mount_line(target, *, source=NFS, fs="nfs", options="rw,relatime"):
    encoded = str(target).replace(" ", "\\040")
    return f"36 25 0:31 / {encoded} {options} - {fs} {source} {options}\n"


@pytest.fixture
def storage(tmp_path, monkeypatch):
    mount = tmp_path / "company storage"
    root = mount / "cst_ai" / "intake"
    root.mkdir(parents=True)
    monkeypatch.setattr(store, "DATA", root)
    monkeypatch.setenv("PORTAL_NFS_SOURCE", NFS)
    monkeypatch.setattr(store.os, "access", lambda path, mode: True)
    monkeypatch.setattr(store.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * GIB))
    original_read = Path.read_text
    state = {"mountinfo": mount_line(mount)}

    def fake_read(path, *args, **kwargs):
        if path == Path("/proc/self/mountinfo"):
            return state["mountinfo"]
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read)
    return mount, root, state


def test_host_nfs_mount_and_escaped_space_path_accepted(storage):
    _, _, _ = storage
    assert store.check_storage(20 * GIB) == {"source": NFS, "free_bytes": 100 * GIB}


def test_container_exact_nfs_subdirectory_source_accepted(storage):
    _, root, state = storage
    state["mountinfo"] = mount_line(root, source=NFS + "/cst_ai/intake", fs="nfs4")
    assert store.check_storage()["source"] == NFS


@pytest.mark.parametrize(
    "source", ["192.168.3.15:/mnt/storage/data", NFS + "-fake", NFS + "/other", NFS + "/cst_ai/intake/subdir"]
)
def test_wrong_or_arbitrary_prefix_source_refused(storage, source):
    _, root, state = storage
    state["mountinfo"] = mount_line(root, source=source)
    with pytest.raises(RuntimeError, match="not mounted read-write"):
        store.check_storage()


def test_existing_local_lookalike_directory_is_not_nfs(storage):
    _, root, state = storage
    assert root.is_dir()
    state["mountinfo"] = mount_line("/", source="/dev/sda2", fs="ext4")
    with pytest.raises(RuntimeError, match="not mounted read-write"):
        store.check_storage()
    assert root.is_dir()


def test_empty_mount_table_does_not_create_missing_directory(storage):
    _, root, state = storage
    root.rmdir()
    state["mountinfo"] = ""
    with pytest.raises(RuntimeError, match="not mounted read-write"):
        store.check_storage()
    assert not root.exists()


def test_readonly_mount_is_refused(storage):
    mount, _, state = storage
    state["mountinfo"] = mount_line(mount, options="ro,relatime")
    with pytest.raises(RuntimeError, match="not mounted read-write"):
        store.check_storage()


def test_more_specific_local_bind_mount_overrides_parent_nfs(storage):
    mount, root, state = storage
    state["mountinfo"] = mount_line(mount) + mount_line(root, source="/dev/sda2", fs="ext4")
    with pytest.raises(RuntimeError, match="not mounted read-write"):
        store.check_storage()


def test_mount_with_similar_path_is_not_an_ancestor(storage):
    mount, _, state = storage
    state["mountinfo"] = mount_line(str(mount)[:-1])
    with pytest.raises(RuntimeError, match="not mounted read-write"):
        store.check_storage()


def test_nfs_directory_must_exist_and_be_writable(storage, monkeypatch):
    _, root, _ = storage
    monkeypatch.setattr(store.os, "access", lambda path, mode: False)
    with pytest.raises(RuntimeError, match="not writable"):
        store.check_storage()
    monkeypatch.setattr(store.os, "access", lambda path, mode: True)
    root.rmdir()
    with pytest.raises(RuntimeError, match="not writable"):
        store.check_storage()
    assert not root.exists()


def test_reserved_bytes_and_safety_margin_are_enforced(storage, monkeypatch):
    monkeypatch.setattr(store.shutil, "disk_usage", lambda path: SimpleNamespace(free=24 * GIB))
    with pytest.raises(RuntimeError, match="Insufficient"):
        store.check_storage(20 * GIB)
    monkeypatch.setattr(store.shutil, "disk_usage", lambda path: SimpleNamespace(free=25 * GIB))
    assert store.check_storage(20 * GIB)["free_bytes"] == 25 * GIB
