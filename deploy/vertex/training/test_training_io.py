import io
import json
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
# I/O contract tests have no Google credential, SDK, GPU or cloud dependency.
fake = types.ModuleType("gcs")
fake.download = lambda *a: None
fake.upload = lambda *a: None
fake.headers = lambda: {}
fake.object_parts = lambda uri: tuple(uri[5:].split("/", 1))
with patch.dict(sys.modules, gcs=fake):
    import train_full as runner
from prepare_data import inspect  # noqa: E402 - test-local import path is set above.


class TrainingIO(unittest.TestCase):
    def test_acceptance_is_single_batch_without_changing_network(self):
        base = json.loads((Path(__file__).parent / "train-full.json").read_text())
        args = types.SimpleNamespace(
            gpu_count=1,
            batch_per_gpu=1,
            iterations=8,
            crop_size=128,
            mode="acceptance",
            validation_tile=128,
            validation_max=1,
            validation_interval=4,
        )
        config = runner.configure(base, args)
        self.assertEqual(config["network_g"], base["network_g"])
        self.assertEqual(base["num_gpu"], 4)  # Caller-owned template is untouched.
        self.assertEqual(config["num_gpu"], 1)
        self.assertEqual(config["datasets"]["train"]["batch_size_per_gpu"], 1)
        self.assertEqual(config["logger"]["save_checkpoint_freq"], 4)
        self.assertEqual(config["val"]["val_freq"], 4)
        self.assertEqual(config["val"]["max_num"], 1)
        self.assertEqual(config["train"]["scheduler"]["milestones"], [])
        args.iterations = 33
        with self.assertRaisesRegex(ValueError, "bounded"):
            runner.configure(base, args)
        args.iterations = 8
        args.crop_size = 127
        with self.assertRaisesRegex(ValueError, "multiple"):
            runner.configure(base, args)

    def test_splits_reject_test_leakage_and_missing_pairs(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for folder in ["GT", "LQ"]:
                (root / folder).mkdir()
                for name in ["a.tif", "b.tif"]:
                    (root / folder / name).write_bytes(b"unchanged source bytes")
            (root / "train.txt").write_text("a.tif\n")
            (root / "val_monitor.txt").write_text("b.tif\n")
            (root / "normalization.json").write_text('{"raw_min":0,"raw_max":65535}')
            self.assertEqual(len(inspect(root)[1]), 7)
            (root / "test.txt").write_text("a.tif\n")
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                inspect(root)
            (root / "test.txt").unlink()
            (root / "LQ" / "b.tif").unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                inspect(root)

    def test_output_prefix_cannot_overwrite_an_existing_run(self):
        occupied = types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"items": [{"name": "old"}]})
        empty = types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
        collision = types.SimpleNamespace(status_code=412)
        with (
            patch.object(runner.requests, "get", return_value=occupied),
            patch.object(runner.requests, "post") as write,
        ):
            with self.assertRaisesRegex(ValueError, "already contains"):
                runner.claim_output("gs://bucket/run", "a" * 64)
            write.assert_not_called()
        with (
            patch.object(runner.requests, "get", return_value=empty),
            patch.object(runner.requests, "post", return_value=collision),
        ):
            with self.assertRaisesRegex(ValueError, "another run"):
                runner.claim_output("gs://bucket/run", "a" * 64)

    def test_tar_rejects_escape_and_symlinks(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for name, kind, target in [("../escape", tarfile.REGTYPE, ""), ("link", tarfile.SYMTYPE, "/etc/passwd")]:
                archive_path = root / "data.tar"
                with tarfile.open(archive_path, "w") as archive:
                    entry = tarfile.TarInfo(name)
                    entry.type, entry.linkname = kind, target
                    archive.addfile(entry, io.BytesIO(b""))
                with patch.object(runner.shutil, "disk_usage", return_value=types.SimpleNamespace(free=100 * 1024**3)):
                    with self.assertRaises(ValueError):
                        runner.unpack(archive_path, root)
            self.assertFalse((root.parent / "escape").exists())

    def test_manifest_commits_immutable_snapshot_and_restore_checks_hash(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            out = root / "out"
            out.mkdir()
            source = out / "train.log"
            source.write_bytes(b"before")
            objects = {}

            def upload(path, uri):
                if "/objects/" in uri:
                    source.write_bytes(b"after")
                objects[uri] = path.read_bytes()

            with patch.object(runner, "upload", side_effect=upload):
                runner.Persist(out, "gs://bucket/run").sync(final=True)
            manifest = json.loads(objects["gs://bucket/run/artifact-manifest.json"])
            obj = manifest["files"]["train.log"]["object"]
            self.assertEqual(objects["gs://bucket/run/" + obj], b"before")
            restored = root / "restored"
            restored.mkdir()

            def download(uri, path):
                path.write_bytes(objects[uri] if uri.endswith("artifact-manifest.json") else b"corrupt")

            with patch.object(runner, "download", side_effect=download):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    runner.restore("gs://bucket/run", restored)


if __name__ == "__main__":
    unittest.main()
