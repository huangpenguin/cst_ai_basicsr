"""CPU-only UI regression tests. Network calls are mocked; never invokes Vertex."""

import base64
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

os.environ.setdefault(
    "VERTEX_PREDICT_URL",
    "https://example.invalid/v1/projects/580558088344/locations/asia-northeast1/endpoints/8911645097217490944:predict",
)
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/missing-test-adc")
spec = importlib.util.spec_from_file_location("vertex_ui", Path(__file__).with_name("internal_ui.py"))
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


class FrontendRegression(unittest.TestCase):
    def test_upload_preserves_uint16_and_clears_old_result(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "image.tif"
            pixels = np.array([[0, 256], [32768, 65535]], dtype=np.uint16)
            Image.fromarray(pixels).save(path)
            before = path.read_bytes()
            result = ui.inspect_upload(str(path), ui.VERTEX_PRESET_ID)
            self.assertEqual(result[1]["dtype"], "uint16")
            self.assertEqual(result[1]["pixel_max"], 65535)
            self.assertEqual(result[0].dtype, np.uint8)
            self.assertEqual(result[2:], (None, None, {}, ""))
            self.assertEqual(path.read_bytes(), before)

    def test_multipage_is_not_silently_truncated(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "stack.tif"
            frame = Image.fromarray(np.zeros((8, 8), dtype=np.uint16))
            frame.save(path, save_all=True, append_images=[frame])
            with self.assertRaisesRegex(Exception, "Multi-page"):
                ui.preview(path, ui.VERTEX_PRESET_ID)

    def test_unavailable_preset_cannot_predict(self):
        other = next(p.id for p in ui.REGISTRY.presets if p.id != ui.VERTEX_PRESET_ID)
        with patch.object(ui.requests, "post") as request:
            with self.assertRaisesRegex(Exception, "not deployed"):
                ui.infer(other, "unused")
            request.assert_not_called()

    def test_cloud_status_does_not_predict_or_invent_zero(self):
        endpoint = {
            "deployedModels": [
                {
                    "id": "123",
                    "status": {"message": "System error"},
                    "dedicatedResources": {"machineSpec": {"acceleratorType": "NVIDIA_TESLA_T4"}},
                }
            ]
        }
        with (
            patch.object(ui, "token", return_value="test"),
            patch.object(ui.requests, "post") as predict,
            patch.object(
                ui.requests, "get", side_effect=[Mock(ok=True, json=lambda: endpoint), Mock(ok=True, json=lambda: {})]
            ),
        ):
            result = ui.cloud_details()
            self.assertEqual(result["replica_samples"], [])
            self.assertIn("Not tested", result["prediction_health"])
            self.assertIn("unknown", result["replica_status"])
            self.assertEqual(result["deployments"][0]["control_plane_status"]["message"], "System error")
            predict.assert_not_called()

    def test_catalogue_yaml_path_is_not_a_gradio_download(self):
        info = Mock(error=None)
        info.model_dump.return_value = {"train_yaml": {"path": "/private/train.yml"}}
        with patch.object(ui, "inspect_preset", return_value=info):
            result = ui.checkpoint_details(ui.VERTEX_PRESET_ID)
        self.assertNotIn("path", result["train_yaml"])
        self.assertEqual(result["train_yaml"]["source_path"], "/private/train.yml")

    def test_infer_retains_download_contract(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "image.tif"
            Image.fromarray(np.arange(64, dtype=np.uint16).reshape(8, 8)).save(path)
            data = {
                "predictions": [
                    {
                        "image_b64": base64.b64encode(path.read_bytes()).decode(),
                        "preset": ui.VERTEX_PRESET_ID,
                        "device": "cuda",
                        "seconds": 0.1,
                    }
                ]
            }
            with (
                patch.object(ui, "token", return_value="test"),
                patch.object(
                    ui.requests, "post", return_value=Mock(ok=True, status_code=200, json=lambda: data)
                ) as predict,
            ):
                result = ui.infer(ui.VERTEX_PRESET_ID, str(path), progress=lambda *a, **k: None)
            self.assertEqual(len(result), 4)
            self.assertEqual(ui.image_details(result[2])["dtype"], "uint16")
            self.assertIn(ui.VERTEX_PRESET_ID, result[3])
            self.assertEqual(predict.call_count, 1)
            Path(result[2]).unlink()
            Path(result[2]).parent.rmdir()


if __name__ == "__main__":
    unittest.main()
