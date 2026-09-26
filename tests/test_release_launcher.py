"""CPU-only integrity checks for the release wrapper, not scientific experiments."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_training
import validate_data


class ReleaseLauncherTests(unittest.TestCase):
    def test_validator_detects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shard_00001.bin"
            index = root / "freq_index.npz"
            shard.write_bytes(b"abcd")
            index.write_bytes(b"index")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": [
                {"name": p.name, "role": role, "bytes": p.stat().st_size,
                 "sha256": validate_data.sha256(p)}
                for p, role in ((shard, "tokenized_shard"), (index, "frequency_index"))
            ]}))
            with mock.patch.object(validate_data, "MANIFEST", manifest):
                self.assertTrue(validate_data.validate(root, index)["ok"])
                shard.write_bytes(b"abce")
                result = validate_data.validate(root, index)
                self.assertFalse(result["ok"])
                self.assertEqual(result["errors"], ["SHA-256 mismatch for shard_00001.bin"])

    def test_plan_does_not_create_output_and_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = [sys.executable, str(ROOT / "scripts/run_training.py"),
                       "--data-dir", str(root / "absent"),
                       "--freq-index", str(root / "absent.npz"),
                       "--run-id", "check_fd", "--out-dir", str(root / "outputs")]
            planned = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(planned.returncode, 0, planned.stderr)
            self.assertIn("Plan only", planned.stdout)
            self.assertFalse((root / "outputs").exists())
            existing = root / "outputs/check_fd_fixed"
            existing.mkdir(parents=True)
            sentinel = existing / "keep.txt"
            sentinel.write_text("retained")
            collision = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(collision.returncode, 0)
            self.assertIn("refusing to overwrite", collision.stderr)
            self.assertEqual(sentinel.read_text(), "retained")

    def test_incomplete_or_inconsistent_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"seed": 42, "steps": 20, "val_interval": 10}
            summary = {"run_id": "fixture_fixed", "seed": 42, "steps": 20,
                       "final_train_loss": 1.0, "final_val_loss": 2.0, "final_gap": 1.0}
            (root / "summary.json").write_text(json.dumps(summary))
            rows = [{"step": step, "train_loss": 1.0, "val_loss": 2.0, "gap": 1.0} for step in (10, 20)]
            (root / "train_log.jsonl").write_text("\n".join(map(json.dumps, rows)))
            for name in ("freq_bin_loss.jsonl", "exact_freq_loss.jsonl"):
                (root / name).write_text("\n".join(json.dumps({"step": step}) for step in (10, 20)))
            self.assertTrue(run_training.result_check(root, "fixture_fixed", config)["ok"])
            (root / "exact_freq_loss.jsonl").write_text('{"step": 10}\n')
            with self.assertRaisesRegex(ValueError, "incomplete"):
                run_training.result_check(root, "fixture_fixed", config)
            summary["final_gap"] = 3.0
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "gap mismatch"):
                run_training.result_check(root, "fixture_fixed", config)


if __name__ == "__main__":
    unittest.main()
