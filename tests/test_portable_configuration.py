import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("configure_stage3", ROOT / "tools/configure_server.py")
configure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(configure)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stage3 portable ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for rel in [
            "st-ghost/configs/stage1_exp1_main.json", "st-ghost/configs/stage3_original_ghost_comparison.json",
            "st-ghost/configs/train_projector_qwen.json", "migration_inputs/ranking_cache/resolved_config.json",
            "migration_inputs/reference/stghost_reference.json",
        ]:
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((ROOT / rel).read_bytes())
        for p in (ROOT / "migration_inputs/ranking_cache").glob("*.csv"):
            (self.root / "migration_inputs/ranking_cache" / p.name).write_bytes(p.read_bytes())
        self.patch = mock.patch.object(configure, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.settings = json.loads((ROOT / "configs/server_paths.example.json").read_text())
        self.settings.update(run_name="test_model", victim_model_path="assets/new victim")
        p = self.root / "local.json"
        p.write_text(json.dumps(self.settings))
        self.settings = configure.load_settings(p)

    def test_profiles_relocate_all_model_paths_and_preserve_stage3_settings(self):
        run = configure.configure(self.settings)
        base = json.loads((run / "configs/base.json").read_text())
        ref = json.loads((run / "stghost_reference/resolved_config.json").read_text())
        original = json.loads((run / "configs/original_ghost.json").read_text())
        self.assertEqual(base["victim_model_path"], str(self.root / "assets/new victim"))
        self.assertEqual(ref["victim_model_path"], base["victim_model_path"])
        self.assertEqual(ref["condition_stage_count"], 21)
        self.assertEqual(ref["lambda_victim"], 2.0)
        self.assertEqual(ref["lambda_ssim"], 1.0)
        self.assertEqual(original["num_images_per_spec"], 400)
        self.assertEqual(original["base_config"], str(run / "configs/base.json"))
        self.assertEqual(len(list((run / "ranking_cache").glob("*.csv"))), 5)
        self.assertEqual(configure.configure(self.settings), run)
        self.assertFalse((run / "original_ghost").exists())
        self.assertFalse((run / "stghost").exists())

    def test_existing_profile_cannot_be_reassigned_to_another_model(self):
        configure.configure(self.settings)
        changed = {**self.settings, "victim_model_path": "/different/model"}
        with self.assertRaisesRegex(ValueError, "different settings"):
            configure.configure(changed)

    def test_generated_launcher_forwards_arguments_and_handles_spaces(self):
        run = configure.configure(self.settings)
        result = subprocess.run(["bash", str(run / "run_st_ghost.sh"), "--validate-only"],
                                env={"STGHOST_PYTHON": "/bin/echo"}, text=True,
                                capture_output=True, check=True)
        self.assertIn(str(self.root / "st-ghost/run_stage3_st_ghost.py"), result.stdout)
        self.assertIn("--validate-only", result.stdout)
        self.assertIn(str(run / "stghost_reference"), result.stdout)

    def test_other_victim_architecture_is_explicitly_unsupported(self):
        p = self.root / "other.json"
        p.write_text(json.dumps({**self.settings, "victim_family": "llava"}))
        with self.assertRaisesRegex(ValueError, "Adapt"):
            configure.load_settings(p)


if __name__ == "__main__":
    unittest.main()
