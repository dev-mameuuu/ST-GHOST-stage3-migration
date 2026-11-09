"""CPU checks for ST-GHOST launch gating and the real shell worker."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "wait_stage3_st_ghost_gpu.py"
SPEC = importlib.util.spec_from_file_location("stage3_st_ghost_gpu_wait", SOURCE)
waiter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(waiter)


class LaunchControlTests(unittest.TestCase):
    def test_waits_at_or_below_46_gib_then_launches(self):
        args = waiter.parse_args([])
        self.assertEqual(args.min_free_gib, 46)
        with mock.patch.object(waiter, "job_status", return_value={"ActiveState": "inactive"}), \
                mock.patch.object(waiter, "target_pids", return_value=set()), \
                mock.patch.object(waiter, "gpu_memory", side_effect=[
                    (47103, 49140), (47104, 49140), (47105, 49140)]), \
                mock.patch.object(waiter, "confirm_start", return_value=True), \
                mock.patch.object(waiter, "systemctl") as systemctl, \
                mock.patch.object(waiter.time, "sleep") as sleep:
            waiter.monitor(args)
        self.assertEqual(systemctl.call_args_list, [
            mock.call("reset-failed", waiter.JOB_UNIT, check=False),
            mock.call("start", waiter.JOB_UNIT)])
        self.assertEqual(sleep.call_args_list, [mock.call(30), mock.call(30)])

    def test_existing_experiment_prevents_duplicate_launch(self):
        with mock.patch.object(waiter, "job_status", return_value={"ActiveState": "inactive"}), \
                mock.patch.object(waiter, "target_pids", side_effect=[{123}, set()]), \
                mock.patch.object(waiter, "gpu_memory", return_value=(48000, 49140)) as memory, \
                mock.patch.object(waiter, "confirm_start", return_value=True), \
                mock.patch.object(waiter, "systemctl") as systemctl, \
                mock.patch.object(waiter.time, "sleep") as sleep:
            waiter.monitor(waiter.parse_args([]))
        self.assertEqual(memory.call_count, 1)
        sleep.assert_called_once_with(30)
        systemctl.assert_any_call("start", waiter.JOB_UNIT)

    def test_installs_independent_units_with_matching_thresholds(self):
        args = waiter.parse_args(["--gpu-index", "2"])
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(waiter.Path, "home", return_value=Path(directory)), \
                mock.patch.object(waiter, "systemctl") as systemctl:
            waiter.install_units(args)
            units = Path(directory) / ".config/systemd/user"
            self.assertEqual({p.name for p in units.iterdir()}, {waiter.JOB_UNIT, waiter.WATCH_UNIT})
            job = (units / waiter.JOB_UNIT).read_text()
            watcher = (units / waiter.WATCH_UNIT).read_text()
        self.assertIn("run_stage3_st_ghost_service.sh", job)
        self.assertIn("STGHOST_MIN_FREE_MIB=47105", job)
        self.assertIn("CUDA_VISIBLE_DEVICES=2", job)
        self.assertIn("STGHOST_GPU_INDEX=2", job)
        self.assertIn('"--min-free-gib" "46.0"', watcher)
        self.assertIn("wait_stage3_st_ghost_gpu.py", watcher)
        self.assertNotIn("original-ghost-comparison", job + watcher)
        self.assertNotIn("PartOf=", job)
        self.assertFalse(waiter.paths()[1].is_relative_to(waiter.ROOT / "outputs"))
        self.assertTrue(waiter.protocol().startswith("stage3_st_ghost_"))
        systemctl.assert_called_once_with("daemon-reload")


class ShellLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "st-ghost").mkdir()
        self.wrapper = self.root / "st-ghost" / waiter.WRAPPER.name
        shutil.copyfile(waiter.WRAPPER, self.wrapper)
        self.fake_bin = self.root / "fake-bin"
        self.fake_bin.mkdir()
        self.env = {**os.environ, "PATH": f"{self.fake_bin}:{os.environ['PATH']}",
                    "STGHOST_STAGE3_SESSION": "stghost-stage3-st-ghost-test",
                    "STGHOST_STAGE3_STATE": str(self.root / "service"),
                    "STGHOST_STAGE3_LOG": str(self.root / "experiment.log"),
                    "STGHOST_GPU_INDEX": "2", "STGHOST_MIN_FREE_MIB": "47105"}
        self.env.pop("CUDA_VISIBLE_DEVICES", None)

    def executable(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\n" + text)
        path.chmod(0o755)

    def run_wrapper(self, *args):
        return subprocess.run(["bash", str(self.wrapper), *args], env=self.env,
                              capture_output=True, text=True, timeout=10)

    def test_worker_runs_full_experiment_and_propagates_failures(self):
        self.executable(self.root / ".venv-stage1/bin/python", '''printf '%s\\n' "$@" > "$STGHOST_TEST_ARGS"
printf '%s\\n' "$CUDA_VISIBLE_DEVICES" > "$STGHOST_TEST_GPU"
printf 'fake experiment output\\n'
exit "$STGHOST_TEST_EXIT"
''')
        self.env.update(STGHOST_TEST_ARGS=str(self.root / "args"),
                        STGHOST_TEST_GPU=str(self.root / "gpu"))
        for code in (0, 23):
            with self.subTest(exit_code=code):
                self.env["STGHOST_TEST_EXIT"] = str(code)
                result = self.run_wrapper("--screen-worker")
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual((self.root / "service/exit_status").read_text().strip(), str(code))
                self.assertEqual((self.root / "args").read_text().splitlines(), [
                    "-u", str(self.root / "st-ghost/run_stage3_st_ghost.py")])
                self.assertEqual((self.root / "gpu").read_text().strip(), "2")
                self.assertIn("fake experiment output", (self.root / "experiment.log").read_text())
                self.assertFalse((self.root / "outputs").exists())

    def test_supervisor_rechecks_memory_before_starting_screen(self):
        self.env["STGHOST_TEST_SCREEN_START"] = str(self.root / "screen-started")
        self.executable(self.fake_bin / "screen", '''if [[ "$1" == "-list" ]]; then exit 1; fi
touch "$STGHOST_TEST_SCREEN_START"
exit 1
''')
        self.executable(self.fake_bin / "nvidia-smi", '''printf '%s\\n' "$STGHOST_TEST_MEMORY"
exit "$STGHOST_TEST_QUERY_EXIT"
''')
        for memory, query_code in (("47104", "0"), ("46080", "0"), ("N/A", "0"), ("48000", "1")):
            with self.subTest(memory=memory, query_exit=query_code):
                self.env.update(STGHOST_TEST_MEMORY=memory, STGHOST_TEST_QUERY_EXIT=query_code)
                result = self.run_wrapper()
                self.assertEqual(result.returncode, 75, result.stderr)
                self.assertFalse((self.root / "screen-started").exists())
                self.assertFalse((self.root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
