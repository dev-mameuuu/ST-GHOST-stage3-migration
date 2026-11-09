"""CPU launch-control and real shell-worker tests; no models or services run."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "wait_stage3_original_ghost_comparison_gpu.py"
SPEC = importlib.util.spec_from_file_location("stage3_gpu_wait", SOURCE)
waiter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(waiter)


class Stage3LaunchControlTests(unittest.TestCase):
    def setUp(self):
        self.args = waiter.parse_args([])

    def test_failed_launch_rechecks_memory_before_retry(self):
        calls = []
        with mock.patch.object(waiter, "job_status", return_value={"ActiveState": "inactive"}), \
                mock.patch.object(waiter, "target_pids", return_value=set()), \
                mock.patch.object(waiter, "gpu_memory", side_effect=[
                    (46080, 49140), (48000, 49140), (46080, 49140), (48000, 49140)]), \
                mock.patch.object(waiter, "confirm_start", side_effect=[False, True]), \
                mock.patch.object(waiter, "systemctl", side_effect=lambda *a, **k: calls.append(a)), \
                mock.patch.object(waiter.time, "sleep") as sleep:
            waiter.monitor(self.args)
        self.assertEqual(calls, [
            ("reset-failed", waiter.JOB_UNIT), ("start", waiter.JOB_UNIT), ("stop", waiter.JOB_UNIT),
            ("reset-failed", waiter.JOB_UNIT), ("start", waiter.JOB_UNIT)])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [30, 60, 30])

    def test_existing_experiment_prevents_duplicate_launch(self):
        with mock.patch.object(waiter, "job_status", return_value={"ActiveState": "inactive"}), \
                mock.patch.object(waiter, "target_pids", side_effect=[{123}, set()]), \
                mock.patch.object(waiter, "gpu_memory", return_value=(48000, 49140)) as memory, \
                mock.patch.object(waiter, "confirm_start", return_value=True), \
                mock.patch.object(waiter, "systemctl") as systemctl, \
                mock.patch.object(waiter.time, "sleep") as sleep:
            waiter.monitor(self.args)
        self.assertEqual(memory.call_count, 1)
        sleep.assert_called_once_with(30)
        systemctl.assert_any_call("start", waiter.JOB_UNIT)

    def test_success_requires_stability_and_gpu_allocation(self):
        with mock.patch.object(waiter, "job_status", return_value={"ActiveState": "active", "ControlGroup": "/job"}), \
                mock.patch.object(waiter, "target_pids", return_value={123}), \
                mock.patch.object(waiter, "belongs_to_job", return_value=True), \
                mock.patch.object(waiter, "screen_alive", return_value=True), \
                mock.patch.object(waiter, "gpu_pids", side_effect=[set(), set(), {123}]), \
                mock.patch.object(waiter.time, "monotonic", side_effect=[0, 0, 0, 0, 130, 130, 130, 140, 140]), \
                mock.patch.object(waiter.time, "sleep") as sleep:
            self.assertTrue(waiter.confirm_start(self.args))
        self.assertEqual(sleep.call_count, 2)

    def test_units_run_stage3_independently_on_requested_gpu(self):
        self.args.gpu_index = 2
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(waiter.Path, "home", return_value=Path(directory)), \
                mock.patch.object(waiter, "systemctl") as systemctl:
            waiter.install_units(self.args)
            units = Path(directory) / ".config/systemd/user"
            self.assertEqual({p.name for p in units.iterdir()}, {waiter.JOB_UNIT, waiter.WATCH_UNIT})
            job = (units / waiter.JOB_UNIT).read_text()
            watcher = (units / waiter.WATCH_UNIT).read_text()
        self.assertIn(str(waiter.WRAPPER), job)
        self.assertIn("STGHOST_MIN_FREE_MIB=46081", job)
        self.assertIn("CUDA_VISIBLE_DEVICES=2", job)
        self.assertIn("STGHOST_GPU_INDEX=2", job)
        self.assertIn("Restart=no", job)
        self.assertIn("KillMode=control-group", job)
        self.assertNotIn("PartOf=", job)
        self.assertNotIn("BindsTo=", job)
        self.assertNotIn("--background", watcher)
        self.assertNotIn("stage2", job + watcher)
        self.assertFalse(waiter.paths()[1].is_relative_to(waiter.ROOT / "outputs"))
        systemctl.assert_called_once_with("daemon-reload")

    def test_read_only_check_does_not_install_or_start_services(self):
        with mock.patch.object(waiter, "gpu_memory", return_value=(46080, 49140)), \
                mock.patch.object(waiter, "target_pids", return_value=set()), \
                mock.patch.object(waiter, "install_units") as install, \
                mock.patch.object(waiter, "systemctl") as systemctl, \
                mock.patch("builtins.print"):
            waiter.main(["--check-only"])
        install.assert_not_called()
        systemctl.assert_not_called()

    def test_running_watcher_is_not_reconfigured(self):
        with mock.patch.object(waiter, "run", return_value=subprocess.CompletedProcess([], 0, "yes\n")), \
                mock.patch.object(waiter, "systemctl", return_value=subprocess.CompletedProcess([], 0)) as systemctl, \
                mock.patch.object(waiter, "install_units") as install, \
                mock.patch.object(waiter.logging, "basicConfig"), \
                mock.patch.object(waiter.logging, "FileHandler"):
            waiter.main(["--background", "--gpu-index", "2"])
        install.assert_not_called()
        systemctl.assert_called_once_with("is-active", "--quiet", waiter.WATCH_UNIT, check=False)


class Stage3ShellLauncherTests(unittest.TestCase):
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
                    "STGHOST_STAGE3_SESSION": "stghost-stage3-test",
                    "STGHOST_STAGE3_STATE": str(self.root / "service"),
                    "STGHOST_STAGE3_LOG": str(self.root / "experiment.log"),
                    "STGHOST_GPU_INDEX": "2", "STGHOST_MIN_FREE_MIB": "46081"}
        self.env.pop("CUDA_VISIBLE_DEVICES", None)

    def executable(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\n" + text)
        path.chmod(0o755)

    def run_wrapper(self, *args):
        return subprocess.run(["bash", str(self.wrapper), *args], env=self.env,
                              capture_output=True, text=True, timeout=10)

    def test_worker_runs_full_experiment_and_preserves_python_exit_status(self):
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
                    "-u", str(self.root / "st-ghost" / waiter.TARGET.name), "--config",
                    str(self.root / "st-ghost/configs/stage3_original_ghost_comparison.json")])
                self.assertEqual((self.root / "gpu").read_text().strip(), "2")
                self.assertTrue((self.root / "service/finished_at").is_file())
                self.assertIn("fake experiment output", (self.root / "experiment.log").read_text())
                self.assertFalse((self.root / "outputs").exists())

    def test_supervisor_defers_for_insufficient_memory_or_query_failure(self):
        self.env["STGHOST_TEST_SCREEN_START"] = str(self.root / "screen-started")
        self.executable(self.fake_bin / "screen", '''if [[ "$1" == "-list" ]]; then exit 1; fi
touch "$STGHOST_TEST_SCREEN_START"
exit 1
''')
        self.executable(self.fake_bin / "nvidia-smi", '''printf '%s\\n' "$STGHOST_TEST_MEMORY"
exit "$STGHOST_TEST_QUERY_EXIT"
''')
        for memory, query_code in (("46080", "0"), ("45000", "0"), ("N/A", "0"), ("48000", "1")):
            with self.subTest(memory=memory, query_exit=query_code):
                self.env.update(STGHOST_TEST_MEMORY=memory, STGHOST_TEST_QUERY_EXIT=query_code)
                result = self.run_wrapper()
                self.assertEqual(result.returncode, 75, result.stderr)
                self.assertFalse((self.root / "screen-started").exists())
                self.assertFalse((self.root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
