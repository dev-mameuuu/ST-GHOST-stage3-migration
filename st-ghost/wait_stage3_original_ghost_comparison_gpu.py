#!/usr/bin/env python3
"""Wait for free GPU memory, then launch the full Stage 3 GHOST comparison.

Uses the existing systemd + screen launch protocol and only the Python standard
library. The experiment runs with configs/stage3_original_ghost_comparison.json;
no validation-only, preparation-only or task-limit flags are passed.
Run with --background to start a persistent systemd user watcher, or --check-only
for a read-only readiness query. The watcher exits after confirming a stable GPU
process; the independent experiment continues running after SSH disconnects.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "st-ghost/run_stage3_original_ghost_comparison.py"
WRAPPER = TARGET.with_name(TARGET.stem + "_service.sh")
PYTHON = ROOT / ".venv-stage1/bin/python"
JOB_UNIT = "st-ghost-stage3-original-ghost-comparison-gpu-job.service"
WATCH_UNIT = "st-ghost-stage3-original-ghost-comparison-gpu-wait.service"
LOG_PATH = ROOT / "logs/stghost-stage3-original-ghost-comparison-gpu-wait.log"
LOG = logging.getLogger("stage3-gpu-wait")


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True, check=check, timeout=30)


def systemctl(*args: str, check: bool = True):
    return run(["systemctl", "--user", *args], check=check)


def quote_unit(value: object) -> str:
    # systemd syntax is not shell syntax. Escape its specifiers and C escapes.
    text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("\n", "\\n") + '"'


def protocol() -> str:
    for node in ast.parse(TARGET.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(name, ast.Name) and name.id == "PROTOCOL" for name in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_]+", value):
                return value
    raise RuntimeError("Cannot read PROTOCOL from the experiment script")


def paths() -> tuple[str, Path, Path]:
    # Stage 3 rejects nonempty outputs without an experiment signature. Keep
    # launcher state outside that directory so the first run can initialize it.
    session = "stghost-stage3-original-ghost-comparison"
    return (session, ROOT / "logs" / f"{session}-service",
            ROOT / "logs" / f"{session}.log")


def enough_memory(free_mib: int, minimum_gib: float) -> bool:
    return free_mib > minimum_gib * 1024  # Strictly greater, as requested.


def gpu_memory(index: int) -> tuple[int, int]:
    line = run(["nvidia-smi", "-i", str(index), "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits"]).stdout.strip()
    free, total = (int(part.strip()) for part in line.split(","))
    return free, total


def gpu_pids(index: int) -> set[int]:
    text = run(["nvidia-smi", "-i", str(index), "--query-compute-apps=pid",
                "--format=csv,noheader,nounits"]).stdout
    return {int(line.strip()) for line in text.splitlines() if line.strip().isdigit()}


def target_pids() -> set[int]:
    found = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not argv or not Path(argv[0]).name.startswith("python"):
                continue
            for arg in argv[1:]:
                if Path(arg).name != TARGET.name:
                    continue
                cwd = (proc / "cwd").resolve()
                if (cwd / arg).resolve() == TARGET:
                    found.add(int(proc.name))
        except (OSError, ValueError):
            continue  # A process may exit while /proc is being read.
    return found


def job_status() -> dict[str, str]:
    result = systemctl("show", JOB_UNIT, "-p", "ActiveState", "-p", "SubState",
                       "-p", "ControlGroup", "-p", "Result", check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Cannot query the experiment service")
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def belongs_to_job(pid: int, control_group: str) -> bool:
    if not control_group:
        return False
    try:
        return any((path := line.split(":", 2)[-1]) == control_group or path.startswith(control_group + "/")
                   for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines())
    except OSError:
        return False


def screen_alive(session: str) -> bool:
    result = run(["screen", "-ls"], check=False)
    return re.search(r"\d+\." + re.escape(session) + r"\s", result.stdout) is not None


def install_units(args: argparse.Namespace) -> None:
    session, state, log = paths()
    minimum_mib = math.floor(args.min_free_gib * 1024) + 1
    environment = {
        "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
        "STGHOST_GPU_INDEX": str(args.gpu_index),
        "STGHOST_MIN_FREE_MIB": str(minimum_mib),
        "STGHOST_STAGE3_SESSION": session,
        "STGHOST_STAGE3_STATE": str(state),
        "STGHOST_STAGE3_LOG": str(log),
    }
    job = "\n".join([
        "[Unit]", "Description=ST-GHOST GPU-gated Stage 3 original GHOST comparison", "",
        "[Service]", "Type=simple", f"WorkingDirectory={str(ROOT).replace('%', '%%')}",
        *(f"Environment={quote_unit(key + '=' + value)}" for key, value in environment.items()),
        f"ExecStart=/bin/bash {quote_unit(WRAPPER)}",
        f"ExecStop=-/usr/bin/screen -S {quote_unit(session)} -X quit",
        "KillMode=control-group", "Restart=no", "TimeoutStopSec=30s", "",
    ])
    command = [str(PYTHON), str(Path(__file__).resolve()),
               "--gpu-index", str(args.gpu_index), "--min-free-gib", str(args.min_free_gib),
               "--poll-seconds", str(args.poll_seconds), "--retry-seconds", str(args.retry_seconds),
               "--startup-seconds", str(args.startup_seconds),
               "--startup-timeout", str(args.startup_timeout)]
    watcher = "\n".join([
        "[Unit]", "Description=ST-GHOST wait for GPU for Stage 3 original GHOST comparison", "",
        "[Service]", "Type=simple", f"WorkingDirectory={str(ROOT).replace('%', '%%')}",
        "ExecStart=" + " ".join(quote_unit(arg) for arg in command),
        "Environment=PYTHONUNBUFFERED=1", "Restart=on-failure", "RestartSec=30s",
        "TimeoutStopSec=15s", "", "[Install]", "WantedBy=default.target", "",
    ])
    directory = Path.home() / ".config/systemd/user"
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in ((JOB_UNIT, job), (WATCH_UNIT, watcher)):
        file = directory / name
        if not file.exists() or file.read_text() != content:
            temporary = file.with_suffix(".service.tmp")
            temporary.write_text(content)
            temporary.replace(file)
    systemctl("daemon-reload")


def confirm_start(args: argparse.Namespace) -> bool:
    session, state, _ = paths()
    deadline = time.monotonic() + args.startup_timeout
    stable_since = None
    stable_pids: set[int] = set()
    while time.monotonic() < deadline:
        status = job_status()
        if status.get("ActiveState") not in {"active", "activating"}:
            # A very short, fully resumed run can finish before the health window.
            exit_status = state / "exit_status"
            if status.get("Result") == "success" and exit_status.exists() and exit_status.read_text().strip() == "0":
                LOG.info("实验已正常完成；监控退出。")
                return True
            LOG.warning("启动未成功：服务状态=%s/%s，Result=%s", status.get("ActiveState"),
                        status.get("SubState"), status.get("Result"))
            return False
        pids = {pid for pid in target_pids() if belongs_to_job(pid, status.get("ControlGroup", ""))}
        now = time.monotonic()
        if pids and screen_alive(session):
            if stable_since is None or pids != stable_pids:
                stable_since, stable_pids = now, pids
            elapsed = now - stable_since
            allocated = pids & gpu_pids(args.gpu_index)
            LOG.info("启动检查：PID=%s，screen正常，已存活%.0f秒，GPU进程=%s", sorted(pids), elapsed, sorted(allocated))
            if elapsed >= args.startup_seconds and allocated:
                LOG.info("启动成功：已稳定运行至少%.0f秒；实验由独立user服务和screen继续执行，监控退出。", args.startup_seconds)
                return True
        else:
            stable_since, stable_pids = None, set()
            LOG.info("启动检查：等待实验进程和screen就绪。")
        time.sleep(min(10, max(0.1, deadline - time.monotonic())))
    LOG.warning("启动检查超过%.0f秒，未确认稳定的GPU实验进程。", args.startup_timeout)
    return False


def monitor(args: argparse.Namespace) -> None:
    attempt = 0
    while True:
        try:
            status = job_status()
            if status.get("ActiveState") in {"active", "activating"}:
                if confirm_start(args):
                    return
                systemctl("stop", JOB_UNIT)
                time.sleep(args.retry_seconds)
                continue
            duplicates = target_pids()
            if duplicates:
                LOG.info("已存在其他会话的同脚本进程%s，等待，避免重复启动。", sorted(duplicates))
                time.sleep(args.poll_seconds)
                continue
            free, total = gpu_memory(args.gpu_index)
            if total <= args.min_free_gib * 1024:
                raise ValueError(f"GPU总显存{total / 1024:.2f}GiB，无法满足严格大于{args.min_free_gib}GiB")
            LOG.info("GPU %d空闲%.2f/%.2fGiB；启动条件：严格大于%.2fGiB。", args.gpu_index, free / 1024, total / 1024, args.min_free_gib)
            if not enough_memory(free, args.min_free_gib):
                time.sleep(args.poll_seconds)
                continue
            attempt += 1
            LOG.info("显存满足条件，开始第%d次启动。", attempt)
            systemctl("reset-failed", JOB_UNIT, check=False)
            systemctl("start", JOB_UNIT)
            if confirm_start(args):
                return
            systemctl("stop", JOB_UNIT)
        except (subprocess.SubprocessError, OSError, RuntimeError, ValueError) as error:
            LOG.warning("查询或启动失败：%s；%.0f秒后重新检查。", error, args.retry_seconds)
        time.sleep(args.retry_seconds)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--min-free-gib", type=float, default=45.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    parser.add_argument("--startup-seconds", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--background", action="store_true", help="Install and start a systemd user watcher; safe to disconnect afterward")
    mode.add_argument("--check-only", action="store_true", help="Read-only readiness query; do not install or start anything")
    args = parser.parse_args(argv)
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    for name in ("min_free_gib", "poll_seconds", "retry_seconds", "startup_seconds", "startup_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.startup_timeout <= args.startup_seconds:
        parser.error("--startup-timeout must be greater than --startup-seconds")
    return args


def main(argv=None):
    args = parse_args(argv)
    for command in ("nvidia-smi", "systemctl", "screen", "loginctl"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Required command not found: {command}")
    if not PYTHON.is_file() or not TARGET.is_file() or not WRAPPER.is_file():
        raise RuntimeError("Missing experiment Python environment, target script or service wrapper")
    if args.check_only:
        free, total = gpu_memory(args.gpu_index)
        print(f"GPU {args.gpu_index}: free={free / 1024:.2f}GiB, total={total / 1024:.2f}GiB; "
              f"free>{args.min_free_gib}GiB: {enough_memory(free, args.min_free_gib)}")
        print(f"Target: {TARGET}\nProtocol: {protocol()}\nScreen: {paths()[0]}\nTarget PIDs: {sorted(target_pids())}")
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)])
    linger = run(["loginctl", "show-user", str(os.getuid()), "-p", "Linger", "--value"]).stdout.strip()
    if linger != "yes":
        raise RuntimeError("Linger is disabled; enable it with loginctl enable-linger before background monitoring")
    if args.background:
        # Do not rewrite or replace a running watcher with different settings.
        if systemctl("is-active", "--quiet", WATCH_UNIT, check=False).returncode == 0:
            LOG.info("监控服务已在运行；本次未更改其配置。")
            return
        install_units(args)
        systemctl("reset-failed", WATCH_UNIT, check=False)
        systemctl("start", WATCH_UNIT)
        LOG.info("后台监控已启动：%s；不会随SSH断开退出。日志：%s", WATCH_UNIT, LOG_PATH)
        return
    with LOG_PATH.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.info("已有显存监控进程；本次退出，避免重复启动。")
            return
        install_units(args)
        LOG.info("开始监控GPU %d；每%.0f秒检查，启动失败等待%.0f秒后重试。", args.gpu_index, args.poll_seconds, args.retry_seconds)
        monitor(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("显存监控已停止；独立实验服务不受影响。")
    except Exception:
        LOG.exception("监控出现错误")
        sys.exit(1)
