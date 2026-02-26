"""
GPU Orchestrator for Dynamic Model Allocation.

Manages vLLM servers with sleep mode to dynamically allocate GPUs across pipeline phases.

With sufficient RAM (768GB+), all models are pre-loaded in sleeping state and woken as needed,
enabling ~6-12 second phase transitions instead of 60-120 second disk reloads.

Modes:
- "task_dp2": Task model on both GPU pairs (DP=2 for ~2x throughput)
- "dual_model": Task on GPUs 0,1 + GenRM on GPUs 2,3

Usage:
    orchestrator = GPUOrchestrator(config)
    await orchestrator.initialize()  # Start all servers, sleep secondary ones

    # Phase 1: Document processing with DP=2
    await orchestrator.enter_task_dp2_mode()
    ports = orchestrator.get_active_task_ports()  # [8000, 8002]

    # Phase 1.5: Need GenRM
    await orchestrator.enter_dual_model_mode()

    # Cleanup
    await orchestrator.shutdown()
"""

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from functools import lru_cache
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import yaml

from src.core.vllm_runtime import resolve_vllm_runtime_flags

logger = logging.getLogger(__name__)


def _looks_like_cuda_oom(message: str) -> bool:
    """Heuristic check for CUDA OOM startup failures in vLLM logs/errors."""
    text = (message or "").lower()
    markers = (
        "cuda out of memory",
        "torch.outofmemoryerror",
        "outofmemoryerror",
        "c10::outofmemoryerror",
        "tried to allocate",
    )
    return any(marker in text for marker in markers)


def _listener_pids_on_port(port: int) -> List[int]:
    """Return PIDs for LISTEN sockets bound to the given TCP port."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-t", f"-iTCP:{int(port)}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            pids: List[int] = []
            for line in result.stdout.splitlines():
                try:
                    pids.append(int(line.strip()))
                except (TypeError, ValueError):
                    continue
            return sorted(set(pids))
        return []
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Error checking port %d listener via lsof: %s", int(port), exc)

    try:
        result = subprocess.run(
            ["ss", "-ltnp", f"sport = :{int(port)}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []

        pids: List[int] = []
        for line in result.stdout.splitlines():
            for token in line.split():
                if "pid=" not in token:
                    continue
                try:
                    pid_part = token.split("pid=", 1)[1]
                    pid_text = pid_part.split(",", 1)[0].rstrip(")")
                    pids.append(int(pid_text))
                except Exception:
                    continue
        return sorted(set(pids))
    except FileNotFoundError:
        logger.warning("Neither lsof nor ss available to check listener on port %d", int(port))
    except Exception as exc:
        logger.warning("Error checking port %d listener via ss: %s", int(port), exc)
    return []


def kill_process_on_port(port: int) -> bool:
    """Kill listener processes bound to the given TCP port.

    Returns True if a process was killed, False if no process was found.
    """
    pids = _listener_pids_on_port(int(port))
    if not pids:
        return False

    for pid_int in pids:
        try:
            logger.info("Killing existing process %d on port %d", int(pid_int), int(port))
            os.kill(int(pid_int), signal.SIGTERM)
        except (ValueError, ProcessLookupError):
            continue
        except Exception as exc:
            logger.warning("Failed to SIGTERM pid=%d on port %d: %s", int(pid_int), int(port), exc)

    # Wait a moment for cleanup
    time.sleep(1.0)
    return True


def _prepend_env_path(env: Dict[str, str], key: str, value: str) -> None:
    if not value:
        return
    current = env.get(key, "")
    parts = [part for part in current.split(":") if part]
    if value in parts:
        return
    env[key] = f"{value}:{current}" if current else value


@lru_cache(maxsize=8)
def _resolve_venv_site_packages(venv_path: str) -> Optional[Path]:
    root = Path(venv_path)
    candidates: List[Path] = []
    for base in (root / "lib", root / "local" / "lib"):
        if not base.is_dir():
            continue
        candidates.extend(base.glob("python*/site-packages"))
        candidates.extend(base.glob("python*/dist-packages"))
    for candidate in sorted(candidates, reverse=True):
        if candidate.is_dir():
            return candidate
    return None


def _configure_nvfp4_runtime_env(env: Dict[str, str], venv_path: str, profile: str) -> None:
    """
    Configure the environment so NVFP4/FlashInfer JIT kernels can load.

    When vLLM is launched programmatically (vs. `scripts/start_vllm.sh`), we
    won't necessarily have CUDA_HOME/LD_LIBRARY_PATH pointing at the pip-
    installed CUDA toolchain inside `venv_path`. FlashInfer's TVM-loaded shared
    objects depend on e.g. `libcudart.so.13`, so we must add those directories
    to the dynamic linker path.
    """
    if "nvfp4" not in str(profile).lower():
        return

    env.setdefault("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    env.setdefault("VLLM_FLASHINFER_MOE_BACKEND", "throughput")

    site_packages = _resolve_venv_site_packages(venv_path)
    if site_packages is None:
        logger.warning(
            "Could not locate site-packages for vLLM venv (%s); NVFP4 runtime may fail",
            venv_path,
        )
        return

    cu13_root = site_packages / "nvidia" / "cu13"
    if cu13_root.is_dir():
        if not shutil.which("nvcc") and (cu13_root / "bin" / "nvcc").is_file():
            env.setdefault("CUDA_HOME", str(cu13_root))

        cuda_home = env.get("CUDA_HOME")
        if cuda_home:
            _prepend_env_path(env, "PATH", str(Path(cuda_home) / "bin"))

        for lib_dir in (cu13_root / "lib64", cu13_root / "lib"):
            if lib_dir.is_dir():
                _prepend_env_path(env, "LD_LIBRARY_PATH", str(lib_dir))

    if Path("/lib/x86_64-linux-gnu").is_dir():
        _prepend_env_path(env, "LD_LIBRARY_PATH", "/lib/x86_64-linux-gnu")

    curand_include = site_packages / "nvidia" / "curand" / "include"
    if curand_include.is_dir():
        _prepend_env_path(env, "CPATH", str(curand_include))


class OrchestratorMode(Enum):
    """Current GPU allocation mode."""
    TASK_DP2 = "task_dp2"       # Task on both GPU pairs (DP=2)
    DUAL_MODEL = "dual_model"   # Task + GenRM on separate GPU pairs
    UNINITIALIZED = "uninitialized"


@dataclass
class ServerConfig:
    """Configuration for a single vLLM server."""
    profile: str
    port: int
    cuda_devices: str
    tensor_parallel: int
    max_model_len: int = 32768
    runtime_args: List[str] = field(default_factory=list)
    enable_prefix_caching: bool = False
    enable_sleep_mode: bool = True
    startup_timeout: float = 300.0
    gpu_memory_utilization: float = 0.90


@dataclass
class OrchestratorConfig:
    """Configuration for the GPU orchestrator."""
    # Server configs
    task_primary: ServerConfig = field(default_factory=lambda: ServerConfig(
        profile="nemotron-30b-nvfp4",
        port=8000,
        cuda_devices="0,1",
        tensor_parallel=2,
        max_model_len=32768,
    ))
    task_replica: ServerConfig = field(default_factory=lambda: ServerConfig(
        profile="nemotron-30b-nvfp4",
        port=8002,
        cuda_devices="2,3",
        tensor_parallel=2,
        max_model_len=32768,
    ))
    genrm: ServerConfig = field(default_factory=lambda: ServerConfig(
        profile="genrm-nvfp4",
        port=8001,
        cuda_devices="2,3",
        tensor_parallel=2,
        max_model_len=32768,
        gpu_memory_utilization=0.95,
    ))

    # Paths
    venv_path: str = "/home/mlinegar/vllm-env"
    config_path: Optional[Path] = None

    # Timeouts
    sleep_timeout: float = 30.0
    wake_timeout: float = 60.0
    health_check_interval: float = 2.0
    post_stop_settle_seconds: float = 6.0
    # Stability-first toggle for servers that share GPUs (task_replica/genrm).
    # When enabled, mode transitions stop the peer process instead of relying
    # on vLLM sleep mode, which avoids frequent wake/start OOM failures.
    shared_gpu_hard_quiesce: bool = False

    @classmethod
    def from_yaml(cls, config_path: Path) -> "OrchestratorConfig":
        """Load orchestrator config from settings.yaml."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        orch_cfg = cfg.get("orchestration", {})
        vllm_cfg = cfg.get("vllm", {})
        models = vllm_cfg.get("models", {})

        # Get model profiles
        task_profile = orch_cfg.get("task_model_profile", vllm_cfg.get("default", "nemotron-30b-nvfp4"))
        genrm_profile = orch_cfg.get("genrm_profile", "genrm-nvfp4")

        # Get model configs
        task_model = models.get(task_profile, {})
        genrm_model = models.get(genrm_profile, {})
        task_runtime_args = resolve_vllm_runtime_flags(
            vllm_cfg=vllm_cfg,
            profile=str(task_profile),
        ).to_cli_args()
        genrm_runtime_args = resolve_vllm_runtime_flags(
            vllm_cfg=vllm_cfg,
            profile=str(genrm_profile),
        ).to_cli_args()
        prefix_cache = bool(vllm_cfg.get("enable_prefix_caching", False))

        return cls(
            task_primary=ServerConfig(
                profile=task_profile,
                port=orch_cfg.get("task_primary_port", 8000),
                cuda_devices=orch_cfg.get("task_primary_gpus", "0,1"),
                tensor_parallel=task_model.get("tensor_parallel", 2),
                max_model_len=task_model.get("max_model_len", 32768),
                runtime_args=list(task_runtime_args),
                enable_prefix_caching=prefix_cache,
                gpu_memory_utilization=orch_cfg.get(
                    "task_primary_gpu_memory_utilization",
                    vllm_cfg.get("gpu_memory_utilization", 0.90),
                ),
            ),
            task_replica=ServerConfig(
                profile=task_profile,
                port=orch_cfg.get("task_replica_port", 8002),
                cuda_devices=orch_cfg.get("task_replica_gpus", "2,3"),
                tensor_parallel=task_model.get("tensor_parallel", 2),
                max_model_len=task_model.get("max_model_len", 32768),
                runtime_args=list(task_runtime_args),
                enable_prefix_caching=prefix_cache,
                gpu_memory_utilization=orch_cfg.get(
                    "task_replica_gpu_memory_utilization",
                    min(float(vllm_cfg.get("gpu_memory_utilization", 0.90)), 0.88),
                ),
            ),
            genrm=ServerConfig(
                profile=genrm_profile,
                port=orch_cfg.get("genrm_port", 8001),
                cuda_devices=orch_cfg.get("genrm_gpus", "2,3"),
                tensor_parallel=genrm_model.get("tensor_parallel", 2),
                max_model_len=genrm_model.get("max_model_len", 32768),
                runtime_args=list(genrm_runtime_args),
                enable_prefix_caching=prefix_cache,
                gpu_memory_utilization=orch_cfg.get("genrm_gpu_memory_utilization", 0.95),
            ),
            venv_path=orch_cfg.get("venv_path", "/home/mlinegar/vllm-env"),
            config_path=config_path,
            sleep_timeout=orch_cfg.get("sleep_timeout", 30.0),
            wake_timeout=orch_cfg.get("wake_timeout", 60.0),
            post_stop_settle_seconds=orch_cfg.get("post_stop_settle_seconds", 6.0),
            shared_gpu_hard_quiesce=bool(orch_cfg.get("shared_gpu_hard_quiesce", False)),
        )


class ManagedServer:
    """Manages a single vLLM server process with sleep mode support."""

    def __init__(
        self,
        config: ServerConfig,
        venv_path: str,
        model_path: str,
        health_check_interval: float = 2.0,
        post_stop_settle_seconds: float = 6.0,
    ):
        self.config = config
        self.venv_path = venv_path
        self.model_path = model_path
        self.health_check_interval = health_check_interval
        self.post_stop_settle_seconds = max(0.0, float(post_stop_settle_seconds))

        self._process: Optional[subprocess.Popen] = None
        self._log_file = None
        self._is_sleeping = False
        self._attached_pids: List[int] = []

    @property
    def port(self) -> int:
        return self.config.port

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def is_running(self) -> bool:
        if self._process is not None and self._process.poll() is None:
            return True
        return bool(_listener_pids_on_port(self.port))

    @property
    def is_sleeping(self) -> bool:
        return self._is_sleeping

    async def start(self) -> None:
        """Start the vLLM server with sleep mode enabled."""
        if self._process is not None and self._process.poll() is None:
            logger.warning("Server on port %d already running (owned process)", int(self.port))
            return

        # If something is already listening on the port, attempt to attach instead of killing.
        listener_pids = _listener_pids_on_port(self.port)
        if listener_pids:
            attached = await self._maybe_attach_existing(listener_pids)
            if attached:
                return

        # Kill any existing process on this port (from previous runs or incompatible servers)
        if kill_process_on_port(self.port):
            logger.info(f"Killed stale server on port {self.port}")
            # Wait a bit more for GPU memory to be released
            await asyncio.sleep(2)

        self._attached_pids = []
        logger.info(f"Starting vLLM server on port {self.port}")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  CUDA devices: {self.config.cuda_devices}")
        logger.info(f"  Tensor parallel: {self.config.tensor_parallel}")
        logger.info(f"  Max model len: {self.config.max_model_len}")
        logger.info(f"  Prefix cache: {self.config.enable_prefix_caching}")
        logger.info(f"  Sleep mode: {self.config.enable_sleep_mode}")
        if self.config.runtime_args:
            logger.info(f"  Runtime args: {' '.join(self.config.runtime_args)}")

        base_gmu = float(self.config.gpu_memory_utilization)
        effective_gmu = base_gmu
        attempted_oom_recovery = False

        while True:
            python_path = os.path.join(self.venv_path, "bin", "python")
            cmd = [
                python_path,
                "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.model_path,
                "--host", "0.0.0.0",
                "--port", str(self.port),
                "--tensor-parallel-size", str(self.config.tensor_parallel),
                "--max-model-len", str(self.config.max_model_len),
                "--gpu-memory-utilization", str(effective_gmu),
                "--trust-remote-code",
            ]

            if self.config.enable_prefix_caching:
                cmd.append("--enable-prefix-caching")

            if self.config.runtime_args:
                cmd.extend(self.config.runtime_args)

            cmd.append("--disable-log-requests")

            if self.config.enable_sleep_mode:
                cmd.append("--enable-sleep-mode")

            # Environment with CUDA device isolation and dev mode for sleep endpoints
            env = os.environ.copy()
            # Ensure vLLM venv binaries (e.g., `ninja`) are available even when this
            # orchestrator is invoked from a different Python environment.
            _prepend_env_path(env, "PATH", str(Path(self.venv_path) / "bin"))
            env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_devices
            if self.config.enable_sleep_mode:
                env["VLLM_SERVER_DEV_MODE"] = "1"
            _configure_nvfp4_runtime_env(env, venv_path=self.venv_path, profile=self.config.profile)

            # Create log file
            self._log_file = tempfile.NamedTemporaryFile(
                mode='w',
                prefix=f'vllm_port{self.port}_',
                suffix='.log',
                delete=False,
            )
            logger.info("  Log file: %s", self._log_file.name)
            logger.info("  Effective gpu_memory_utilization: %.2f", effective_gmu)

            # Start process
            self._process = subprocess.Popen(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                env=env,
            )

            logger.info(f"Server process started (PID: {self._process.pid})")

            try:
                # Wait for ready
                await self._wait_for_ready()
                self._is_sleeping = False
                return
            except Exception as exc:
                err_text = str(exc)
                self.stop()
                if attempted_oom_recovery or not _looks_like_cuda_oom(err_text):
                    raise

                attempted_oom_recovery = True
                reduced_gmu = max(0.80, round(effective_gmu - 0.03, 2))
                logger.warning(
                    "Detected CUDA OOM while starting port %d. "
                    "Performing quick clear and retrying once%s.",
                    int(self.port),
                    f" (gmu {effective_gmu:.2f} -> {reduced_gmu:.2f})"
                    if reduced_gmu < effective_gmu
                    else "",
                )

                kill_process_on_port(self.port)
                await asyncio.sleep(max(2.0, self.post_stop_settle_seconds))
                effective_gmu = reduced_gmu

    def _model_ids_match(self, served_model_ids: List[str]) -> bool:
        expected = str(self.model_path).rstrip("/")
        expected_base = os.path.basename(expected)
        for raw in served_model_ids:
            mid = str(raw).rstrip("/")
            if mid == expected:
                return True
            if os.path.basename(mid) == expected_base:
                return True
        return False

    async def _fetch_served_model_ids(self, timeout: float = 5.0) -> Optional[List[str]]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.url}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=float(timeout)),
                ) as resp:
                    if resp.status != 200:
                        return None
                    payload = await resp.json()
        except Exception:
            return None

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return None
        model_ids: List[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            mid = item.get("id")
            if mid is None:
                continue
            model_ids.append(str(mid))
        return model_ids or None

    async def _maybe_attach_existing(self, listener_pids: List[int]) -> bool:
        served_ids: Optional[List[str]] = None
        for attempt in range(1, 6):
            served_ids = await self._fetch_served_model_ids(timeout=2.0)
            if served_ids:
                break
            # If a vLLM server is still starting, the port can begin listening
            # before /v1/models is ready. Give it a short grace period so
            # sequential CV folds can attach without churn.
            await asyncio.sleep(min(1.0 * attempt, 2.0))

        served_ids = served_ids or None
        if not served_ids:
            logger.info(
                "Port %d has listener pids=%s but /v1/models is not ready; treating as stale",
                int(self.port),
                listener_pids,
            )
            return False

        if not self._model_ids_match(served_ids):
            logger.info(
                "Port %d already serving %s (expected %s); restarting",
                int(self.port),
                served_ids,
                str(self.model_path),
            )
            return False

        if self.config.enable_sleep_mode:
            sleep_state = await self.is_server_sleeping()
            if sleep_state is None:
                logger.info(
                    "Port %d serves expected model but sleep endpoints unavailable; restarting with sleep mode",
                    int(self.port),
                )
                return False

            self._is_sleeping = bool(sleep_state)
        else:
            self._is_sleeping = False

        self._attached_pids = list(listener_pids)
        logger.info(
            "Attached to existing vLLM server on port %d (pids=%s, sleeping=%s)",
            int(self.port),
            self._attached_pids,
            self._is_sleeping,
        )
        return True

    async def _wait_for_ready(self) -> None:
        """Wait for server to be ready."""
        start_time = time.time()
        timeout = self.config.startup_timeout

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                if self._process.poll() is not None:
                    # Process died
                    try:
                        self._log_file.flush()
                        with open(self._log_file.name, 'r') as f:
                            output = f.read()
                    except Exception:
                        output = "Could not read log file"
                    raise RuntimeError(
                        f"vLLM server on port {self.port} exited with code {self._process.returncode}. "
                        f"Output:\n{output[-2000:]}"
                    )

                try:
                    async with session.get(
                        f"{self.url}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            elapsed = time.time() - start_time
                            logger.info(f"Server on port {self.port} ready in {elapsed:.1f}s")
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass

                await asyncio.sleep(self.health_check_interval)

        self.stop()
        raise TimeoutError(f"Server on port {self.port} did not start within {timeout}s")

    def stop(self) -> None:
        """Stop the server."""
        if self._process is None:
            killed = kill_process_on_port(self.port)
            if killed:
                logger.info("Stopped external server on port %d", int(self.port))
            self._attached_pids = []
            self._is_sleeping = False
            return

        logger.info(f"Stopping server on port {self.port} (PID: {self._process.pid})")

        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
                logger.info(f"Server on port {self.port} stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"Server on port {self.port} did not stop gracefully, forcing kill")
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait()
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.warning(f"Error stopping server on port {self.port}: {e}")

        self._process = None
        self._is_sleeping = False

        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    async def sleep(self, level: int = 1, timeout: float = 30.0) -> bool:
        """Put server to sleep (offload weights to CPU RAM)."""
        if not self.is_running:
            logger.warning(f"Cannot sleep server on port {self.port}: not running")
            return False

        if self._is_sleeping:
            # Avoid trusting local state when wake/sleep errors happened earlier.
            actual = await self.is_server_sleeping()
            if actual is True:
                logger.debug(f"Server on port {self.port} already sleeping")
                return True
            if actual is False:
                logger.warning(
                    "Server on port %d local sleep flag stale (server reports awake); retrying sleep",
                    self.port,
                )
            else:
                logger.warning(
                    "Server on port %d local sleep flag stale/unverifiable; retrying sleep",
                    self.port,
                )
            self._is_sleeping = False

        logger.info(f"Putting server on port {self.port} to sleep (level={level})...")
        start = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.url}/sleep?level={level}",
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 200:
                        elapsed = time.time() - start
                        logger.info(f"Server on port {self.port} sleeping in {elapsed:.1f}s")
                        self._is_sleeping = True
                        return True
                    else:
                        text = await resp.text()
                        logger.error(f"Sleep request failed: {resp.status} - {text}")
                        self._is_sleeping = False
                        return False
        except Exception as e:
            logger.error(f"Failed to sleep server on port {self.port}: {e}")
            self._is_sleeping = False
            return False

    async def wake(self, timeout: float = 60.0) -> bool:
        """Wake server from sleep (reload weights from CPU RAM)."""
        if not self.is_running:
            logger.warning(f"Cannot wake server on port {self.port}: not running")
            return False

        if not self._is_sleeping:
            # Local state can become stale after wake/sleep endpoint errors.
            actual = await self.is_server_sleeping()
            if actual is True:
                logger.warning(
                    "Server on port %d local sleep flag stale (server reports sleeping); issuing wake request",
                    self.port,
                )
                self._is_sleeping = True
            else:
                logger.debug(f"Server on port {self.port} already awake")
                return True

        logger.info(f"Waking server on port {self.port}...")
        start = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.url}/wake_up",
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 200:
                        elapsed = time.time() - start
                        logger.info(f"Server on port {self.port} awake in {elapsed:.1f}s")
                        self._is_sleeping = False
                        return True
                    else:
                        text = await resp.text()
                        logger.error(f"Wake request failed: {resp.status} - {text}")
                        # Wake failure may still leave the server awake.
                        # Confirm sleep status before forcing expensive restart.
                        actual = await self.is_server_sleeping()
                        if actual is False:
                            logger.warning(
                                "Wake request failed on port %d but server reports awake; continuing without restart",
                                self.port,
                            )
                            self._is_sleeping = False
                            return True
                        # Keep conservative failure path when server still reports sleeping
                        # or state cannot be verified.
                        self._is_sleeping = False
                        return False
        except Exception as e:
            logger.error(f"Failed to wake server on port {self.port}: {e}")
            actual = await self.is_server_sleeping()
            if actual is False:
                logger.warning(
                    "Wake request errored on port %d but server reports awake; continuing without restart",
                    self.port,
                )
                self._is_sleeping = False
                return True
            self._is_sleeping = False
            return False

    async def is_server_sleeping(self) -> Optional[bool]:
        """Check if server is sleeping via API."""
        if not self.is_running:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.url}/is_sleeping",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("is_sleeping", False)
        except Exception:
            pass
        return None


def load_model_path(profile: str, config_path: Optional[Path] = None) -> str:
    """Load model path from settings.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("vllm", {}).get("models", {})
    if profile not in models:
        raise ValueError(f"Profile '{profile}' not found. Available: {list(models.keys())}")

    return models[profile]["path"]


class GPUOrchestrator:
    """
    Manages dynamic GPU allocation via vLLM sleep mode.

    Pre-loads all models in sleeping state, then wakes them as needed for each phase.
    Transitions take ~6-12 seconds (CPU RAM to GPU) instead of 60-120 seconds (disk to GPU).
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize orchestrator.

        Args:
            config: Orchestrator configuration. If None, loads from settings.yaml.
        """
        if config is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
            config = OrchestratorConfig.from_yaml(config_path)

        self.config = config
        self._mode = OrchestratorMode.UNINITIALIZED
        self._recovery_lock = threading.Lock()

        # Create managed servers
        self._task_primary = ManagedServer(
            config=config.task_primary,
            venv_path=config.venv_path,
            model_path=load_model_path(config.task_primary.profile, config.config_path),
            health_check_interval=config.health_check_interval,
            post_stop_settle_seconds=config.post_stop_settle_seconds,
        )
        self._task_replica = ManagedServer(
            config=config.task_replica,
            venv_path=config.venv_path,
            model_path=load_model_path(config.task_replica.profile, config.config_path),
            health_check_interval=config.health_check_interval,
            post_stop_settle_seconds=config.post_stop_settle_seconds,
        )
        self._genrm = ManagedServer(
            config=config.genrm,
            venv_path=config.venv_path,
            model_path=load_model_path(config.genrm.profile, config.config_path),
            health_check_interval=config.health_check_interval,
            post_stop_settle_seconds=config.post_stop_settle_seconds,
        )

    def _managed_server_for_port(self, port: int) -> Tuple[Optional[ManagedServer], str]:
        """Resolve an orchestrator-managed server for a TCP port."""
        port_int = int(port)
        if int(self._task_primary.port) == port_int:
            return self._task_primary, "task_primary"
        if int(self._task_replica.port) == port_int:
            return self._task_replica, "task_replica"
        if int(self._genrm.port) == port_int:
            return self._genrm, "genrm"
        return None, "unknown"

    async def _ensure_server_quiesced(
        self,
        server: ManagedServer,
        *,
        role_label: str,
        reason: str,
        force_stop: Optional[bool] = None,
    ) -> None:
        """Ensure `server` is not actively occupying GPUs (sleep or stop)."""
        if not server.is_running:
            return

        hard_quiesce = self.config.shared_gpu_hard_quiesce if force_stop is None else bool(force_stop)
        if hard_quiesce:
            logger.info(
                "%s: hard-quiescing %s on port %d (stop to fully free shared GPUs)",
                reason,
                role_label,
                int(server.port),
            )
            server.stop()
            await asyncio.sleep(self.config.post_stop_settle_seconds)
            return

        logger.info("%s: quiescing %s on port %d", reason, role_label, int(server.port))
        slept = False
        try:
            slept = await server.sleep(timeout=self.config.sleep_timeout)
        except Exception as exc:
            logger.warning("%s: %s sleep raised: %s", reason, role_label, exc)
            slept = False

        if slept:
            actual = await server.is_server_sleeping()
            if actual is True:
                return
            if actual is False:
                logger.warning(
                    "%s: %s reports awake after sleep; stopping process to free GPUs.",
                    reason,
                    role_label,
                )
            else:
                logger.warning(
                    "%s: %s sleep state unverified; stopping process to free GPUs.",
                    reason,
                    role_label,
                )
        else:
            logger.warning(
                "%s: %s failed to sleep cleanly; stopping process to free GPUs.",
                reason,
                role_label,
            )

        server.stop()
        await asyncio.sleep(self.config.post_stop_settle_seconds)

    async def recover_port(self, port: int, *, reason: str = "") -> bool:
        """
        Force-restart a managed server for `port` and restore current mode.

        This is intended for runtime recovery after connection failures.
        """
        try:
            port_int = int(port)
        except (TypeError, ValueError):
            logger.warning("recover_port called with invalid port: %s", port)
            return False

        server, role = self._managed_server_for_port(port_int)
        if server is None:
            logger.warning("recover_port: port %d is not orchestrator-managed", port_int)
            return False

        with self._recovery_lock:
            logger.warning(
                "Recovering %s server on port %d%s",
                role,
                port_int,
                f" (reason={reason})" if reason else "",
            )

            # task_replica and genrm share GPUs 2,3. Ensure the peer is not awake
            # before restarting the target.
            if role == "task_replica":
                await self._ensure_server_quiesced(
                    self._genrm,
                    role_label="GenRM",
                    reason="Recovery",
                    force_stop=True,
                )

            if role == "genrm":
                await self._ensure_server_quiesced(
                    self._task_replica,
                    role_label="task replica",
                    reason="Recovery",
                    force_stop=True,
                )

            if server.is_running:
                server.stop()
                await asyncio.sleep(2)

            try:
                await server.start()
            except Exception as exc:
                logger.error("Recovery: failed to start %s on port %d: %s", role, port_int, exc)
                return False

            # Re-assert current orchestrator mode so wake/sleep state is coherent.
            if role in {"task_primary", "task_replica"}:
                if self._mode == OrchestratorMode.TASK_DP2:
                    return await self.enter_task_dp2_mode()
                if self._mode == OrchestratorMode.DUAL_MODEL:
                    if role == "task_primary":
                        awake = await self._task_primary.wake(timeout=self.config.wake_timeout)
                        return bool(awake)
                    # Replica should generally remain sleeping in dual-model mode.
                    if self._task_replica.is_running:
                        try:
                            await self._task_replica.sleep(timeout=self.config.sleep_timeout)
                        except Exception:
                            pass
                    return True
                return True

            if role == "genrm":
                if self._mode == OrchestratorMode.DUAL_MODEL:
                    return await self.enter_dual_model_mode()
                return True

            return True

    @property
    def mode(self) -> OrchestratorMode:
        return self._mode

    async def initialize(self, initial_mode: OrchestratorMode = OrchestratorMode.TASK_DP2) -> None:
        """
        Start all servers and configure initial mode.

        Starts all three servers, then puts secondary ones to sleep.
        Initial mode determines which servers stay awake.

        Args:
            initial_mode: Initial GPU allocation mode.
        """
        logger.info("Initializing GPU orchestrator...")
        logger.info(f"  Task primary: port {self.config.task_primary.port}, GPUs {self.config.task_primary.cuda_devices}")
        logger.info(f"  Task replica: port {self.config.task_replica.port}, GPUs {self.config.task_replica.cuda_devices}")
        logger.info(f"  GenRM: port {self.config.genrm.port}, GPUs {self.config.genrm.cuda_devices}")

        # Start all servers (sequentially to avoid GPU memory contention during load).
        #
        # Important: task_replica and genrm share GPUs 2,3, so we must ensure the *other*
        # shared-GPU peer is quiesced (sleep/stop) before attempting to start one.
        #
        # This also handles the common "mixed static/dynamic" situation where users have
        # already started a GenRM server manually on port 8001 (no sleep endpoints), which
        # would otherwise leave only a few GiB free on GPUs 2,3 and cause task_replica
        # startup to fail.
        await self._task_primary.start()

        # For TASK_DP2: We want task_replica active and genrm sleeping
        # For DUAL_MODEL: We want genrm active and task_replica sleeping
        # Either way, we need to load both into RAM, but only one can use GPUs 2,3 at a time
        if initial_mode == OrchestratorMode.TASK_DP2:
            # Ensure GenRM is not occupying the shared GPUs before starting the replica.
            await self._ensure_server_quiesced(
                self._genrm,
                role_label="GenRM",
                reason="initialize:prepare_task_replica",
                force_stop=None,
            )
            # 1. Start task_replica on GPUs 2,3
            await self._task_replica.start()
            # 2. Sleep task_replica (offload to RAM, free GPUs 2,3)
            await self._task_replica.sleep()
            # 3. Start genrm on GPUs 2,3 (now free)
            await self._genrm.start()
            # 4. Sleep genrm (offload to RAM, free GPUs 2,3)
            await self._genrm.sleep()
            # 5. Wake task_replica for DP=2 mode
            awake = await self._task_replica.wake()
            if not awake:
                logger.warning(
                    "Initialization wake failed for task replica on port %d; cold restarting replica",
                    int(self._task_replica.port),
                )
                self._task_replica.stop()
                await asyncio.sleep(self.config.post_stop_settle_seconds)
                await self._task_replica.start()
        else:  # DUAL_MODEL
            # Ensure the replica is not occupying the shared GPUs before starting GenRM.
            await self._ensure_server_quiesced(
                self._task_replica,
                role_label="task replica",
                reason="initialize:prepare_genrm",
                force_stop=None,
            )
            # 1. Start genrm on GPUs 2,3
            await self._genrm.start()
            # 2. Sleep genrm (offload to RAM, free GPUs 2,3)
            await self._genrm.sleep()
            # 3. Start task_replica on GPUs 2,3 (now free)
            await self._task_replica.start()
            # 4. Sleep task_replica (offload to RAM, free GPUs 2,3)
            await self._task_replica.sleep()
            # 5. Wake genrm for DUAL_MODEL mode
            awake = await self._genrm.wake()
            if not awake:
                logger.warning(
                    "Initialization wake failed for GenRM on port %d; cold restarting GenRM",
                    int(self._genrm.port),
                )
                self._genrm.stop()
                await asyncio.sleep(self.config.post_stop_settle_seconds)
                await self._genrm.start()

        self._mode = initial_mode
        logger.info(f"Orchestrator initialized in {initial_mode.value} mode")

    async def enter_task_dp2_mode(self) -> bool:
        """
        Switch to DP=2 mode (both task models active, GenRM sleeping).

        Returns:
            True if transition successful.
        """
        if self._mode == OrchestratorMode.TASK_DP2:
            logger.debug("Ensuring task_dp2 mode...")
        else:
            logger.info("Transitioning to task_dp2 mode...")
        start = time.time()

        if self._genrm.is_running:
            await self._ensure_server_quiesced(
                self._genrm,
                role_label="GenRM",
                reason="task_dp2 transition",
                force_stop=None,
            )

        if not self._task_replica.is_running:
            logger.warning(
                "Task replica on port %d is not running; restarting.",
                int(self._task_replica.port),
            )
            try:
                await self._task_replica.start()
            except Exception as exc:
                logger.error("Failed to start task replica on port %d: %s", int(self._task_replica.port), exc)
                return False

        awake = await self._task_replica.wake(timeout=self.config.wake_timeout)
        if not awake:
            logger.error(
                "Failed to wake task replica on port %d; attempting cold restart",
                int(self._task_replica.port),
            )
            # Wake can fail with shared-memory/OOM issues on long runs; recover in-place.
            self._task_replica.stop()
            await asyncio.sleep(self.config.post_stop_settle_seconds)
            try:
                await self._task_replica.start()
            except Exception as exc:
                logger.error(
                    "Cold restart failed for task replica on port %d: %s",
                    int(self._task_replica.port),
                    exc,
                )
                return False

        elapsed = time.time() - start
        logger.info(f"Transitioned to task_dp2 mode in {elapsed:.1f}s")
        self._mode = OrchestratorMode.TASK_DP2
        return True

    async def enter_dual_model_mode(self) -> bool:
        """
        Switch to dual model mode (task + GenRM, replica sleeping).

        Returns:
            True if transition successful.
        """
        if self._mode == OrchestratorMode.DUAL_MODEL:
            logger.debug("Ensuring dual_model mode...")
        else:
            logger.info("Transitioning to dual_model mode...")
        start = time.time()

        if self._task_replica.is_running:
            await self._ensure_server_quiesced(
                self._task_replica,
                role_label="task replica",
                reason="dual_model transition",
                force_stop=None,
            )

        if not self._genrm.is_running:
            logger.warning("GenRM on port %d is not running; restarting.", int(self._genrm.port))
            try:
                await self._genrm.start()
            except Exception as exc:
                logger.error("Failed to start GenRM on port %d: %s", int(self._genrm.port), exc)
                return False

        awake = await self._genrm.wake(timeout=self.config.wake_timeout)
        if not awake:
            logger.error(
                "Failed to wake GenRM on port %d; attempting cold restart",
                int(self._genrm.port),
            )
            # Wake can fail with shared-memory/OOM issues on long runs; recover in-place.
            self._genrm.stop()
            await asyncio.sleep(self.config.post_stop_settle_seconds)
            try:
                await self._genrm.start()
            except Exception as exc:
                logger.error(
                    "Cold restart failed for GenRM on port %d: %s",
                    int(self._genrm.port),
                    exc,
                )
                return False

        elapsed = time.time() - start
        logger.info(f"Transitioned to dual_model mode in {elapsed:.1f}s")
        self._mode = OrchestratorMode.DUAL_MODEL
        return True

    def get_active_task_ports(self) -> List[int]:
        """Get list of active task model ports for current mode."""
        if self._mode == OrchestratorMode.TASK_DP2:
            return [self.config.task_primary.port, self.config.task_replica.port]
        else:
            return [self.config.task_primary.port]

    def get_genrm_port(self) -> int:
        """Get GenRM port."""
        return self.config.genrm.port

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        return {
            "mode": self._mode.value,
            "task_primary": {
                "port": self._task_primary.port,
                "running": self._task_primary.is_running,
                "sleeping": self._task_primary.is_sleeping,
            },
            "task_replica": {
                "port": self._task_replica.port,
                "running": self._task_replica.is_running,
                "sleeping": self._task_replica.is_sleeping,
            },
            "genrm": {
                "port": self._genrm.port,
                "running": self._genrm.is_running,
                "sleeping": self._genrm.is_sleeping,
            },
        }

    async def shutdown(self) -> None:
        """Stop all servers."""
        logger.info("Shutting down GPU orchestrator...")
        self._task_primary.stop()
        self._task_replica.stop()
        self._genrm.stop()
        self._mode = OrchestratorMode.UNINITIALIZED
        logger.info("GPU orchestrator shutdown complete")

    async def __aenter__(self) -> "GPUOrchestrator":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()
