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
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import yaml

logger = logging.getLogger(__name__)


def kill_process_on_port(port: int) -> bool:
    """Kill any process listening on the given port.

    Returns True if a process was killed, False if no process was found.
    """
    try:
        # Find process using lsof
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    pid_int = int(pid.strip())
                    logger.info(f"Killing existing process {pid_int} on port {port}")
                    os.kill(pid_int, signal.SIGTERM)
                except (ValueError, ProcessLookupError):
                    pass
            # Wait a moment for cleanup
            time.sleep(1)
            return True
    except FileNotFoundError:
        # lsof not available, try fuser
        try:
            result = subprocess.run(
                ["fuser", f"{port}/tcp"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split()
                for pid in pids:
                    try:
                        pid_int = int(pid.strip())
                        logger.info(f"Killing existing process {pid_int} on port {port}")
                        os.kill(pid_int, signal.SIGTERM)
                    except (ValueError, ProcessLookupError):
                        pass
                time.sleep(1)
                return True
        except FileNotFoundError:
            logger.warning("Neither lsof nor fuser available to check port")
    except Exception as e:
        logger.warning(f"Error checking port {port}: {e}")
    return False


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
    enable_sleep_mode: bool = True
    startup_timeout: float = 300.0
    gpu_memory_utilization: float = 0.90


@dataclass
class OrchestratorConfig:
    """Configuration for the GPU orchestrator."""
    # Server configs
    task_primary: ServerConfig = field(default_factory=lambda: ServerConfig(
        profile="nemotron-30b-fp8",
        port=8000,
        cuda_devices="0,1",
        tensor_parallel=2,
    ))
    task_replica: ServerConfig = field(default_factory=lambda: ServerConfig(
        profile="nemotron-30b-fp8",
        port=8002,
        cuda_devices="2,3",
        tensor_parallel=2,
    ))
    genrm: ServerConfig = field(default_factory=lambda: ServerConfig(
        profile="genrm-nvfp4",
        port=8001,
        cuda_devices="2,3",
        tensor_parallel=2,
        gpu_memory_utilization=0.95,
    ))

    # Paths
    venv_path: str = "/home/mlinegar/vllm-env"
    config_path: Optional[Path] = None

    # Timeouts
    sleep_timeout: float = 30.0
    wake_timeout: float = 60.0
    health_check_interval: float = 2.0

    @classmethod
    def from_yaml(cls, config_path: Path) -> "OrchestratorConfig":
        """Load orchestrator config from settings.yaml."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        orch_cfg = cfg.get("orchestration", {})
        vllm_cfg = cfg.get("vllm", {})
        models = vllm_cfg.get("models", {})

        # Get model profiles
        task_profile = orch_cfg.get("task_model_profile", vllm_cfg.get("default", "nemotron-30b-fp8"))
        genrm_profile = orch_cfg.get("genrm_profile", "genrm-nvfp4")

        # Get model configs
        task_model = models.get(task_profile, {})
        genrm_model = models.get(genrm_profile, {})

        return cls(
            task_primary=ServerConfig(
                profile=task_profile,
                port=orch_cfg.get("task_primary_port", 8000),
                cuda_devices=orch_cfg.get("task_primary_gpus", "0,1"),
                tensor_parallel=task_model.get("tensor_parallel", 2),
                gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.90),
            ),
            task_replica=ServerConfig(
                profile=task_profile,
                port=orch_cfg.get("task_replica_port", 8002),
                cuda_devices=orch_cfg.get("task_replica_gpus", "2,3"),
                tensor_parallel=task_model.get("tensor_parallel", 2),
                gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.90),
            ),
            genrm=ServerConfig(
                profile=genrm_profile,
                port=orch_cfg.get("genrm_port", 8001),
                cuda_devices=orch_cfg.get("genrm_gpus", "2,3"),
                tensor_parallel=genrm_model.get("tensor_parallel", 2),
                gpu_memory_utilization=orch_cfg.get("genrm_gpu_memory_utilization", 0.95),
            ),
            venv_path=orch_cfg.get("venv_path", "/home/mlinegar/vllm-env"),
            config_path=config_path,
            sleep_timeout=orch_cfg.get("sleep_timeout", 30.0),
            wake_timeout=orch_cfg.get("wake_timeout", 60.0),
        )


class ManagedServer:
    """Manages a single vLLM server process with sleep mode support."""

    def __init__(
        self,
        config: ServerConfig,
        venv_path: str,
        model_path: str,
        health_check_interval: float = 2.0,
    ):
        self.config = config
        self.venv_path = venv_path
        self.model_path = model_path
        self.health_check_interval = health_check_interval

        self._process: Optional[subprocess.Popen] = None
        self._log_file = None
        self._is_sleeping = False

    @property
    def port(self) -> int:
        return self.config.port

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def is_sleeping(self) -> bool:
        return self._is_sleeping

    async def start(self) -> None:
        """Start the vLLM server with sleep mode enabled."""
        if self.is_running:
            logger.warning(f"Server on port {self.port} already running (our process)")
            return

        # Kill any existing process on this port (from previous runs)
        if kill_process_on_port(self.port):
            logger.info(f"Killed stale server on port {self.port}")
            # Wait a bit more for GPU memory to be released
            await asyncio.sleep(2)

        logger.info(f"Starting vLLM server on port {self.port}")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  CUDA devices: {self.config.cuda_devices}")
        logger.info(f"  Tensor parallel: {self.config.tensor_parallel}")
        logger.info(f"  Sleep mode: {self.config.enable_sleep_mode}")

        python_path = os.path.join(self.venv_path, "bin", "python")
        cmd = [
            python_path,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.config.tensor_parallel),
            "--max-model-len", "32768",
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--trust-remote-code",
            "--enforce-eager",
            "--disable-log-requests",
        ]

        if self.config.enable_sleep_mode:
            cmd.append("--enable-sleep-mode")

        # Environment with CUDA device isolation and dev mode for sleep endpoints
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_devices
        if self.config.enable_sleep_mode:
            env["VLLM_SERVER_DEV_MODE"] = "1"

        # Create log file
        self._log_file = tempfile.NamedTemporaryFile(
            mode='w',
            prefix=f'vllm_port{self.port}_',
            suffix='.log',
            delete=False,
        )
        logger.info(f"  Log file: {self._log_file.name}")

        # Start process
        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=env,
        )

        logger.info(f"Server process started (PID: {self._process.pid})")

        # Wait for ready
        await self._wait_for_ready()
        self._is_sleeping = False

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
            logger.debug(f"Server on port {self.port} already sleeping")
            return True

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
                        return False
        except Exception as e:
            logger.error(f"Failed to sleep server on port {self.port}: {e}")
            return False

    async def wake(self, timeout: float = 60.0) -> bool:
        """Wake server from sleep (reload weights from CPU RAM)."""
        if not self.is_running:
            logger.warning(f"Cannot wake server on port {self.port}: not running")
            return False

        if not self._is_sleeping:
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
                        return False
        except Exception as e:
            logger.error(f"Failed to wake server on port {self.port}: {e}")
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

        # Create managed servers
        self._task_primary = ManagedServer(
            config=config.task_primary,
            venv_path=config.venv_path,
            model_path=load_model_path(config.task_primary.profile, config.config_path),
            health_check_interval=config.health_check_interval,
        )
        self._task_replica = ManagedServer(
            config=config.task_replica,
            venv_path=config.venv_path,
            model_path=load_model_path(config.task_replica.profile, config.config_path),
            health_check_interval=config.health_check_interval,
        )
        self._genrm = ManagedServer(
            config=config.genrm,
            venv_path=config.venv_path,
            model_path=load_model_path(config.genrm.profile, config.config_path),
            health_check_interval=config.health_check_interval,
        )

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

        # Start all servers (sequentially to avoid GPU memory contention during load)
        # Note: task_replica and genrm share GPUs 2,3, so we must sleep one before starting the other
        await self._task_primary.start()

        # For TASK_DP2: We want task_replica active and genrm sleeping
        # For DUAL_MODEL: We want genrm active and task_replica sleeping
        # Either way, we need to load both into RAM, but only one can use GPUs 2,3 at a time
        if initial_mode == OrchestratorMode.TASK_DP2:
            # 1. Start task_replica on GPUs 2,3
            await self._task_replica.start()
            # 2. Sleep task_replica (offload to RAM, free GPUs 2,3)
            await self._task_replica.sleep()
            # 3. Start genrm on GPUs 2,3 (now free)
            await self._genrm.start()
            # 4. Sleep genrm (offload to RAM, free GPUs 2,3)
            await self._genrm.sleep()
            # 5. Wake task_replica for DP=2 mode
            await self._task_replica.wake()
        else:  # DUAL_MODEL
            # 1. Start genrm on GPUs 2,3
            await self._genrm.start()
            # 2. Sleep genrm (offload to RAM, free GPUs 2,3)
            await self._genrm.sleep()
            # 3. Start task_replica on GPUs 2,3 (now free)
            await self._task_replica.start()
            # 4. Sleep task_replica (offload to RAM, free GPUs 2,3)
            await self._task_replica.sleep()
            # 5. Wake genrm for DUAL_MODEL mode
            await self._genrm.wake()

        self._mode = initial_mode
        logger.info(f"Orchestrator initialized in {initial_mode.value} mode")

    async def enter_task_dp2_mode(self) -> bool:
        """
        Switch to DP=2 mode (both task models active, GenRM sleeping).

        Returns:
            True if transition successful.
        """
        if self._mode == OrchestratorMode.TASK_DP2:
            logger.debug("Already in task_dp2 mode")
            return True

        logger.info("Transitioning to task_dp2 mode...")
        start = time.time()

        # Sleep GenRM and wake task replica in parallel
        results = await asyncio.gather(
            self._genrm.sleep(timeout=self.config.sleep_timeout),
            self._task_replica.wake(timeout=self.config.wake_timeout),
            return_exceptions=True,
        )

        # Check results
        success = all(r is True for r in results if not isinstance(r, Exception))
        if not success:
            logger.error(f"Transition to task_dp2 failed: {results}")
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
            logger.debug("Already in dual_model mode")
            return True

        logger.info("Transitioning to dual_model mode...")
        start = time.time()

        # Sleep task replica and wake GenRM in parallel
        results = await asyncio.gather(
            self._task_replica.sleep(timeout=self.config.sleep_timeout),
            self._genrm.wake(timeout=self.config.wake_timeout),
            return_exceptions=True,
        )

        # Check results
        success = all(r is True for r in results if not isinstance(r, Exception))
        if not success:
            logger.error(f"Transition to dual_model failed: {results}")
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
