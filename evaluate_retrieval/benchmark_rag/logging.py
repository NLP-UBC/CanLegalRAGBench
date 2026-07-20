"""
Structured logging for benchmark experiments.

Improvements over cad_rag:
  - Experiment ID is baked into every log line via a LoggerAdapter.
  - setup_experiment_logging() is the single call-site; no global mutable state.
  - Resource monitor is encapsulated in a class (no module-level _stop flag).
  - JSON structured log file alongside the human-readable one.
  - No psutil / pynvml hard dependencies — gracefully degrades.

Usage
-----
    from benchmark_rag.logging import setup_experiment_logging, get_logger

    setup_experiment_logging(experiment_id="exp_001", log_dir="runs/exp_001/logs")
    log = get_logger(__name__)
    log.info("Starting pipeline")          # → "exp_001 | __main__ | INFO | Starting pipeline"
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import MutableMapping, Any


# ---------------------------------------------------------------------------
# Adapter — injects experiment_id into every log record
# ---------------------------------------------------------------------------


class _ExperimentAdapter(logging.LoggerAdapter):
    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        eid = self.extra.get("experiment_id", "")
        return f"[{eid}] {msg}", kwargs


# Module-level experiment_id so get_logger() can pick it up without arguments.
_active_experiment_id: str = ""


def setup_experiment_logging(
    experiment_id: str,
    log_dir: str | Path,
    level: str = "INFO",
    resource_monitor_interval: float = 30.0,
) -> None:
    """
    Configure the root logger for a single experiment run.

    Creates two files in log_dir:
      - {experiment_id}.log   — human-readable
      - {experiment_id}.jsonl — one JSON object per log record (for analysis)

    Parameters
    ----------
    experiment_id:
        Injected into every log line.
    log_dir:
        Directory where log files are written.  Created if missing.
    level:
        Root log level ("DEBUG", "INFO", …).
    resource_monitor_interval:
        Seconds between resource snapshots.  Pass 0 to disable.
    """
    global _active_experiment_id
    _active_experiment_id = experiment_id

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # Silence chatty third-party loggers regardless of our level setting
    for noisy in ("httpcore", "httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Human-readable file
    fh = logging.FileHandler(log_dir / f"{experiment_id}.log")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # JSON lines file
    jh = _JsonlHandler(log_dir / f"{experiment_id}.jsonl", experiment_id=experiment_id)
    root.addHandler(jh)

    if resource_monitor_interval > 0:
        monitor = ResourceMonitor(
            experiment_id=experiment_id,
            interval=resource_monitor_interval,
        )
        monitor.start()


def get_logger(name: str) -> _ExperimentAdapter:
    """
    Return a logger adapter that prepends the active experiment ID.

    Call setup_experiment_logging() once before using get_logger().
    Falls back gracefully if setup was never called.
    """
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )
    base = logging.getLogger(name)
    return _ExperimentAdapter(base, {"experiment_id": _active_experiment_id})


# ---------------------------------------------------------------------------
# JSON lines handler
# ---------------------------------------------------------------------------


class _JsonlHandler(logging.Handler):
    def __init__(self, path: Path, experiment_id: str):
        super().__init__()
        self._path = path
        self._experiment_id = experiment_id
        self._file = open(path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        from datetime import datetime
        entry = {
            "ts": datetime.fromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment_id": self._experiment_id,
            "logger": record.name,
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        try:
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        self._file.close()
        super().close()


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------


class ResourceMonitor:
    """
    Background thread that periodically logs CPU / RAM / GPU usage.

    Encapsulated as a class to avoid module-level mutable state.
    """

    def __init__(self, experiment_id: str, interval: float = 30.0):
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"ResourceMonitor-{experiment_id}",
            daemon=True,
        )
        self._log = get_logger("ResourceMonitor")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._snapshot()

    def _snapshot(self) -> None:
        parts: list[str] = []

        try:
            import psutil

            proc = psutil.Process(os.getpid())
            cpu = proc.cpu_percent(interval=None)
            ram_mb = proc.memory_info().rss / 1024**2
            sys_ram = psutil.virtual_memory().percent
            parts.append(f"CPU {cpu:.1f}% | RAM {ram_mb:.0f}MB (sys {sys_ram:.0f}%)")
        except ImportError:
            parts.append("psutil not installed — CPU/RAM stats unavailable")

        try:
            import pynvml

            pynvml.nvmlInit()
            gpu_parts = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                gpu_parts.append(
                    f"GPU{i} {util.gpu}% util {mem.used // 1024**2}/{mem.total // 1024**2}MB"
                )
            parts.append(" | ".join(gpu_parts))
        except Exception:
            pass  # no GPU or pynvml not installed

        self._log.info(" | ".join(parts))


def log_resource_snapshot(logger: _ExperimentAdapter | logging.Logger) -> None:
    """One-shot resource log — call inside loops for on-demand snapshots."""
    mon = ResourceMonitor.__new__(ResourceMonitor)
    mon._log = logger  # type: ignore[attr-defined]
    mon._snapshot()
