"""Background job execution.

Heavy work (separation, transcription) runs in a separate *process* pool so that:
  * the API stays responsive (no GIL contention, no blocking the event loop);
  * a crash or out-of-memory in a model kills only the worker, not the server;
  * memory is returned to the OS after each job (workers are recycled).

Workers report progress by writing ``project.json`` (atomic replace), which the API
reads on demand - no shared memory or message broker needed for a single machine.
For multi-machine deployments the same ``run_job`` entry point can be called from a
Celery/RQ worker instead.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from concurrent.futures import Future, ProcessPoolExecutor

from . import storage
from .config import settings

log = logging.getLogger("tabforge.jobs")


class JobManager:
    def __init__(self, workers: int | None = None):
        self.workers = max(1, workers or settings.workers)
        self._pool: ProcessPoolExecutor | None = None
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            ctx = mp.get_context("spawn")
            # max_tasks_per_child recycles workers so model memory does not accumulate
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx, max_tasks_per_child=1)
        return self._pool

    def submit(self, kind: str, pid: str, args: dict | None = None) -> None:
        from .pipeline import run_job  # noqa: PLC0415

        with self._lock:
            f = self._futures.get(pid)
            if f is not None and not f.done():
                raise RuntimeError("A job is already running for this project")
            storage.set_status(pid, "queued", "queued", 0.0, "Waiting for a worker")
            fut = self._get_pool().submit(run_job, kind, pid, args or {})
            self._futures[pid] = fut
            fut.add_done_callback(lambda f, pid=pid: self._done(pid, f))

    def _done(self, pid: str, fut: Future) -> None:
        exc = fut.exception()
        if exc is not None:
            log.error("job for %s failed: %s", pid, exc)
            try:
                meta = storage.load_project(pid)
                if meta["status"].get("state") not in ("failed",):
                    meta["status"].update({"state": "failed", "message": f"Worker crashed: {exc}"})
                    storage.save_project(meta)
            except Exception:  # noqa: BLE001
                pass

    def is_running(self, pid: str) -> bool:
        f = self._futures.get(pid)
        return f is not None and not f.done()

    def recover(self) -> int:
        """Re-queue projects that were waiting or interrupted when the server stopped."""
        n = 0
        for meta in storage.list_projects():
            state = meta.get("status", {}).get("state")
            if state in ("uploaded", "queued", "processing") and meta.get("source"):
                try:
                    self.submit("process", meta["id"])
                    n += 1
                except Exception as e:  # noqa: BLE001
                    log.warning("could not recover %s: %s", meta["id"], e)
        return n

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)


jobs = JobManager()
