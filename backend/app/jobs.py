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
from concurrent.futures.process import BrokenProcessPool

from . import storage
from .config import settings

log = logging.getLogger("tabforge.jobs")


class JobBusy(Exception):
    """A job is already running for this project."""


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
                raise JobBusy("A job is already running for this project")
            # remember what is running so a restart re-runs the same job, not a full reprocess
            meta = storage.load_project(pid)
            meta["status"].update({"state": "queued", "stage": "queued", "progress": 0.0,
                                   "message": "Waiting for a worker", "job": {"kind": kind, "args": args or {}}})
            meta["status"].pop("error", None)
            storage.save_project(meta)
            try:
                fut = self._get_pool().submit(run_job, kind, pid, args or {})
            except BrokenProcessPool:
                # a worker died (out of memory, segfault): start a fresh pool and retry once
                log.warning("process pool was broken; restarting it")
                self._reset_pool()
                fut = self._get_pool().submit(run_job, kind, pid, args or {})
            self._futures[pid] = fut
            fut.add_done_callback(lambda f, pid=pid, kind=kind: self._done(pid, kind, f))

    def _reset_pool(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass

    def _done(self, pid: str, kind: str, fut: Future) -> None:
        exc = fut.exception()
        if exc is None:
            return
        log.error("job %s for %s failed: %s", kind, pid, exc)
        if isinstance(exc, BrokenProcessPool):
            self._reset_pool()
        try:
            meta = storage.load_project(pid)
            st = meta["status"]
            if st.get("state") in ("queued", "processing"):
                msg = "The worker process stopped unexpectedly (often out of memory)"
                if kind == "process":
                    st.update({"state": "failed", "message": f"{msg}: {exc}"})
                else:
                    # an on-demand transcription failed; the existing results stay usable
                    st.update({"state": "done", "stage": "done", "progress": 1.0, "message": f"{msg}: {exc}",
                               "last_error": str(exc)})
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
                job = meta.get("status", {}).get("job") or {"kind": "process", "args": {}}
                try:
                    self.submit(job.get("kind", "process"), meta["id"], job.get("args") or {})
                    n += 1
                except Exception as e:  # noqa: BLE001
                    log.warning("could not recover %s: %s", meta["id"], e)
        return n

    def shutdown(self) -> None:
        """Stop immediately: running jobs are killed and re-queued by recover() on next start."""
        pool = self._pool
        if pool is None:
            return
        for proc in list(getattr(pool, "_processes", {}).values()):
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        pool.shutdown(wait=False, cancel_futures=True)


jobs = JobManager()
