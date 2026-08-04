import logging
import multiprocessing as mp
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException

from clip_process import ClipOfflineAnalyzeWorker
from expert_process import ExpertOfflineAnalyzeWorker, WeaponOfflineAnalyzeWorker
from general.analyzer_general import logger, log_exception
from general.offline_analytics import get_ipc_sockets, get_offline_storage_path
from offline_mq_common import OfflineAnalyzerWorker

# Global references to workers
workers: List[OfflineAnalyzerWorker] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    # Startup: Create and start workers
    logger.info("Starting background workers...")
    log_queue = mp.Queue(-1)
    listener = logging.handlers.QueueListener(log_queue, *logger.handlers)
    listener.start()
    old_umask = os.umask(0)  # Clear umask temporarily

    try:
        clip_socket_file, expert_socket_file, weapon_socket_file = get_ipc_sockets(make_dir=True)
        os.makedirs(get_offline_storage_path(), mode=0o777, exist_ok=True)

    except Exception as e:
        log_exception(logger, "Failed to create offline storage directory", e)
    finally:
        os.umask(old_umask)  # Restore original umask

    clip_worker = ClipOfflineAnalyzeWorker(ipc_socket=clip_socket_file, log_queue=log_queue)
    expert_worker = ExpertOfflineAnalyzeWorker(ipc_socket=expert_socket_file, log_queue=log_queue)
    weapon_worker = WeaponOfflineAnalyzeWorker(ipc_socket=weapon_socket_file, log_queue=log_queue)
    workers.extend([clip_worker, expert_worker, weapon_worker])
    for worker in workers:
        worker.start()

    logger.info("Workers started successfully")
    logger.info("Health check available at /health")

    yield  # Application runs here

    # Shutdown: Clean up workers
    logger.info("Shutting down workers...")
    for worker in workers:
        if hasattr(worker, "stop"):
            worker.stop()

    # Wait for workers to finish
    for worker in workers:
        if worker.is_alive():
            worker.join(timeout=5.0)

    logger.info("Workers stopped")

    # Stop the listener when done
    listener.stop()


app = FastAPI(title="Offline Analyzer Health API", lifespan=lifespan)


@app.get("/expert_health")
@app.get("/health")
async def health():
    # Overall health check - verify both process alive AND socket bound
    workers_status = [w.is_alive() and w.ready_event.is_set() for w in workers]
    if all(workers_status):
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "workers": {w.name: _get_worker_status(w) for w in workers},
        }
    else:
        raise HTTPException(status_code=500, detail="error")


@app.get("/metrics")
async def metrics():
    """Detailed metrics"""
    return {w.name: w.get_metrics() for w in workers}


@app.post("/flush_queues")
async def flush_queues():
    """Flush all queues in all workers"""
    for worker in workers:
        worker.flush_event.set()
    return {"status": "ok", "message": "All queues flushed"}


def _get_worker_status(worker):
    return {
        "status": "running" if worker.is_alive() else "stopped",
        "ready": worker.ready_event.is_set(),
        "pid": worker.pid,
    }
