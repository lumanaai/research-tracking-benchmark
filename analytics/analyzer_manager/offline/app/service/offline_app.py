import asyncio
import multiprocessing as mp
import os
from contextlib import asynccontextmanager
from queue import Full
from typing import Dict

from fastapi import FastAPI, HTTPException

from clip_worker import ClipPayload, ClipBackgroundWorker
from expert_worker import ExpertPayload, ExpertBackgroundWorker
from general.analyzer_general import logger, log_exception
# from utils.infra.cloud_logger import logger
from weapon_worker import WcBackgroundWorker, WeaponPayload

MAX_QUEUE_SIZE = 20
RESULTS_QUEUE_SIZE = 50
CLIP_MAX_QUEUE_SIZE = 200

instance_id = str(os.environ.get("INSTANCE_ID", "expert"))
os.environ["TRITON_ENABLED"] = "True"  # "False" #


# Start the expert worker for handling expert tasks
EXPERT_MAX_QUEUE_SIZE = 20

expert_queue = mp.Queue(maxsize=EXPERT_MAX_QUEUE_SIZE)
expert_result_queue = mp.Queue()

# Per-camera async queues
expert_camera_states: Dict[str, asyncio.Queue] = {}
camera_state_lock = asyncio.Lock()


# Start the background worker
expert_background_worker = ExpertBackgroundWorker(expert_queue, expert_result_queue)
expert_background_worker.start()

weapon_camera_states: Dict[str, asyncio.Queue] = {}
weapons_queue = mp.Queue(maxsize=MAX_QUEUE_SIZE)
weapons_result_queue = mp.Queue()

# Start the background worker
weapons_background_worker = WcBackgroundWorker(weapons_queue, weapons_result_queue, expert_result_queue)
weapons_background_worker.start()


clip_queue = mp.Queue(maxsize=CLIP_MAX_QUEUE_SIZE)
clip_result_queue = mp.Queue()

# Start the background worker
clip_camera_results: Dict[str, asyncio.Queue] = {}
clip_background_worker = ClipBackgroundWorker(clip_queue, clip_result_queue)
clip_background_worker.start()

queues_and_states = [
    (expert_result_queue, expert_camera_states, "expert"),
    (clip_result_queue, clip_camera_results, "clip"),
    (weapons_result_queue, weapon_camera_states, "weapons"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage background tasks lifecycle"""
    # Startup: Create background task
    drain_task = asyncio.create_task(drain_queues_periodically(interval_seconds=1))
    logger.info("Started drain_queues_periodically background task")

    yield  # App runs here

    # Shutdown: Cancel background task
    drain_task.cancel()
    try:
        await drain_task
    except asyncio.CancelledError:
        logger.info("Stopped drain_queues_periodically background task")


# FastAPI app initialization with lifespan
app = FastAPI(lifespan=lifespan)
logger.info("Offline analytics service started")


def get_queue_results(camera_id: str, camera_state: Dict[str, asyncio.Queue]):
    queue = camera_state.get(camera_id)

    if not queue:
        return False, [], "Camera not enrolled"

    # Drain the queue (asyncio.Queue is async-safe, no lock needed)
    results = []
    while not queue.empty():
        try:
            results.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return True, results, ""


@app.post("/enroll_camera")
async def enroll_camera(camera_id: str):
    """Explicitly create queue for a camera before use"""
    async with camera_state_lock:
        if camera_id not in expert_camera_states:
            expert_camera_states[camera_id] = asyncio.Queue(maxsize=RESULTS_QUEUE_SIZE)
        if camera_id not in clip_camera_results:
            clip_camera_results[camera_id] = asyncio.Queue(maxsize=RESULTS_QUEUE_SIZE)
        if camera_id not in weapon_camera_states:
            weapon_camera_states[camera_id] = asyncio.Queue(maxsize=RESULTS_QUEUE_SIZE)
    return {"success": True, "message": f"Camera {camera_id} enrolled"}


@app.get("/flush_queues")
async def flush_all_queues():
    """Flush all queues and camera states for testing purposes"""
    has_error = False
    message = ""
    async with camera_state_lock:
        try:
            # Flush queues and states
            for tup in queues_and_states:
                while not tup[0].empty():
                    try:
                        tup[0].get_nowait()
                    except asyncio.QueueFull:
                        break
                tup[1].clear()
        except Exception as e:
            log_exception(logger, "Error flushing queues", e)
            has_error = True
            message += f"Error flushing {tup[2]} queue: {str(e)}; "
    logger.info("All queues and camera states flushed successfully")
    message = message if has_error else "All queues and camera states flushed"
    return {"success": not has_error, "message": message}


@app.post("/upload_weapons_data")
async def process_guns_data(payload: WeaponPayload):
    # add to queue:
    try:
        weapons_queue.put_nowait(payload)
    except Full:
        return {"success": False, "message": "Queue is full, try again later"}
    return {"success": True, "message": "Image processed and state updated"}


@app.get("/get_weapon_alerts")
async def get_active_alerts(camera_id: str):
    success, res, msg = get_queue_results(camera_id, weapon_camera_states)
    msg = "camera alerts sent" if success else msg
    return {"success": success, "message": msg, "alerts": res}


@app.get("/terminate")
async def terminate():
    workers_lst = [weapons_background_worker, clip_background_worker, expert_background_worker]
    for worker in workers_lst:
        worker.terminate()
    return {"status": "ok"}


@app.get("/health")
async def health():
    if (
        weapons_background_worker.is_alive()
        and clip_background_worker.is_alive()
        and expert_background_worker.is_alive()
    ):
        return {"status": "ok"}
    raise HTTPException(status_code=500, detail="error")


@app.post("/upload_clip_data")
async def process_clip_data(payload: ClipPayload):
    try:
        clip_queue.put_nowait(payload)
    except Full:
        return {"success": False, "message": "Queue is full, try again later"}
    return {"success": True, "message": "Image Received for processing"}


@app.get("/get_clip_encodings")
async def get_clip_results(camera_id: str):
    success, res, msg = get_queue_results(camera_id, clip_camera_results)
    msg = "clip encodings flushed" if success else msg
    return {"success": success, "message": msg, "encodings": res}


@app.get("/expert_health")
async def expert_health():
    if expert_background_worker.is_alive():
        return {"status": "ok"}
    return {"status": "error"}


@app.post("/upload_expert_data")
async def process_expert_data(payload: ExpertPayload):
    # add to queue:
    try:
        expert_queue.put_nowait(payload)
    except Full:
        return {"success": False, "message": "Expert queue is full, try again later"}
    return {"success": True, "message": "Image processed and state updated"}


@app.get("/get_expert_detections")
async def get_expert_detections(camera_id: str):
    success, res, msg = get_queue_results(camera_id, expert_camera_states)
    msg = "expert detections sent" if success else msg
    return {"success": success, "message": msg, "detections": res}


async def drain_queues_periodically(interval_seconds: float = 1):
    """Drain mp.Queue and distribute to per-camera async queues"""

    while True:
        for task_queue, camera_states, task_name in queues_and_states:
            try:
                while not task_queue.empty():
                    try:
                        result = task_queue.get_nowait()
                        camera_id = result["cameraId"]

                        # Check if queue exists, create if not
                        queue = camera_states.get(camera_id)
                        if not queue:
                            async with camera_state_lock:
                                # Double-check after acquiring lock
                                queue = camera_states.get(camera_id)
                                if not queue:
                                    queue = asyncio.Queue(maxsize=RESULTS_QUEUE_SIZE)
                                    camera_states[camera_id] = queue
                                    logger.info(f"Auto-enrolled camera {camera_id} for {task_name}")

                        queue.put_nowait(result)
                    except asyncio.QueueFull:
                        logger.warning(
                            f"{task_name} queue for camera {camera_id} is full, dropping result for {task_name}"
                        )
                        break
                    except Exception as e:
                        log_exception(logger, "Error draining expert queue", e)
                        break
            except Exception as e:
                log_exception(logger, "Error in background drain task", e)

        await asyncio.sleep(interval_seconds)
