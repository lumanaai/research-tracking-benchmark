import json
import logging
import logging.handlers
import os
import threading
import time
from dataclasses import dataclass
from multiprocessing import Process, Event
from queue import Queue, Full
from threading import Thread
from typing import List

import zmq

from general.analyzer_general import log_exception
from general.offline_analytics import get_offline_storage_path
from general.proj import load_configAnalytic


@dataclass
class MessageData:
    ident: bytes
    meta: dict
    image_bytes: bytes


class OfflineAnalyzerWorker(Process):
    name = "OfflineAnalyzerWorker"
    min_batch_size = 2
    max_batch_size = 8
    max_hold_time = 5.0  # seconds
    use_files = False

    def __init__(self, ipc_socket, timeout_ms=500, num_workers=4, max_queue_size=20, hwm_per_client=10, log_queue=None):
        super().__init__()
        self.ipc_socket = ipc_socket
        self.timeout_ms = timeout_ms
        self.num_workers = num_workers
        self.hwm_per_client = hwm_per_client

        self.analytic_config = load_configAnalytic() or {}
        self.stop_event = threading.Event()

        # Thread references for proper cleanup
        self.io_thread = None
        self.worker_thread = None

        # Queues for communication between IO thread and worker threads
        self.response_queue = Queue(maxsize=max_queue_size * 3)

        # immediate and awaiting jobs
        self.jobs = {True: Queue(maxsize=max_queue_size), False: Queue(maxsize=int(max_queue_size * 1.5))}
        self.last_inserted_ts = 0
        self.metrics = {"received": 0, "processed": 0, "sent": 0, "dropped": 0, "duration": 0.0}
        self.logger = self.init_logger(log_queue) if log_queue else logging.getLogger(self.name)

        self.flush_event = Event()
        # Signal that the IO thread has successfully bound the socket and is ready
        self.ready_event = Event()

        self.storage_path = get_offline_storage_path()

        # Store the stripped socket path for cleanup
        self.sock_path = (
            self.ipc_socket.replace("ipc://", "") if self.ipc_socket.startswith("ipc://") else self.ipc_socket
        )

    def init_logger(self, queue):
        logger = logging.getLogger(self.name)
        # Clear any existing handlers to avoid duplicates
        logger.handlers = []
        # Add ONLY the QueueHandler
        logger.addHandler(logging.handlers.QueueHandler(queue))
        # Ensure level is low enough to capture everything you want
        logger.setLevel(logging.INFO)
        return logger

    def io_thread_func(self):
        """IO thread - handles ZMQ socket communication"""
        try:
            self._io_thread_inner()
        except Exception as e:
            # This catches any fatal exception that would silently kill the thread
            print(f"[FATAL] {self.name} IO thread died: {type(e).__name__}: {e}", flush=True)
            log_exception(self.logger, f"FATAL: IO thread died", e)

    def _io_thread_inner(self):
        """Actual IO thread logic"""
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)

        # Set HWM BEFORE bind so it takes effect for all connections
        socket.setsockopt(zmq.RCVHWM, self.hwm_per_client)
        socket.setsockopt(zmq.SNDHWM, self.hwm_per_client)

        # Clean up existing socket file
        old_umask = os.umask(0o000)  # rwxrwxr-x
        try:
            try:
                os.unlink(self.sock_path)
                self.logger.info(f"Removed stale socket file: {self.sock_path}")
            except FileNotFoundError:
                pass
            except Exception as e:
                self.logger.warning(f"Failed to unlink socket file {self.sock_path}: {e}")
            socket.bind(self.ipc_socket)
            self.logger.info(f"Server started on {self.ipc_socket}")
        finally:
            os.umask(old_umask)

        # Signal that socket is bound and ready to accept connections
        self.ready_event.set()

        # Poller for timeout
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        last_cleanup_time = time.time()
        last_heartbeat_time = time.time()
        try:
            while not self.stop_event.is_set():
                try:
                    # Poll for incoming messages
                    socks = dict(poller.poll(timeout=self.timeout_ms))

                    if socket in socks and socks[socket] == zmq.POLLIN:
                        # Receive multipart: [Identity, JSON, ImageBytes]
                        ident, meta_json, image_bytes = socket.recv_multipart()
                        self.metrics["received"] += 1
                        # Parse JSON metadata
                        meta = json.loads(meta_json)
                        immediate = meta.get("immediate", False)

                        # Put request in queue for worker threads (non-blocking to prevent IO thread deadlock)
                        try:
                            self.jobs[immediate].put_nowait(MessageData(ident, meta, image_bytes))
                            if not immediate:
                                self.last_inserted_ts = time.time()
                        except Full:
                            self.metrics["dropped"] += 1
                            self.logger.warning(
                                f"Job queue full, dropping {self.name} job: {meta.get('camera_id', 'unknown')} ts: {meta.get('timestamp', 'unknown')}"
                            )

                        self.logger.debug(f"Received request from {meta.get('camera_id', 'unknown')}")

                    # Check for responses to send back
                    while not self.response_queue.empty():
                        self.metrics["sent"] += 1
                        ident, result = self.response_queue.get_nowait()
                        socket.send_multipart([ident, json.dumps(result).encode()])
                        self.logger.debug(f"Sent response for {result.get('cam_id', 'unknown')}")

                except zmq.ZMQError as e:
                    if not self.stop_event.is_set():
                        log_exception(self.logger, "ZMQ Error", e)

                except Exception as e:
                    if not self.stop_event.is_set():
                        log_exception(self.logger, "IO thread exception", e)
                try:
                    if self.flush_event.is_set():
                        self.flush()
                except Exception as e:
                    log_exception(self.logger, "Error during flush", e)
                finally:
                    self.flush_event.clear()

                # Periodic heartbeat log to detect thread stalls
                if time.time() - last_heartbeat_time > 60:
                    self.logger.info(
                        f"[heartbeat] {self.name} IO thread alive | "
                        f"received={self.metrics['received']} sent={self.metrics['sent']} "
                        f"dropped={self.metrics['dropped']} "
                        f"job_q_imm={self.jobs[True].qsize()} job_q_reg={self.jobs[False].qsize()} "
                        f"resp_q={self.response_queue.qsize()}"
                    )
                    last_heartbeat_time = time.time()

                if self.use_files and time.time() - last_cleanup_time > 60:  # every 60 seconds
                    try:
                        # find all files older than last_cleanup_time in storage path
                        for filename in os.listdir(self.storage_path):
                            file_path = os.path.join(self.storage_path, filename)
                            if os.path.isfile(file_path):
                                file_mtime = os.path.getmtime(file_path)
                                if file_mtime < last_cleanup_time:
                                    self.logger.info(
                                        f"Deleting file {file_path} during cleanup; mtime {file_mtime} last_cleanup_time {last_cleanup_time}"
                                    )
                                    try:
                                        os.unlink(file_path)
                                    except Exception:
                                        self.logger.warning(f"Failed to delete file {file_path} during cleanup")
                                else:
                                    self.logger.info(
                                        f"File {file_path} was not deleted during cleanup; it is still recent: mtime {file_mtime} last_cleanup_time {last_cleanup_time}"
                                    )

                    except Exception as e:
                        log_exception(self.logger, "Error during periodic cleanup", e)
                    finally:
                        last_cleanup_time = time.time()

        finally:
            # Clean up ZMQ resources
            self.logger.info("Cleaning up ZMQ resources")
            poller.unregister(socket)
            socket.close()
            context.term()
            # Clean up socket file using the correct stripped path
            try:
                os.unlink(self.sock_path)
            except Exception as e:
                pass

    def _handle_job_list(self, job_list: List[MessageData]) -> List:
        return []

    def worker_thread_func(self):
        try:
            self._worker_thread_inner()
        except Exception as e:
            print(f"[FATAL] {self.name} worker thread died: {type(e).__name__}: {e}", flush=True)
            log_exception(self.logger, f"FATAL: Worker thread died", e)

    def _worker_thread_inner(self):
        last_heartbeat = time.time()
        while not self.stop_event.is_set():
            try:
                job_list = []
                while not self.jobs[True].empty():
                    job_list.append(self.jobs[True].get_nowait())

                if (
                    self.jobs[False].qsize() >= self.min_batch_size
                    or time.time() - self.last_inserted_ts > self.max_hold_time
                    or self.max_batch_size > len(job_list) > 0
                ):
                    while not self.jobs[False].empty() and len(job_list) < self.max_batch_size:
                        job_list.append(self.jobs[False].get_nowait())

                if job_list:
                    t_start = time.time()
                    results = self._handle_job_list(job_list)
                    duration = time.time() - t_start
                    if duration > 30:
                        self.logger.warning(
                            f"[{self.name}] _handle_job_list took {duration:.1f}s for {len(job_list)} jobs"
                        )
                    for ident, result in results:
                        try:
                            self.response_queue.put_nowait((ident, result))
                        except Full:
                            self.logger.warning(f"Response queue full, dropping response for {self.name}")
                else:
                    time.sleep(0.5)

                # Worker thread heartbeat
                if time.time() - last_heartbeat > 60:
                    self.logger.info(
                        f"[heartbeat] {self.name} worker thread alive | "
                        f"processed={self.metrics['processed']} avg_duration={self.metrics['duration']:.2f}s"
                    )
                    last_heartbeat = time.time()
            except Exception as e:
                log_exception(self.logger, "Worker processing error", e)

    def run(self):
        """Main process entry point"""
        self.logger.info(f"Starting {self.name} process with {self.num_workers} workers")

        # Start IO thread
        self.io_thread = Thread(target=self.io_thread_func, daemon=False)
        self.io_thread.start()

        # Start worker dispatcher thread
        self.worker_thread = Thread(target=self.worker_thread_func, daemon=False)
        self.worker_thread.start()

        # Wait for stop signal
        while not self.stop_event.is_set():
            time.sleep(0.5)

        # Wait for threads to finish
        self.logger.info("Waiting for threads to complete...")
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3)
        if self.io_thread and self.io_thread.is_alive():
            self.io_thread.join(timeout=3)

        self.logger.info(f"{self.name} process stopped")

    def start_worker(self):
        """Helper method to start the worker process"""
        self.start()

    def stop(self):
        """Stop the worker process and all threads gracefully"""
        if self.stop_event.is_set():
            self.logger.warning(f"{self.name} is already stopped or stopping")
            return

        self.logger.info(f"Stopping {self.name}...")
        self.stop_event.set()

        # If called from another process, terminate the process
        if self.is_alive():
            self.logger.info(f"Waiting for {self.name} process to terminate...")
            self.join(timeout=3)
            if self.is_alive():
                self.logger.warning(f"{self.name} did not terminate gracefully, forcing termination")
                self.terminate()
                self.join(timeout=2)

        self.logger.info(f"{self.name} stopped successfully")

    def get_metrics(self):
        return self.metrics

    def flush(self):
        """Flush all queues"""
        try:
            while not self.response_queue.empty():
                self.response_queue.get_nowait()
            for key in self.jobs:
                while not self.jobs[key].empty():
                    self.jobs[key].get_nowait()
            self.logger.info(f"Flushed all queues for {self.name}")
        except Exception as e:
            log_exception(self.logger, f"Error flushing queues for {self.name}", e)
        self.metrics = {"received": 0, "processed": 0, "sent": 0, "dropped": 0, "duration": 0.0}

    def _update_exec_metrics(self, duration: float, n_jobs: int = 1):
        processed = self.metrics["processed"]
        self.metrics["processed"] += n_jobs
        self.metrics["duration"] = (processed * self.metrics["duration"] + duration) / self.metrics["processed"]
