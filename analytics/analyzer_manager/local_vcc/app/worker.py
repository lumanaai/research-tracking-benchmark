import functools
import json
import queue
import threading
import time

import pika
from pika.exceptions import AMQPConnectionError, ChannelClosed, ConnectionClosed, StreamLostError

from app.logger import logger, log_exception
from app.models import AlertVerificationRequest, AlertVerificationResponse
from app.verifier import AlertVerifier

CONNECTION_EXCEPTIONS = (
    ConnectionClosed,
    ChannelClosed,
    StreamLostError,
    AMQPConnectionError,
    pika.exceptions.ChannelWrongStateError,
)


class _PendingJob:
    __slots__ = ("delivery_tag", "request", "enqueued_at")

    def __init__(self, delivery_tag, request: AlertVerificationRequest, enqueued_at: float):
        self.delivery_tag = delivery_tag
        self.request = request
        self.enqueued_at = enqueued_at


class VccWorker:
    """RabbitMQ worker that verifies alerts using the local VLM.

    Consuming from RabbitMQ is decoupled from running the (slow, GPU-bound)
    VLM query so the broker connection keeps draining ``request_queue`` even
    while jobs are backed up:

    - The IO thread (running ``start_consuming``) parses/validates each
      message and drops it onto a bounded in-process queue (``_pending``),
      sized to ``max_timeout_seconds`` (assuming ~1s/job, that's roughly how
      many jobs could still meet their deadline). If ``_pending`` is full,
      the request is answered immediately with a failure response instead of
      being queued - it would already be guaranteed to miss its deadline.
    - A single worker thread (matching llama-server's ``--parallel 1``) pulls
      jobs off ``_pending`` in order, drops any that have already waited
      ``>= max_timeout_seconds`` since being accepted without doing any VLM
      work, and otherwise calls ``AlertVerifier.verify`` (whose own VLM calls
      are bounded by ``VlmClient``'s per-request timeout).
    - Every accepted job - whether it completes, times out waiting, or blows
      up - gets exactly one publish + ack, scheduled back onto the IO thread
      via ``connection.add_callback_threadsafe`` since pika's
      ``BlockingConnection`` is not thread-safe and must only be touched from
      the thread that owns it.

    Note: the per-job deadline is only checked when a job is *dequeued*, not
    continuously during ``verify()`` - an alert with many images can still
    overrun ``max_timeout_seconds`` once it starts processing.
    """

    def __init__(
        self,
        verifier: AlertVerifier,
        host: str,
        request_queue: str,
        response_queue: str,
        request_queue_ttl_ms: int,
        response_queue_ttl_ms: int,
        max_timeout_seconds: float = 30.0,
    ):
        self.verifier = verifier
        self.host = host
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.request_queue_ttl_ms = request_queue_ttl_ms
        self.response_queue_ttl_ms = response_queue_ttl_ms
        self.max_timeout_seconds = max_timeout_seconds
        self.connection = None
        self.channel = None
        self._should_stop = False

        queue_size = max(1, int(max_timeout_seconds))
        self._pending = queue.Queue(maxsize=queue_size)
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

    # ------------------------------------------------------------------
    # connection management
    # ------------------------------------------------------------------
    def _connect(self):
        params = pika.ConnectionParameters(host=self.host, heartbeat=60)
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
        request_args = {"x-message-ttl": int(self.request_queue_ttl_ms)}
        response_args = {"x-message-ttl": int(self.response_queue_ttl_ms)}
        self.channel.queue_declare(queue=self.request_queue, arguments=request_args)
        self.channel.queue_declare(queue=self.response_queue, arguments=response_args)
        # Unlimited prefetch: keep pulling from the broker as fast as it'll
        # send. `_pending` (bounded, see class docstring) is the real
        # backpressure mechanism now, not RabbitMQ's delivery throttling.
        self.channel.basic_qos(prefetch_count=0)
        self.channel.basic_consume(
            queue=self.request_queue,
            on_message_callback=self._on_message,
            auto_ack=False,
        )
        logger.info(
            f"Connected to RabbitMQ @ {self.host}; consuming from "
            f"{self.request_queue}, publishing to {self.response_queue} "
            f"(max_timeout_seconds={self.max_timeout_seconds}, "
            f"pending_queue_size={self._pending.maxsize})"
        )

    def _safe_close(self):
        try:
            if self.channel and self.channel.is_open:
                self.channel.close()
        except Exception:
            pass
        try:
            if self.connection and self.connection.is_open:
                self.connection.close()
        except Exception:
            pass
        self.channel = None
        self.connection = None

    # ------------------------------------------------------------------
    # IO-thread: message intake
    # ------------------------------------------------------------------
    def _on_message(self, channel, method, _properties, body):
        alert_instance_id = None
        camera_id = ""
        try:
            payload = json.loads(body)
            alert_instance_id = payload.get("alertInstanceId")
            camera_id = payload.get("cameraId", "")
            request = AlertVerificationRequest.model_validate(payload)
        except Exception as e:
            log_exception(logger, "Failed to process message", e)
            response = AlertVerificationResponse(
                alertInstanceId=alert_instance_id if alert_instance_id is not None else -1,
                cameraId=camera_id,
                verified=False,
                success=False,
                message=f"Exception while processing request: {e}",
            )
            self._publish_response(response)
            channel.basic_ack(delivery_tag=method.delivery_tag)
            return

        try:
            self._pending.put_nowait(_PendingJob(method.delivery_tag, request, time.time()))
        except queue.Full:
            logger.warning(
                f"Pending queue full (size={self._pending.maxsize}); alertInstanceId="
                f"{request.alertInstanceId} would already miss its deadline - rejecting"
            )
            response = AlertVerificationResponse(
                alertInstanceId=request.alertInstanceId,
                cameraId=request.cameraId,
                verified=False,
                success=False,
                message=f"Rejected: internal queue full (>{self._pending.maxsize} pending jobs)",
            )
            self._publish_response(response)
            channel.basic_ack(delivery_tag=method.delivery_tag)

    def _publish_response(self, response: AlertVerificationResponse):
        body = response.model_dump_json()
        try:
            self.channel.basic_publish(
                exchange="",
                routing_key=self.response_queue,
                body=body,
            )
            logger.info(f"Published response: {body}")
        except CONNECTION_EXCEPTIONS as e:
            logger.warning(f"Connection error while publishing response: {e}")
            raise

    def _finish_job(self, delivery_tag, response: AlertVerificationResponse):
        """Runs on the IO thread (scheduled via add_callback_threadsafe)."""
        try:
            self._publish_response(response)
        finally:
            try:
                self.channel.basic_ack(delivery_tag=delivery_tag)
            except Exception as e:
                logger.warning(f"Failed to ack delivery_tag={delivery_tag}: {e}")

    def _schedule_finish(self, delivery_tag, response: AlertVerificationResponse):
        connection = self.connection
        if connection is None or not connection.is_open:
            logger.warning(
                f"No open connection to publish response for delivery_tag={delivery_tag}; "
                f"RabbitMQ will redeliver this message once reconnected"
            )
            return
        try:
            connection.add_callback_threadsafe(
                functools.partial(self._finish_job, delivery_tag, response)
            )
        except Exception as e:
            logger.warning(f"Failed to schedule response publish: {e}")

    # ------------------------------------------------------------------
    # worker thread: VLM processing
    # ------------------------------------------------------------------
    def _worker_loop(self):
        while not self._should_stop:
            try:
                job = self._pending.get(timeout=1.0)
            except queue.Empty:
                continue

            waited = time.time() - job.enqueued_at
            if waited >= self.max_timeout_seconds:
                logger.warning(
                    f"alertInstanceId={job.request.alertInstanceId} timed out after "
                    f"{waited:.1f}s waiting in the internal queue "
                    f"(max_timeout_seconds={self.max_timeout_seconds})"
                )
                response = AlertVerificationResponse(
                    alertInstanceId=job.request.alertInstanceId,
                    cameraId=job.request.cameraId,
                    verified=False,
                    success=False,
                    message=f"Timed out after {waited:.1f}s waiting to be processed",
                )
            else:
                try:
                    response = self.verifier.verify(job.request)
                except Exception as e:
                    log_exception(logger, "Verifier raised unexpectedly", e)
                    response = AlertVerificationResponse(
                        alertInstanceId=job.request.alertInstanceId,
                        cameraId=job.request.cameraId,
                        verified=False,
                        success=False,
                        message=f"Exception while processing request: {e}",
                    )

            self._schedule_finish(job.delivery_tag, response)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self):
        if not self._worker_thread.is_alive():
            self._worker_thread.start()

        backoff = 1.0
        while not self._should_stop:
            try:
                self._connect()
                backoff = 1.0
                self.channel.start_consuming()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received, shutting down")
                self._should_stop = True
            except CONNECTION_EXCEPTIONS as e:
                logger.warning(
                    f"RabbitMQ connection lost: {e} - reconnecting in {backoff:.1f}s"
                )
                self._safe_close()
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except Exception as e:
                log_exception(logger, "Unexpected error in worker loop", e)
                self._safe_close()
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

        self._safe_close()
        self._worker_thread.join(timeout=5.0)
        logger.info("Worker stopped")

    def stop(self):
        self._should_stop = True
        try:
            if self.channel and self.channel.is_open:
                self.channel.stop_consuming()
        except Exception:
            pass
