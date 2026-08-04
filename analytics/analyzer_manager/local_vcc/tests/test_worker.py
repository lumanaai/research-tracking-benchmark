"""Unit tests for VccWorker message handling (no real RabbitMQ).

``_on_message`` now only parses/validates and enqueues onto the internal
``_pending`` queue (or answers immediately on malformed payload / full
queue); the actual `verify()` call + response publish happens on a
background thread. Tests that exercise that path start the real
``_worker_thread`` and poll for the (mocked) channel to observe the
publish, mirroring how ``connection.add_callback_threadsafe`` hands the
publish/ack back to the IO thread in production.
"""
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.models import AlertVerificationRequest, AlertVerificationResponse
from app.worker import VccWorker, _PendingJob


class FakeVerifier:
    def __init__(self, response=None, raise_exc=None):
        self.response = response
        self.raise_exc = raise_exc
        self.calls = []

    def verify(self, request: AlertVerificationRequest) -> AlertVerificationResponse:
        self.calls.append(request)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


def _make_worker(verifier, max_timeout_seconds=30.0):
    worker = VccWorker(
        verifier=verifier,
        host="localhost",
        request_queue="req",
        response_queue="resp",
        request_queue_ttl_ms=1000,
        response_queue_ttl_ms=1000,
        max_timeout_seconds=max_timeout_seconds,
    )
    worker.channel = MagicMock()
    worker.connection = MagicMock()
    worker.connection.is_open = True
    # In production, add_callback_threadsafe hands the callback to pika's IO
    # loop, which runs it on the connection's thread. There's no real IO loop
    # in these tests, and the callback only touches the (mocked) channel, so
    # it's safe to just run it inline from whichever thread calls it.
    worker.connection.add_callback_threadsafe.side_effect = lambda cb: cb()
    return worker


def _method(delivery_tag=42):
    return SimpleNamespace(delivery_tag=delivery_tag)


def _wait_for(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def _run_with_worker_thread(worker, fn):
    """Start the background worker thread, run fn(), then stop the thread."""
    worker._worker_thread.start()
    try:
        fn()
    finally:
        worker._should_stop = True
        worker._worker_thread.join(timeout=2.0)


def test_on_message_publishes_response_and_acks():
    resp = AlertVerificationResponse(alertInstanceId=1, cameraId="cam-1", verified=True, success=True, message="ok")
    verifier = FakeVerifier(response=resp)
    worker = _make_worker(verifier)

    body = json.dumps({
        "alertInstanceId": 1,
        "cameraId": "cam-1",
        "eventTypeId": 1000003,  # AlertCategory.Safety * 1M + Weapon
        "images": ["/tmp/img.jpg"],
    })

    def _do():
        worker._on_message(worker.channel, _method(), None, body.encode())
        _wait_for(lambda: worker.channel.basic_publish.called)

    _run_with_worker_thread(worker, _do)

    kwargs = worker.channel.basic_publish.call_args.kwargs
    assert kwargs["routing_key"] == "resp"
    published = json.loads(kwargs["body"])
    assert published == {"alertInstanceId": 1, "cameraId": "cam-1", "verified": True, "success": True, "message": "ok"}
    worker.channel.basic_ack.assert_called_once_with(delivery_tag=42)


def test_on_message_publishes_failure_on_verifier_exception():
    verifier = FakeVerifier(raise_exc=RuntimeError("kaboom"))
    worker = _make_worker(verifier)

    body = json.dumps({"alertInstanceId": 7, "cameraId": "cam-7", "eventTypeId": 1000003})

    def _do():
        worker._on_message(worker.channel, _method(), None, body.encode())
        _wait_for(lambda: worker.channel.basic_publish.called)

    _run_with_worker_thread(worker, _do)

    published = json.loads(worker.channel.basic_publish.call_args.kwargs["body"])
    assert published["alertInstanceId"] == 7
    assert published["cameraId"] == "cam-7"
    assert published["success"] is False
    assert published["verified"] is False
    assert "kaboom" in published["message"]
    worker.channel.basic_ack.assert_called_once()


def test_on_message_publishes_failure_on_malformed_payload():
    """Malformed payloads never touch the internal queue - answered inline,
    synchronously, on the IO thread itself, exactly as before."""
    verifier = FakeVerifier(response=None)
    worker = _make_worker(verifier)

    worker._on_message(worker.channel, _method(), None, b"not json")

    published = json.loads(worker.channel.basic_publish.call_args.kwargs["body"])
    assert published["success"] is False
    assert published["verified"] is False
    assert published["alertInstanceId"] == -1
    assert published["cameraId"] == ""  # fallback when payload is unparseable
    assert verifier.calls == []
    worker.channel.basic_ack.assert_called_once()
    assert worker._pending.qsize() == 0


def test_on_message_rejects_immediately_when_pending_queue_full():
    """A full internal queue means any new job would already miss its
    deadline - answer it immediately instead of queueing or calling verify."""
    verifier = FakeVerifier(response=None)
    worker = _make_worker(verifier, max_timeout_seconds=1.0)  # -> pending maxsize=1
    # Fill the only slot without starting the worker thread, so it stays full.
    worker._pending.put_nowait(_PendingJob(delivery_tag=1, request=None, enqueued_at=time.time()))

    body = json.dumps({"alertInstanceId": 99, "cameraId": "cam-99", "eventTypeId": 1000003})
    worker._on_message(worker.channel, _method(delivery_tag=2), None, body.encode())

    published = json.loads(worker.channel.basic_publish.call_args.kwargs["body"])
    assert published["alertInstanceId"] == 99
    assert published["cameraId"] == "cam-99"
    assert published["success"] is False
    assert published["verified"] is False
    assert "queue full" in published["message"]
    worker.channel.basic_ack.assert_called_once_with(delivery_tag=2)
    assert verifier.calls == []


def test_worker_loop_times_out_stale_job_without_calling_verifier():
    """A job that already waited past max_timeout_seconds gets a timeout
    response without ever reaching the (slow) verifier."""
    verifier = FakeVerifier(response=AlertVerificationResponse(
        alertInstanceId=5, cameraId="cam-5", verified=True, success=True, message="should not be used",
    ))
    worker = _make_worker(verifier, max_timeout_seconds=0.05)

    stale_request = AlertVerificationRequest(
        alertInstanceId=5, cameraId="cam-5", eventTypeId=1000003, images=["/tmp/img.jpg"],
    )
    worker._pending.put_nowait(_PendingJob(
        delivery_tag=3, request=stale_request, enqueued_at=time.time() - 10.0,
    ))

    _run_with_worker_thread(worker, lambda: _wait_for(lambda: worker.channel.basic_publish.called))

    published = json.loads(worker.channel.basic_publish.call_args.kwargs["body"])
    assert published["alertInstanceId"] == 5
    assert published["cameraId"] == "cam-5"
    assert published["success"] is False
    assert published["verified"] is False
    assert "Timed out" in published["message"]
    worker.channel.basic_ack.assert_called_once_with(delivery_tag=3)
    assert verifier.calls == []
