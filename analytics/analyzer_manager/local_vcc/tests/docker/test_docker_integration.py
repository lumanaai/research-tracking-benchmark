"""Docker-level integration test for local_vcc.

Requires ``docker`` (with the compose plugin) available on PATH.
The test:

1. Builds the local_vcc image (tagged ``lumixai/local_vcc:test``) from the
   repo-root Dockerfile.
2. Brings up rabbitmq + a stub OpenAI server + the local_vcc worker via
   ``docker compose``.
3. Publishes an AlertVerificationRequest to the request queue.
4. Waits for a matching AlertVerificationResponse on the response queue.

Skipped automatically when docker is unavailable so it doesn't break
plain ``pytest`` runs on dev machines without docker.

Run manually::

    pytest analyzer_manager/local_vcc/tests/docker/test_docker_integration.py -v -s
"""
import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from PIL import Image

pika = pytest.importorskip("pika")


HERE = Path(__file__).resolve().parent
COMPOSE_FILE = HERE / "docker-compose.test.yml"
ANALYZER_MANAGER_DIR = HERE.parents[2]              # .../analyzer_manager
DOCKERFILE = ANALYZER_MANAGER_DIR / "Dockerfile.local_vcc"
IMAGE_TAG = "lumixai/local_vcc:test"
PROJECT_NAME = "local_vcc_it"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="docker daemon not available; skipping docker-level tests",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(cmd, env=None, check=True, capture=False):
    print("+", " ".join(cmd), flush=True)
    kwargs = {"env": env}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    result = subprocess.run(cmd, **kwargs)
    if capture:
        print(result.stdout.decode(errors="replace"))
        print(result.stderr.decode(errors="replace"))
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return result


@pytest.fixture(scope="module")
def built_image():
    _run(
        [
            "docker", "build",
            "-f", str(DOCKERFILE),
            "-t", IMAGE_TAG,
            str(ANALYZER_MANAGER_DIR),
        ]
    )
    return IMAGE_TAG


@pytest.fixture
def shared_images_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    Image.new("RGB", (16, 16), (255, 0, 0)).save(d / "alert.jpg", "JPEG")
    return d


def _wait_for_rabbit(host, port, timeout=60):
    end = time.time() + timeout
    last_err = None
    while time.time() < end:
        try:
            conn = pika.BlockingConnection(
                pika.ConnectionParameters(host=host, port=port, socket_timeout=2)
            )
            conn.close()
            return
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"rabbitmq not reachable on {host}:{port}: {last_err}")


@pytest.fixture
def compose_stack(built_image, shared_images_dir):
    rabbit_port = _free_port()
    env = os.environ.copy()
    env.update({
        "RABBIT_PORT": str(rabbit_port),
        "SHARED_IMAGES_DIR": str(shared_images_dir),
        "STUB_VERDICT": env.get("STUB_VERDICT", "yes"),
    })

    base = ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE)]
    _run(base + ["up", "-d", "--build"], env=env)
    try:
        _wait_for_rabbit("127.0.0.1", rabbit_port)
        # Give the worker a moment to declare queues + start consuming.
        time.sleep(3)
        yield {"rabbit_port": rabbit_port, "compose_base": base, "env": env}
    finally:
        _run(base + ["logs", "local_vcc"], env=env, check=False, capture=True)
        _run(base + ["down", "-v", "--remove-orphans"], env=env, check=False)


def _publish_and_await(rabbit_port, request, response_timeout=30):
    conn = pika.BlockingConnection(pika.ConnectionParameters(host="127.0.0.1", port=rabbit_port))
    ch = conn.channel()
    args = {"x-message-ttl": 60000}
    ch.queue_declare(queue="vccVerificationRequestQueue", arguments=args)
    ch.queue_declare(queue="vccVerificationResponseQueue", arguments=args)
    # Drain any stale responses.
    while ch.basic_get(queue="vccVerificationResponseQueue", auto_ack=True)[0] is not None:
        pass

    ch.basic_publish(
        exchange="",
        routing_key="vccVerificationRequestQueue",
        body=json.dumps(request).encode(),
    )

    deadline = time.time() + response_timeout
    while time.time() < deadline:
        method, _props, body = ch.basic_get(queue="vccVerificationResponseQueue", auto_ack=True)
        if method is not None:
            conn.close()
            return json.loads(body)
        time.sleep(0.5)
    conn.close()
    raise AssertionError("Timed out waiting for AlertVerificationResponse")


def test_verifies_alert_yes(compose_stack):
    # Safety category (0) * 1_000_000 + Weapon flow (3) = 3
    request = {
        "alertInstanceId": 12345,
        "cameraId": "cam-1",
        "eventTypeId": 3,
        "images": ["/data/images/alert.jpg"],
    }
    response = _publish_and_await(compose_stack["rabbit_port"], request)
    assert response["alertInstanceId"] == 12345
    assert response["cameraId"] == "cam-1"
    assert response["success"] is True
    assert response["verified"] is True


def test_missing_image_reports_failure(compose_stack):
    request = {
        "alertInstanceId": 999,
        "cameraId": "cam-2",
        "eventTypeId": 3,
        "images": ["/data/images/nonexistent.jpg"],
    }
    response = _publish_and_await(compose_stack["rabbit_port"], request)
    assert response["alertInstanceId"] == 999
    assert response["cameraId"] == "cam-2"
    assert response["verified"] is False
    assert response["success"] is False


def test_unknown_event_without_filter_prompt(compose_stack):
    request = {
        "alertInstanceId": 42,
        "cameraId": "cam-3",
        "eventTypeId": 999_000_001,  # unmapped
        "images": ["/data/images/alert.jpg"],
    }
    response = _publish_and_await(compose_stack["rabbit_port"], request)
    assert response["alertInstanceId"] == 42
    assert response["cameraId"] == "cam-3"
    assert response["verified"] is False
    assert response["success"] is False


def test_malformed_request_publishes_failure(compose_stack):
    conn = pika.BlockingConnection(
        pika.ConnectionParameters(host="127.0.0.1", port=compose_stack["rabbit_port"])
    )
    ch = conn.channel()
    args = {"x-message-ttl": 60000}
    ch.queue_declare(queue="vccVerificationRequestQueue", arguments=args)
    ch.queue_declare(queue="vccVerificationResponseQueue", arguments=args)
    while ch.basic_get(queue="vccVerificationResponseQueue", auto_ack=True)[0] is not None:
        pass
    ch.basic_publish(
        exchange="",
        routing_key="vccVerificationRequestQueue",
        body=b"{not json",
    )

    deadline = time.time() + 15
    payload = None
    while time.time() < deadline:
        method, _props, body = ch.basic_get(queue="vccVerificationResponseQueue", auto_ack=True)
        if method is not None:
            payload = json.loads(body)
            break
        time.sleep(0.3)
    conn.close()
    assert payload is not None, "no response for malformed request"
    assert payload["success"] is False
    assert payload["verified"] is False

