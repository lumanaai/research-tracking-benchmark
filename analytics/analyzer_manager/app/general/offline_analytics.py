import base64
import json
import os
import socket
import time
import zlib
from collections import deque
from copy import copy
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import requests
import zmq

from detection.yolov5.yolov8_detector import YoloV8ExpertArgs
from general import proj
from general.analyzer_general import log_exception, logger
from general.core import AnalyticImage
from preprocessing.snapshot_resizer import SnapshotResizerFactory

MAX_FLEXIBILITY = 5000  # milliseconds
DEFAULT_TIMEOUT = 0.2


class TimestampResultCache:
    def __init__(self, max_size=8):
        self._timestamps = deque()  # Ordered deque of timestamps
        self._results = {}  # Map timestamp -> results
        self._latest = None
        self._max_size = max_size

    def insert(self, timestamp: int, results: Optional[Dict]):
        if timestamp not in self._results:
            self._timestamps.append(timestamp)
            self._results[timestamp] = results
            if results is not None:
                self._latest = timestamp

            # Enforce max size: remove oldest if needed
            while len(self._timestamps) > self._max_size:
                oldest = self._timestamps.popleft()
                self._results.pop(oldest)
        else:
            # Optionally update results if timestamp already exists
            self._results[timestamp] = results

    def query(self, timestamp, flexibility) -> Tuple[bool, Optional[Dict]]:
        """
        Return results for the closest timestamp if within flexibility (same units as timestamp).
        Otherwise, return None.
        """
        found = False
        if not self._timestamps:
            return found, None
        ts_array = np.array(self._timestamps)
        idx = np.argmin(np.abs(ts_array - timestamp))
        closest = ts_array[idx]
        found = abs(closest - timestamp) <= flexibility
        return found, self._results[closest]

    @property
    def latest(self):
        return self._latest

    def __len__(self):
        return len(self._timestamps)

    def clear(self):
        self._timestamps.clear()
        self._results.clear()
        self._latest = None


def check_expert_availability(offline_analytics_settings) -> Optional[str]:
    try:
        expert_url = offline_analytics_settings.get("uri", None)
        if expert_url is None:
            logger.error("Expert detector is enabled but no uri is provided. Disabling expert detector")
        else:
            protocol = offline_analytics_settings.get("protocol", "http")
            if protocol == "http":
                expert_url = f"http://{expert_url}"
            return expert_url
    except Exception as e:
        log_exception(logger, "could not use expert detector", e)
    return None


def decode_image_from_base64(image: str) -> np.ndarray:
    image_data = base64.b64decode(image)
    np_array = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)


def encode_image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode()


def compress_image(image) -> bytes:
    image_bytes = image.tobytes()
    return zlib.compress(image_bytes)


def flush_server_queues(server_address: str):
    """Flush all server queues before running tests"""
    url = f"{server_address}/flush_queues"
    try:
        response = requests.post(url, timeout=10)
        if response.status_code != 200:
            logger.error(f"Server queue flush failed with status {response.status_code}")
            return False
        result = response.json()
        if result.get("status") == "ok":
            logger.info("Server queues flushed successfully")
            return True
        logger.error(f"Server queue flush failed with status {response.status_code}")
        return False
    except requests.exceptions.RequestException as e:
        log_exception(logger, "Could not connect to server for queue flush:", e)
        return False
    except Exception as e:
        log_exception(logger, "Unexpected error during queue flush:", e)
        return False


def resolve_host_to_ip(host: str) -> Optional[str]:
    """
    Resolve a hostname to its IP address.

    Args:
        host: The hostname to resolve (e.g., 'example.com' or 'localhost')

    Returns:
        IP address as a string, or None if resolution fails
    """
    try:
        ip_address = socket.gethostbyname(host)
        return ip_address
    except socket.gaierror as e:
        log_exception(logger, f"Failed to resolve hostname '{host}':", e)
        return None
    except Exception as e:
        log_exception(logger, f"Unexpected error resolving hostname '{host}':", e)
        return None


def check_multiple_messages(sock) -> List[Dict]:
    messages = []
    while True:
        try:
            frames = sock.recv_multipart(zmq.DONTWAIT)
            # Parse the JSON payload (first frame)
            data = json.loads(frames[0].decode("utf-8"))
            messages.append(data)
        except zmq.Again:
            break
    return messages


def get_offline_storage_path() -> str:
    config = proj.load_config()
    return config.get("locations", {}).get("offlineImages", "/dev/shm/offlineImages")


def get_ipc_sockets(make_dir: bool = False) -> Tuple[str, str, str]:
    config = proj.load_config()
    base_path = config.get("locations", {}).get("analyticSockets", "/tmp/analyticSockets")
    if make_dir:
        os.makedirs(base_path, mode=0o777, exist_ok=True)
    clip_socket = f"ipc://{base_path}/clip.ipc"
    expert_socket = f"ipc://{base_path}/expert.ipc"
    weapon_socket = f"ipc://{base_path}/weapon.ipc"
    return clip_socket, expert_socket, weapon_socket


class OfflineAnalyticsClient:
    clip_socket: zmq.Socket
    expert_socket: zmq.Socket
    weapon_socket: zmq.Socket
    poller: zmq.Poller
    send_hwm_per_client = 2
    recv_hwm_per_client = 10
    clip_socket_file: str  #  "ipc:///dev/shm/analyticSockets/clip.ipc"
    expert_socket_file: str  #  "ipc:///dev/shm/analyticSockets/expert.ipc"
    weapon_socket_file: str  #  "ipc:///dev/shm/analyticSockets/weapon.ipc"

    def __init__(self, server_address: str, camera_id: str):
        self.original_address = server_address
        self.server_address = self.original_address
        self.default_timeout = DEFAULT_TIMEOUT
        self.camera_id = camera_id
        self.expert_cache = TimestampResultCache(max_size=8)
        self.weapon_cache = TimestampResultCache(max_size=8)

        self._init_session()
        self._retries = 0
        self.max_retries = 3
        self.last_sync_sec = 0
        assert self.check_server_health(), "Expert server is not healthy during initialization"
        self.init_queues()

        self.clip_results = deque(maxlen=self.recv_hwm_per_client * 2)
        self.weapon_results = deque(maxlen=self.recv_hwm_per_client)

        self.expert_args = YoloV8ExpertArgs({})
        self.detector_img_size = tuple(self.expert_args.im_size)
        self.resizer = SnapshotResizerFactory.get_resizer(self.camera_id, self.detector_img_size)
        self.resizer.antialias = self.expert_args.antialias
        self.storage_path = get_offline_storage_path()

    def init_queues(self):
        self.clip_socket_file, self.expert_socket_file, self.weapon_socket_file = get_ipc_sockets()
        self._zmq_context = zmq.Context()

        # Socket for Server 1 (HWM must be set BEFORE connect to take effect)
        socket1 = self._zmq_context.socket(zmq.DEALER)
        socket1.setsockopt_string(zmq.IDENTITY, "clip_" + self.camera_id)
        socket1.setsockopt(zmq.SNDHWM, self.send_hwm_per_client)
        socket1.setsockopt(zmq.RCVHWM, self.recv_hwm_per_client)
        socket1.connect(self.clip_socket_file)
        self.clip_socket = socket1

        # Socket for Server 2
        socket2 = self._zmq_context.socket(zmq.DEALER)
        socket2.setsockopt_string(zmq.IDENTITY, "expert_" + self.camera_id)
        socket2.setsockopt(zmq.SNDHWM, self.send_hwm_per_client)
        socket2.setsockopt(zmq.RCVHWM, self.recv_hwm_per_client)
        socket2.connect(self.expert_socket_file)
        self.expert_socket = socket2

        # Socket for Server 3
        socket3 = self._zmq_context.socket(zmq.DEALER)
        socket3.setsockopt_string(zmq.IDENTITY, "weapon_" + self.camera_id)
        socket3.setsockopt(zmq.SNDHWM, self.send_hwm_per_client)
        socket3.setsockopt(zmq.RCVHWM, self.recv_hwm_per_client)
        socket3.connect(self.weapon_socket_file)
        self.weapon_socket = socket3

        # Poller for all sockets
        self.poller = zmq.Poller()
        self.poller.register(socket1, zmq.POLLIN)
        self.poller.register(socket2, zmq.POLLIN)
        self.poller.register(socket3, zmq.POLLIN)

        # Track socket file inodes to detect server restarts
        self._socket_inodes = self._get_socket_inodes()

        logger.info(f"Client {self.camera_id} connected to {self.clip_socket_file} and {self.expert_socket_file}")

    def _get_socket_inodes(self) -> dict:
        """Get current inodes of socket files to detect server restarts."""
        inodes = {}
        for name, path in [
            ("clip", self.clip_socket_file),
            ("expert", self.expert_socket_file),
            ("weapon", self.weapon_socket_file),
        ]:
            sock_path = path.replace("ipc://", "") if path.startswith("ipc://") else path
            try:
                stat = os.stat(sock_path)
                inodes[name] = stat.st_ino
            except (FileNotFoundError, OSError):
                inodes[name] = None
        return inodes

    def _check_socket_inodes_changed(self) -> bool:
        """Check if socket file inodes changed (server restarted and re-created them)."""
        current = self._get_socket_inodes()
        for name in current:
            if self._socket_inodes.get(name) is not None and current[name] != self._socket_inodes.get(name):
                logger.warning(
                    f"Socket inode changed for {name}: {self._socket_inodes.get(name)} -> {current[name]}. "
                    f"Server likely restarted."
                )
                return True
        return False

    def reconnect_queues(self):
        """Tear down and re-create ZMQ sockets (e.g., after detecting server restart)."""
        logger.info(f"Reconnecting ZMQ sockets for camera {self.camera_id}...")
        self._close_zmq()
        self.init_queues()
        self._retries = 0
        logger.info(f"ZMQ sockets reconnected for camera {self.camera_id}")

    def sync(self) -> bool:
        success = True
        try:
            # Check for messages
            socks = dict(self.poller.poll(timeout=0))  # immediate return

            if self.clip_socket in socks and socks[self.clip_socket] == zmq.POLLIN:
                # Message received from clip worker
                messages = check_multiple_messages(self.clip_socket)
                # handle clip messages
                self.clip_results.extend([copy(m) for m in messages])

            if self.expert_socket in socks and socks[self.expert_socket] == zmq.POLLIN:
                messages = check_multiple_messages(self.expert_socket)
                for message in messages:
                    self.expert_cache.insert(message["timestamp"], message)

            if self.weapon_socket in socks and socks[self.weapon_socket] == zmq.POLLIN:
                # Message received from Server 2
                messages = check_multiple_messages(self.weapon_socket)
                for message in messages:
                    if "weapon" in message:
                        alert = message["weapon"].copy()
                        alert["zoom_crop"] = cv2.cvtColor(
                            decode_image_from_base64(alert["zoom_crop"]), cv2.COLOR_BGR2RGB
                        )
                        alert["validation_crop"] = cv2.cvtColor(
                            decode_image_from_base64(alert["validation_crop"]), cv2.COLOR_BGR2RGB
                        )
                        alert["snapshot"] = decode_image_from_base64(alert["snapshot"])
                        self.weapon_results.append(alert)
                    self.weapon_cache.insert(message["timestamp"], message)
            self.last_sync_sec = int(time.time())

        except Exception as e:
            log_exception(logger, "An error occurred during zmq sync:", e)
            success = False
            self._retries += 1
        return success

    def _init_session(self):
        host = self.original_address.replace("http://", "").replace("https://", "").split(":")[0]
        resolved_ip = resolve_host_to_ip(host)

        if resolved_ip:
            # Replace hostname with IP in server_address
            self.server_address = self.original_address.replace(host, resolved_ip)
            logger.info(f"Resolved {host} to {resolved_ip}")
        else:
            # Fallback to original hostname
            logger.warning(f"Could not resolve {host}, using hostname directly")

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=2,
            pool_maxsize=10,
            max_retries=0,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _close_zmq(self):
        """Close ZMQ sockets and context."""
        try:
            self.poller.unregister(self.clip_socket)
            self.poller.unregister(self.expert_socket)
            self.poller.unregister(self.weapon_socket)
            self.clip_socket.close()
            self.expert_socket.close()
            self.weapon_socket.close()
            self._zmq_context.term()
        except Exception as e:
            log_exception(logger, "Error closing ZMQ resources", e)

    def close(self):
        self.session.close()
        self._close_zmq()

    def prepare_batch(self) -> bool:
        self.weapon_results.clear()
        return self.sync()

    def get_active_weapon_alerts(self) -> Optional[List]:
        alerts = [res.copy() for res in self.weapon_results]
        if alerts:
            logger.info(f"received {len(alerts)} weapon alerts")
        return alerts

    def _internal_send_socket(self, socket: zmq.Socket, payload: dict, image: bytes) -> bool:
        success = True
        try:
            socket.send_multipart([json.dumps(payload).encode("utf-8"), image], flags=zmq.NOBLOCK)
            self._retries = 0
        except zmq.ZMQError as e:
            logger.warning("Send failed: outbound queue full or no connection")
            self._retries += 1
            success = False
            # If we hit max retries, check if server restarted (inode changed)
            if self._retries >= self.max_retries:
                if self._check_socket_inodes_changed():
                    self.reconnect_queues()
        except Exception as e:
            log_exception(logger, "Unexpected error during send on zmq", e)
            success = False
        return success

    def send_clip_request(
        self, crop: np.ndarray, id_base: int, id_index: int, immediate: bool = False, extra: Optional[dict] = None
    ) -> bool:
        payload = {
            "camera_id": self.camera_id,
            "id_base": id_base,
            "id_index": id_index,
            "immediate": immediate,
        }
        if extra is not None:
            payload["extra"] = extra
        return self._internal_send_socket(self.clip_socket, payload, crop.tobytes())

    def get_clip_encodings(self) -> Optional[List]:
        # Take all messages from the queue and return them
        results = list(self.clip_results)
        self.clip_results.clear()
        return results

    def check_server_health(self, verbose: bool = False) -> bool:
        _is_healthy = False
        try:
            with self.session.get(self.server_address + "/health", timeout=self.default_timeout) as response:
                response.raise_for_status()
                _is_healthy = response.json().get("status") == "ok"
                if verbose:
                    logger.info(response.json())
                self._retries = 0
        except requests.exceptions.RequestException as e:
            log_exception(logger, "An error occurred trying to check server health:", e)
            self._retries += 1
        return _is_healthy

    def send_expert_request(
        self,
        image: AnalyticImage,
        detections: np.ndarray = None,
        flexibility: int = MAX_FLEXIBILITY,
        immediate: bool = False,
    ) -> bool:
        success = True
        found, _ = self.expert_cache.query(image.timestamp, flexibility)
        if not found:
            success = self._internal_expert_request(image, detections, immediate, False)
            if success:
                self.expert_cache.insert(image.timestamp, None)
        return success

    def send_weapon_request(
        self,
        image: AnalyticImage,
        detections: np.ndarray = None,
        flexibility: int = MAX_FLEXIBILITY,
        immediate: bool = False,
        confidence: Optional[int] = None,
    ) -> bool:
        success = True
        found, _ = self.weapon_cache.query(image.timestamp, flexibility)
        if not found:
            extra = None if confidence is None else {"confidence": confidence}
            success = self._internal_expert_request(image, detections, immediate, True, extra)
            if success:
                self.weapon_cache.insert(image.timestamp, None)
        return success

    def _internal_expert_request(
        self,
        image: AnalyticImage,
        detections: np.ndarray = None,
        immediate: bool = False,
        is_weapon: bool = False,
        extra: dict = None,
    ) -> bool:
        payload = {
            "camera_id": self.camera_id,
            "timestamp": image.timestamp,
            "batch_data": detections.tolist() if detections is not None else [],
            "immediate": immediate,
        }
        if extra is not None:
            payload.update(extra)
        # resize the image to detector size
        socket_ = self.expert_socket
        if is_weapon:
            socket_ = self.weapon_socket
            try:
                fname = f"expert_{self.camera_id}_{image.timestamp}.png"
                cv2.imwrite(os.path.join(self.storage_path, fname), image.frame)
                payload["image_path"] = fname
            except Exception as e:
                log_exception(logger, "Error preparing weapon image for expert request:", e)
        compressed = compress_image(self.resizer.resize(image, is_bgr=False))
        success = self._internal_send_socket(socket_, payload, compressed)
        return success

    def get_expert_detections(
        self, timestamp: int, flexibility: int = MAX_FLEXIBILITY, is_weapon: bool = False
    ) -> Optional[List]:
        cache = self.weapon_cache if is_weapon else self.expert_cache
        found, results = cache.query(timestamp, flexibility)
        if found:
            return [results] if results is not None else []
        else:
            return None

    def try_heal(self):
        """Recreate the session and adapter to recover from connection issues"""
        try:
            # Close existing session
            self.session.close()

            # reinitialize session
            self._init_session()

            # Verify server is reachable
            if self.check_server_health():
                logger.info(f"Session healed successfully for camera {self.camera_id}")
                # Also reconnect ZMQ sockets in case the server restarted
                self.reconnect_queues()
                return True
            else:
                logger.warning(f"Heal attempt failed - server still unreachable for camera {self.camera_id}")
                return False

        except Exception as e:
            log_exception(logger, f"Error during session healing for camera {self.camera_id}:", e)
            return False

    @property
    def is_healthy(self) -> bool:
        return self._retries < self.max_retries


class OfflineAnalyticsClientFactory:
    """
    Factory for OfflineAnalyticsClient that ensures a singleton per (server_address, camera_id) pair.
    """

    _clients = {}

    @classmethod
    def get_client(cls, server_address: str, camera_id: str) -> OfflineAnalyticsClient:
        key = (server_address, camera_id)
        if key not in cls._clients:
            cls._clients[key] = OfflineAnalyticsClient(server_address, camera_id)
        return cls._clients[key]

    @classmethod
    def close_all(cls):
        for client in cls._clients.values():
            client.close()
        cls._clients.clear()
