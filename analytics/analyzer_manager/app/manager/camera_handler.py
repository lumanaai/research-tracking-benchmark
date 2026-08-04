import dataclasses
import json
import os
import threading
import time
from copy import copy
from queue import Empty, Queue
from typing import List, Optional, Dict, Union

import pika

from analyzer_class import Analyzer
from general.analyzer_general import log_exception, logger
from general.core import AnalyticImage, ThumbnailInfo, np_encoder
from manager.common import (
    ConsumerState,
    ErrorState,
    gen_connection_channel,
    FrameData,
    DataType,
    CameraManagementType,
    ComponentStatus,
    connection_exceptions,
)


class CameraHandler(threading.Thread):
    connection: pika.BlockingConnection
    channel: pika.adapters.blocking_connection.BlockingChannel

    def __init__(self, camera_id: str, init_data: dict, context, updates: List[dict] = None, queue: Queue = None):

        super().__init__(name=f"{camera_id}_handler")
        self.race_counter = 0
        self.dropped_counter = 0
        self.camera_id = camera_id
        self.init_data = init_data
        self.config = context.config
        self.context = context
        self.updates = updates or []

        # queues
        self.analytics_q = init_data["analyticQueue"]
        self.thumb_q = init_data["thumbQueue"]
        self.motion_q = init_data["motionVectorsQueue"]
        self.train_q = init_data["trainingQueue"]

        # initializing state
        self.state = ConsumerState.INIT
        self.file_processing = init_data["footageAnalysis"] if "footageAnalysis" in init_data else False

        max_queue_size = self.config["analytics"].get("maxInternalQueueSize", 100)
        if self.file_processing:
            max_queue_size = 1000
        self.queue = Queue(maxsize=max_queue_size) if queue is None else queue

        self.management_work = []

        self.edge_id = init_data["edgeId"]
        self.images_path = os.path.join(self.config["locations"]["analyticInImages"], self.edge_id, self.camera_id)
        self.audio_path = os.path.join(self.config["locations"]["analyticInAudio"], self.edge_id, self.camera_id)

        self.next_update = 0
        self.last_frame_number = 0
        self.last_audio_file = 0
        self.analyzer = None

        # self-healing
        self.exception_count = 0
        self.max_exceptions = self.config["analytics"].get("consecutiveExceptionsAllowed", 3)
        self.error: Optional[ErrorState] = None
        self.target_fps = 5
        self.over_flow_upper_limit = self.config["analytics"].get("queueOverflowUpperLimit", self.target_fps * 3)
        self.over_flow_lower_limit = self.config["analytics"].get("queueOverflowLowerLimit", self.target_fps * 2)

        self.file_processing = init_data["footageAnalysis"] if "footageAnalysis" in init_data else False

    def _open_output_connection(self):
        # declaring all queues
        self.channel.queue_declare(
            self.analytics_q, arguments={"x-message-ttl": self.config["msgQueue"]["analyticsResultsQueueTTL"]}
        )
        self.channel.queue_declare(
            self.thumb_q, arguments={"x-message-ttl": self.config["msgQueue"]["analyticsResultsQueueTTL"]}
        )
        self.channel.queue_declare(
            self.motion_q, arguments={"x-message-ttl": self.config["msgQueue"]["motionVectorsQueueTTL"]}
        )
        self.channel.queue_declare(
            self.train_q, arguments={"x-message-ttl": self.config["msgQueue"]["trainingThumbanilQueueTTL"]}
        )
        self.channel.basic_qos(prefetch_count=1)

    def run(self):
        # initialize the analyzer in the working thread
        try:
            if self.context.analyzers.get(self.camera_id, None) is None:
                self.analyzer = Analyzer(self.init_data, self.config)

                for update in self.updates:
                    try:
                        self.analyzer.update(update)
                    except Exception as e:
                        log_exception(logger, f"Exception caught while applying updates for camera {self.camera_id}", e)
                        logger.error(f"Ignoring the update: {update}")
                self.context.analyzers[self.camera_id] = self.analyzer
            else:
                self.analyzer = self.context.analyzers[self.camera_id]

            # declaring all queues in the current thread
            self.connection, self.channel = gen_connection_channel()
            self._open_output_connection()

            self._change_state(ConsumerState.WORKING)
            last_processed_ts = 0
        except Exception as e:
            self._change_state(ConsumerState.INIT_FAULT)
            log_exception(logger, f"Exception caught initializing the analyzer for camera {self.camera_id}", e)
            return
        idle_counter = 0
        timeout = 1
        empty_queue(self.queue)
        interleaved = 0
        while True:
            if self.state == ConsumerState.STOPPED:
                logger.info(f"Receive stopped state from manager, breaking from main loop")
                try:
                    self.analyzer.clear()
                except Exception as e:
                    self.context.analyzers.pop(self.camera_id)
                    log_exception(logger, f"Exception caught while clearing the analyzer", e)
                break  # stop the thread
            try:
                data: FrameData = self.queue.get(timeout=timeout)
                if data.message_type == DataType.Audio.value:
                    # place holder for now we delete audio files
                    local_path = os.path.join(self.audio_path, f"snd-{data.timestamp}.pcm")
                    self.last_audio_file = data.frame_number
                    self.remove_file_safely(local_path)
                    continue

                if interleaved > 0:
                    interleaved -= 1
                    if interleaved % 2 == 0:
                        self.dropped_counter += 1
                        local_path = os.path.join(self.images_path, f"img-{data.timestamp}.yuv")
                        self.remove_file_safely(local_path)
                        continue

                self.last_frame_number = data.frame_number
                qsize = self.queue.qsize()
                if qsize > self.over_flow_lower_limit and not self.file_processing:
                    if qsize > self.over_flow_upper_limit:
                        self.handle_burst()
                        interleaved = 0  # revoke interleaved state if we had a full burst
                    else:
                        interleaved = qsize  # enter interleaved state to drop half of the frames

                is_processed_batch = self.on_image_received(data)
                if is_processed_batch:
                    idle_counter = 0  # reset the idle counter
                else:
                    continue  # avoid checking for maintenance every iteration

            except Empty:
                # print(f"empty counter: {idle_counter}")
                idle_counter += 1
                if idle_counter == (10 // timeout):  # notify just one time
                    logger.warning("No images to consume after waiting for 10 seconds")
            except EnvironmentError:
                if self.state == ConsumerState.FAULT:
                    # attempt to recover
                    is_healed = self._health_check()
                    if not is_healed:
                        logger.error(f"Camera {self.camera_id} is not recoverable")
                        self.context.analyzers.pop(self.camera_id)
                        break
                    else:
                        self._change_state(ConsumerState.WORKING)
            except Exception as e:
                log_exception(logger, "Unexpected exception caught in main loop", e)
                self._change_state(ConsumerState.FAULT)
                self.context.analyzers.pop(self.camera_id)
                break  # wait for a start message from manager to recover

            if self.management_work:
                self._handle_management_work()
        logger.info(f"Analytic thread for camera {self.camera_id} Finished")

    def set_resolution(self, requested_resolution):
        self.management_work.append((CameraManagementType.RESOLUTION, requested_resolution))

    def stop(self):
        self._change_state(ConsumerState.STOPPED)

        logger.info(f"clearing internal queue for camera {self.camera_id}")
        empty_queue(self.queue)

    def update(self, update_msg):
        self.management_work.append((CameraManagementType.UPDATE, update_msg))

    def add_model(self, update_msg):
        self.management_work.append((CameraManagementType.ADD_MODEL, update_msg))

    def reset_proper_fitting(self, update_msg):
        self.management_work.append((CameraManagementType.RESET_PROPER_FITTING, update_msg))

    def get_versions(self) -> Dict:
        registered_versions = {}
        if self.analyzer is None:
            logger.error(f"attempt to retrieve model versions for camera {self.camera_id} before initialization")
            return registered_versions
        try:
            registered_versions = copy(self.analyzer.registered_models)
        except Exception as ex:
            logger.error(f"Exception caught trying to retrieve model versions: {str(ex)}")

        return registered_versions

    def on_image_received(self, frame_data: FrameData) -> bool:
        is_processed = False
        try:
            local_path = os.path.join(self.images_path, f"img-{frame_data.timestamp}.yuv")
            image = self.read_and_remove(local_path)
            if image is None:
                logger.warning(f"Overflow: image was deleted by uploader before read by analyzer")
                return False
            analytic_image = AnalyticImage(image, frame_data.timestamp, frame_data.frame_number)
            result = self.analyzer.process_image(self.queue.qsize(), analytic_image)
            # print(f"timestamp: {frame_data.timestamp}, index {frame_data.frame_number}, results: {result.metadata}")
            if result.metadata is not None:
                is_processed = True
                success = self._publish_result(result)
                if not success:
                    logger.warning("got failure attempting to send results, trying to recover")
                    # attempt to recover the connection and send again
                    self.connection, self.channel = gen_connection_channel()
                    self._open_output_connection()
                    success = self._publish_result(result)
                    self.error = ErrorState.CONNECTION_ERROR
                    if not success:
                        raise ConnectionError("Failed to recover the connection")

            self.exception_count = 0  # reset the exception count
        except Exception as e:
            log_exception(logger, "Exception caught in command queue listener", e)
            self.exception_count += 1
            if self.exception_count > self.max_exceptions:
                self.state = ConsumerState.FAULT
                raise EnvironmentError("Too many exceptions")
        return is_processed

    def _change_state(self, new_state: ConsumerState):
        if self.state != new_state:
            self.state = new_state

            # notify new state
            self.context.camera_status_update.append((self.camera_id, new_state, {}))

    def _send_images(self, queue_name, images: List[Union[ThumbnailInfo, dict]]):
        for thum_info in images:
            thum_dict = thum_info
            if isinstance(thum_info, ThumbnailInfo):
                thum_dict = dataclasses.asdict(thum_info)
            thum_body = json.dumps(thum_dict, default=np_encoder)
            self.channel.basic_publish(exchange="", routing_key=queue_name, body=thum_body)

    def _publish_result(self, result) -> bool:
        should_hold = False

        # attempt to send analytics
        try:
            if result.metadata.info:
                info = result.metadata.info
                msg = {"type": "analytics", "data": info}
                json_msg = json.dumps(msg, default=np_encoder)  # , cls=NumpyEncoder)
                logger.debug(f"Sent analytic results for cameraId: {self.camera_id}")

                self.channel.basic_publish(exchange="", routing_key=self.analytics_q, body=json_msg)
                result.metadata.info = None
        except connection_exceptions as e:
            log_exception(logger, "Exception caught while publishing analytic message", e)
            should_hold = True
        except Exception as e:
            log_exception(logger, "General exception caught", e)
            logger.error("due to exception analytic message could not be sent")

        try:
            if result.counting_info:
                keys = list(result.counting_info.keys())
                for k in keys:
                    v = result.counting_info[k]
                    msg = {"type": k, "data": v}
                    json_msg = json.dumps(msg, default=np_encoder)  # , cls=NumpyEncoder)
                    logger.info(f"Sent {k} results for cameraId: {self.camera_id}")
                    self.channel.basic_publish(exchange="", routing_key=self.analytics_q, body=json_msg)
                    result.counting_info.pop(k)
                result.counting_info = None
        except connection_exceptions as e:
            log_exception(logger, "Exception caught while publishing analytic message", e)
            should_hold = True
        except Exception as e:
            log_exception(logger, "General exception caught", e)
            logger.error("due to exception analytic message could not be sent")

        try:
            if result.encodings:
                msg = {"type": "object_encodings", "data": result.encodings}
                json_msg = json.dumps(msg, default=np_encoder)  # , cls=NumpyEncoder)
                logger.debug(f"Sent object encodings results for cameraId: {self.camera_id}")

                self.channel.basic_publish(exchange="", routing_key=self.analytics_q, body=json_msg)
                result.encodings = None
        except connection_exceptions as e:
            log_exception(logger, "Exception caught while publishing analytic message", e)
            should_hold = True
        except Exception as e:
            log_exception(logger, "General exception caught", e)
            logger.error("due to exception analytic message could not be sent")

        # attempt to send motion vectors
        try:
            if result.motion_info:
                info = result.motion_info
                json_msg = json.dumps(info, default=np_encoder)  # , cls=NumpyEncoder)
                logger.debug(f"Sent motion results for cameraId: {self.camera_id}")

                self.channel.basic_publish(exchange="", routing_key=self.motion_q, body=json_msg)
                result.motion_info = None
        except connection_exceptions as e:
            log_exception(logger, "Exception caught while publishing motion vectors", e)
            should_hold = True
        except Exception as e:
            log_exception(logger, "General exception caught", e)
            logger.error("due to exception motion vectors could not be sent")

        # attempt to send thumbnails
        try:
            if result.metadata.thumbnails:
                self._send_images(self.thumb_q, result.metadata.thumbnails)
                result.metadata.thumbnails = []
            if result.metadata.training:
                logger.info(f"sent training images: {result.metadata.training}")
                self._send_images(self.train_q, result.metadata.training)
                result.metadata.training = []
            if result.metadata.snapshots:
                self._send_images(self.train_q, result.metadata.snapshots)
                result.metadata.snapshots = []

        except connection_exceptions as e:
            log_exception(logger, "Exception caught while publishing analytic message", e)
            should_hold = True
        except Exception as e:
            log_exception(logger, "General exception caught", e)
            logger.error("due to exception analytic message thumbnails could not be sent")
            should_hold = True
        return not should_hold

    def _health_check(self) -> bool:
        return False

    def _handle_management_work(self):
        while self.management_work:
            work = self.management_work.pop(0)
            update_status = ConsumerState.UPDATE_OK
            res = "ok"
            if work[0] == CameraManagementType.RESOLUTION:
                try:
                    self.init_data["analyticAppParameters"]["inputStream"] = work[1]
                    self.analyzer.update_resolution(work[1])
                    self.analyzer.image_batch.clear()  # clear the current batch
                    empty_queue(self.queue)  # empty the queue so no work will be done in mixed resolution
                    res = {"resolution": work[1]}
                except Exception as e:
                    update_status = ConsumerState.UPDATE_FAILED
                    res = str(e)
            elif work[0] == CameraManagementType.UPDATE:
                try:
                    self.analyzer.update(work[1])
                except Exception as e:
                    update_status = ConsumerState.UPDATE_FAILED
                    res = str(e)
            elif work[0] == CameraManagementType.ADD_MODEL:
                try:
                    logger.info(f"Adding model: {work[1]}")
                    self.analyzer.add_model()
                except Exception as e:
                    update_status = ConsumerState.UPDATE_FAILED
                    res = str(e)
            elif work[0] == CameraManagementType.RESET_PROPER_FITTING:
                try:
                    logger.info(f"reset PF with flags: {work[1]}")
                    self.analyzer.reset_pf(flags=work[1])
                except Exception as e:
                    update_status = ConsumerState.UPDATE_FAILED
                    res = str(e)
            message = {"type": work[0].value, "message": work[1], "result": res}
            if update_status == ConsumerState.UPDATE_OK:
                logger.info(f"Camera {self.camera_id} management work completed: {message}")
            else:
                logger.error(f"Camera {self.camera_id} management work failed: {message}")
            self.context.camera_status_update.append((self.camera_id, update_status, message))

    def remove_file_safely(self, local_path: str):
        # Step 2: Delete the file
        try:
            os.remove(local_path)
        except OSError:
            self.race_counter += 1

    def read_and_remove(self, local_path: str) -> Optional[bytes]:
        # Step 1: Read the file
        try:
            with open(local_path, "rb") as file:
                content = file.read()
        except FileNotFoundError:
            self.race_counter += 1
            self.dropped_counter += 1
            # logger.warning(f"Race: Didnt managed to read a file {local_path}")
            return None

        # Step 2: Delete the file
        self.remove_file_safely(local_path)
        return content

    def get_period_race_files(self) -> int:
        race_files, self.race_counter = self.race_counter, 0
        if race_files:
            logger.warning(f"Encountered {race_files} occasions of race condition reading files in the last period")
        return race_files

    def get_period_dropped_files(self) -> int:
        n_files, self.dropped_counter = self.dropped_counter, 0
        if n_files:
            logger.warning(f"had to drop {n_files} files due to buffer overflow in the last period")
        return n_files

    def get_health_state(self) -> int:
        if self.analyzer is None:
            return -1
        # Convert flags to int with bit per flag
        flags = [
            self.analyzer.alert_status,
            self.analyzer.image_quality,
            self.analyzer.synced_status,
            self.analyzer.offline_status,
        ]
        if all(flags):
            return 0

        health_state = 0
        for i, flag in enumerate(flags):
            if not flag:
                health_state |= 1 << i
        return health_state

    @property
    def camera_resolution(self):
        return self.init_data["analyticAppParameters"]["inputStream"]

    @property
    def alert_status(self) -> bool:
        if self.analyzer is None:
            return False
        return self.analyzer.alert_status

    @property
    def synced_status(self) -> bool:
        if self.analyzer is None:
            return False
        return self.analyzer.synced_status

    @property
    def image_quality(self) -> bool:
        if self.analyzer is None:
            return False
        return self.analyzer.image_quality

    @property
    def offline_status(self) -> bool:
        if self.analyzer is None:
            return False
        return self.analyzer.offline_status

    @property
    def total_processed(self) -> int:
        if self.analyzer is None:
            return -1
        return self.analyzer.total_processed

    @property
    def bad_alerts(self) -> List[Dict[str, str]]:
        if self.analyzer is None:
            return []
        return self.analyzer.bad_alerts

    def handle_burst(self):
        qsize = self.queue.qsize()
        _, rem = divmod(qsize, self.target_fps)
        to_drop = qsize - (rem or self.target_fps)
        i = 0
        while i < to_drop:
            try:
                data: FrameData = self.queue.get_nowait()
                local_path = os.path.join(self.images_path, f"img-{data.timestamp}.yuv")
                self.remove_file_safely(local_path)
                i += 1
            except Empty:
                break
        self.dropped_counter += to_drop


def empty_queue(queue: Queue):
    while True:
        try:
            queue.get_nowait()
        except Empty:
            break
    pass


def next_round_period(period: int = 60):
    return int(time.time() / period) * period + period


def consumer_status_to_component(status: ConsumerState) -> ComponentStatus:
    if status == ConsumerState.WORKING:
        return ComponentStatus.AVAILABLE
    return ComponentStatus.UNAVAILABLE
