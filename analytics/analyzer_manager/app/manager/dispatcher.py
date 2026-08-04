import gc
import json
import os
import time
from queue import SimpleQueue, Empty, Queue
from typing import Dict, Optional

from analyzer_class import Analyzer
from general.analyzer_general import logger, log_exception
from general.proj import load_config
from manager.camera_handler import CameraHandler, next_round_period, consumer_status_to_component
from manager.common import (
    ConsumerState,
    ResponseAction,
    ManagementMessageData,
    QueueParams,
    gen_connection_channel,
    connection_exceptions,
    ManagementMessage,
    ALL_CAMERAS_ID,
)
from manager.message_consumer import (
    image_message_handling,
    ExternalMessageConsumer,
    ImagesExternalMessageConsumer,
    management_message_handling,
)


class AnalyticsDispatcher:
    management_consumer: ExternalMessageConsumer

    def __init__(self):
        self.instance_id = os.environ.get("INSTANCE_ID", "0")
        self.config = load_config()
        self.camera_handlers: Dict[str, CameraHandler] = {}  # camera_id -> CameraHandler
        self.camera_ext_consumers: Dict[str, ExternalMessageConsumer] = {}  # camera_id -> CameraHandler
        self.camera_queues: Dict[str, Queue] = {}  # camera_id -> CameraHandler
        self.analyzers: Dict[str, Analyzer] = {}
        self.connection, self.channel = gen_connection_channel()
        self.state = ConsumerState.INIT

        # response queue
        self.response_queue = self.config["msgQueue"]["analyticsResponseQueue"]
        ttl = self.config["msgQueue"]["analyticsResponseQueueTTL"]
        self.channel.queue_declare(self.response_queue, arguments={"x-message-ttl": ttl})

        # management queue
        self.management_queue = SimpleQueue()
        self.camera_status_update = []
        self.progress_update_interval = int(self.config["analytics"].get("progressUpdateInterval", 60000) / 1000)  # sec
        self.healthcheck_interval = int(self.config["analytics"].get("healthCheckInterval", 60000) / 1000)  # sec
        self._create_management_consumer()
        self.edge_id = ""
        if "app" in self.config and "edgeId" in self.config["app"]:
            self.edge_id = self.config["app"]["edgeId"]

        # health check
        self.purge_enabled = not self.config["analytics"].get("disableConsumerPurge", False)
        self.purge_period = int(self.config["analytics"].get("consumerPurgeInterval", 60000) / 1000)  # sec
        self.inactive_cameras: Dict[str, int] = {}

        self.management_error_counter = 0
        self.healthcheck_error_counter = 0
        self.max_allowed_errors = 3

    def run(self):
        self.management_consumer.start()
        self.handle_management_queue()

    def _create_management_consumer(self):
        queue_name = self.config["msgQueue"]["analyticsCmdQueue"] + "_" + self.instance_id
        ttl = self.config["msgQueue"]["analyticsCmdQueueTTL"]
        ext_params = QueueParams(queue_name, ttl)
        self.management_consumer = ExternalMessageConsumer(
            self.management_queue, ext_params, management_message_handling
        )

    def handle_management_queue(self):
        timeout = 2  # seconds
        self.state = ConsumerState.WORKING
        self.send_analyzer_ready()
        self.send_progress_update()

        next_progress_update = next_round_period(self.progress_update_interval)
        next_healthcheck_time = next_round_period(self.healthcheck_interval)
        while self.state != ConsumerState.STOPPED:
            data = None
            try:
                # Try to get an item from the queue, waiting up to "timeout" seconds
                data = self.management_queue.get(timeout=timeout)
                self.handle_management_message(data)
            except Empty:
                pass  # no need to do anything, just continue the loop
            except Exception as e:
                log_exception(logger, f"Error executing the following command: {data}", e)
            # handle any pending status updates
            try:
                while len(self.camera_status_update) > 0:
                    camera_id, status, message = self.camera_status_update.pop(0)
                    if status in [ConsumerState.UPDATE_OK, ConsumerState.UPDATE_FAILED]:
                        # self._publish_response(self.edge_id, camera_id, ResponseAction.UPDATE_STATUS, message)
                        pass  # for now ignore the update status
                    else:
                        component_status = consumer_status_to_component(status)
                        self._publish_camera_status(self.edge_id, camera_id, component_status)
            except Exception as e:
                log_exception(logger, "Error sending camera status update", e)

            # send progress update if needed
            try:
                if time.time() > next_progress_update:
                    next_progress_update = next_round_period(self.progress_update_interval)
                    self.send_progress_update()
                    self.management_error_counter = 0  # reset the error counter

            except connection_exceptions as e:
                log_exception(logger, "Error sending progress update", e)
                logger.info("attempting to self heal the connection")
                self.connection, self.channel = gen_connection_channel()
                ttl = self.config["msgQueue"]["analyticsResponseQueueTTL"]
                self.channel.queue_declare(self.response_queue, arguments={"x-message-ttl": ttl})
                self.management_error_counter += 1

            except Exception as e:
                log_exception(logger, "Error sending progress update", e)

            # perform health check and cleanups
            try:
                if self.management_queue.qsize() == 0 and time.time() > next_healthcheck_time:
                    self.health_check()
                    next_healthcheck_time = next_round_period(self.healthcheck_interval)
                    self.healthcheck_error_counter = 0  # reset the error counter

            except Exception as e:
                log_exception(logger, "Error while preforming health check", e)
                self.healthcheck_error_counter += 1

            if self.management_error_counter > self.max_allowed_errors:
                logger.error("Too many connection errors, stopping the dispatcher")
                self.state = ConsumerState.STOPPED
            elif self.healthcheck_error_counter > self.max_allowed_errors:
                logger.error("Too many health check errors, stopping the dispatcher")
                self.state = ConsumerState.STOPPED

        # stop all consumers
        logger.info("Stopping all consumers and handlers")
        self.management_consumer.stop()
        for camera_id, consumer in self.camera_ext_consumers.items():
            consumer.stop()
        for camera_id, handler in self.camera_handlers.items():
            handler.stop()
        logger.info("Exiting Analytics Dispatcher")

    def handle_management_message(self, data: ManagementMessageData):
        body = data.message_body
        camera_id = data.camera_id

        if data.action == ManagementMessage.START:
            self._on_camera_start(camera_id, body)
        elif data.action == ManagementMessage.STOP:
            self._on_camera_stop(camera_id)
        elif data.action in [
            ManagementMessage.UPDATE_ALERT,
            ManagementMessage.UPDATE_SEARCH,
            ManagementMessage.UPDATE_VARIABLE,
            ManagementMessage.BIOMETRIC_UPDATE,
            ManagementMessage.ARM_POLICY,
            ManagementMessage.UPDATE_ANALYTIC_ASSET,
            ManagementMessage.EXTERNAL_EVENT_VALIDATION,
            ManagementMessage.UPDATE_CONTROL_ZONE,
        ]:
            self._on_camera_update(camera_id, body)
        elif data.action == ManagementMessage.GET_MODELS_VERSION:
            if camera_id in self.camera_handlers:
                logger.info(f"model versions requested for camera {camera_id}")
                versions_dict = self.camera_handlers[camera_id].get_versions()
                response = {"versions": versions_dict, "token": body["token"]}
                self._publish_response(data.edge_id, camera_id, ResponseAction.MODEL_VERSIONS, response)
            else:
                logger.error(f"model versions requested for unknown camera {camera_id}")
        elif data.action == ManagementMessage.ADD_MODEL:
            self._on_camera_add_model(camera_id, body)
        elif data.action == ManagementMessage.RESET_PROPER_FITTING:
            self._on_camera_reset_proper_fitting(camera_id, body)
        elif data.action == ManagementMessage.CHECK_ANALYTIC_STATUS:
            logger.warning(f"got check analytic status message for camera {camera_id}: data: {data.message_body}")
        else:
            logger.error(f"Unknown action {data.action} for camera {camera_id}")

    def _on_camera_start(self, camera_id: str, init_data: dict):
        self.inactive_cameras.pop(camera_id, None)  # in case it was inactive
        if camera_id in self.camera_handlers:
            if self.camera_handlers[camera_id].state in [ConsumerState.WORKING]:
                requested_resolution = init_data["analyticAppParameters"]["inputStream"]
                current_resolution = self.camera_handlers[camera_id].camera_resolution
                if current_resolution == requested_resolution:
                    logger.warning(f"Camera {camera_id} already started, Ignoring message")
                else:
                    self.camera_handlers[camera_id].set_resolution(requested_resolution)
                return
            elif self.camera_handlers[camera_id].state in [ConsumerState.INIT]:
                logger.warning(f"Camera {camera_id} is initializing, Ignoring message")
                return
            else:  # fault state
                logger.warning(f"Camera {camera_id} is in fault state, attempting to restart")
                try:
                    self.camera_ext_consumers[camera_id].stop()
                except Exception as e:
                    logger.error(f"Error stopping external consumer for camera {camera_id}: {e}")

        self.camera_handlers[camera_id] = CameraHandler(camera_id, init_data, self)
        self.camera_queues[camera_id] = self.camera_handlers[camera_id].queue
        self._activate_camera_consumer(camera_id, init_data)
        self.camera_handlers[camera_id].start()
        self.edge_id = self.camera_handlers[camera_id].edge_id

    def _activate_camera_consumer(self, camera_id, init_data: dict):
        queue_name = init_data["queue"]
        ttl = self.config["msgQueue"]["bmpMessageTTL"]
        self.camera_ext_consumers[camera_id] = ImagesExternalMessageConsumer(
            self.camera_queues[camera_id], QueueParams(queue_name, ttl), image_message_handling
        )
        self.camera_ext_consumers[camera_id].start()

    def _on_camera_stop(self, camera_id):
        if camera_id in self.camera_handlers:
            logger.info(f"stopping camera consumer for camera {camera_id}")
            self.camera_ext_consumers[camera_id].stop()  # stop the camera consumer from consuming from rabbit

            logger.info(f"stopping camera analyzer for camera {camera_id}")
            self.camera_handlers[camera_id].stop()  # stop the camera handler from consuming new images
            self.inactive_cameras[camera_id] = int(time.time())

        else:
            logger.warning(f"Camera {camera_id} already stopped, Ignoring message")

    def _on_camera_update(self, camera_id, update_msg: dict):
        if camera_id == ALL_CAMERAS_ID:
            for cam_id in self.camera_handlers:
                self.camera_handlers[cam_id].update(update_msg)
            return
        if camera_id in self.camera_handlers:
            self.camera_handlers[camera_id].update(update_msg)
        else:
            logger.error(f"Got update message for cameraId: {camera_id} that isn't initialized")

    def _on_camera_add_model(self, camera_id, update_msg: dict):
        if camera_id in self.camera_handlers:
            self.camera_handlers[camera_id].add_model(update_msg)
        else:
            logger.error(f"Got add model message for cameraId: {camera_id} that isn't initialized")

    def _on_camera_reset_proper_fitting(self, camera_id: str, update_msg: dict):
        if camera_id in self.camera_handlers:
            self.camera_handlers[camera_id].add_model(update_msg)
        else:
            logger.error(f"Got reset PF message for cameraId: {camera_id} that isn't initialized")

    def _basic_publish_safe(self, body):
        try:
            self.channel.basic_publish(exchange="", routing_key=self.response_queue, body=body)
        except connection_exceptions:
            self.connection, self.channel = gen_connection_channel()
            self.channel.basic_publish(exchange="", routing_key=self.response_queue, body=body)
            logger.info("Connection re-established and message published")

    def _publish_response(self, edge_id: str, camera_id: Optional[str], action: ResponseAction, data: dict = None):

        msg = {
            "edgeId": edge_id,
            "action": action.value,
        }
        if camera_id:
            msg["cameraId"] = camera_id

        if data:
            msg.update(data)
        logger.info(f"Publish: {msg}")

        self._basic_publish_safe(json.dumps(msg))

    def _publish_camera_status(self, edge_id: str, camera_id: str, component_status):
        msg = {
            "edgeId": edge_id,
            "cameraId": camera_id,
            "action": ResponseAction.COMPONENT_STATUS.value,
            "component": "analytics",
            "status": component_status.value,
        }
        logger.info(f"Publish: {msg}")
        self._basic_publish_safe(json.dumps(msg))

    def send_analyzer_ready(self):
        msg = {
            "action": ResponseAction.ANALYZER_READY.value,
            "instanceId": int(self.instance_id),
        }
        logger.info(f"Sending AnalyzerReady notification", extra={"instanceId": self.instance_id})
        self._basic_publish_safe(json.dumps(msg))

    def send_progress_update(self):
        # don't send progress if uninitialized
        if not self.edge_id:
            return

        report = []
        for camera_id, handler in self.camera_handlers.items():
            last_received = self.camera_ext_consumers[camera_id].last_message_index
            report.append(
                {
                    "cameraId": camera_id,
                    "last_processed": handler.last_frame_number,
                    "last_received": last_received,
                    "alert_status": handler.alert_status,
                    "synced_status": handler.synced_status,
                    "health_status": handler.get_health_state(),
                    "offline_status": handler.offline_status,
                    "image_quality": handler.image_quality,
                    "total_processed": handler.total_processed,
                    "race_files": handler.get_period_race_files(),
                    "dropped_files": handler.get_period_dropped_files(),
                    "bad_alerts": handler.bad_alerts,
                }
            )
        data = {"timestamp": int(time.time() * 1000), "progress": report}
        self._publish_response(self.edge_id, None, ResponseAction.ANALYTIC_PROGRESS, data)

    def health_check(self):
        cleaned_cameras = set()
        # first purge inactive cameras
        for camera, ts in self.inactive_cameras.items():
            if ts + self.purge_period < time.time():
                logger.info(f"Purging inactive camera {camera}")

                # force external consumer to stop
                ext_consumer = self.camera_ext_consumers.pop(camera, None)
                if ext_consumer:
                    ext_consumer.force_shutdown()
                    # del ext_consumer # not sure if this is needed

                # remove the camera from the list
                handler = self.camera_handlers.pop(camera, None)
                if handler is not None:
                    handler.analyzer = None  # remove pointer to the analyzer
                    del handler

                analyzer = self.analyzers.pop(camera, None)
                if analyzer is not None:
                    del analyzer

                queue = self.camera_queues.pop(camera, None)
                if queue is not None:
                    del queue
                gc.collect()  # Force garbage collection
                cleaned_cameras.add(camera)

        # remove the cameras from the inactive list
        for camera in cleaned_cameras:
            self.inactive_cameras.pop(camera)

        # list of cameras that supposed to be OK:
        active_cameras = [cam for cam in self.camera_handlers if cam not in self.inactive_cameras]

        # check if all cameras are working
        for camera_id in active_cameras:
            if self.camera_handlers[camera_id].state in [ConsumerState.FAULT, ConsumerState.STOPPED]:
                logger.info(f"Camera {camera_id} is in fault state - attempting recovery")
                try:
                    init_data = self.camera_handlers[camera_id].init_data
                    updates = self.camera_handlers[camera_id].updates
                    self.camera_handlers[camera_id] = CameraHandler(
                        camera_id, init_data, self, updates, self.camera_queues[camera_id]
                    )
                    self.camera_handlers[camera_id].start()
                except Exception as e:
                    log_exception(logger, f"Exception caught while attempting to recover camera {camera_id}", e)
                    self.camera_handlers[camera_id].state = ConsumerState.INIT_FAULT

            if self.camera_ext_consumers[camera_id].state in [ConsumerState.FAULT, ConsumerState.STOPPED]:
                logger.info(f"External messages consumer of camera {camera_id} is in fault state, attempting recovery")
                try:
                    init_data = self.camera_handlers[camera_id].init_data
                    self.camera_ext_consumers[camera_id].force_shutdown()
                    self._activate_camera_consumer(camera_id, init_data)
                except Exception as e:
                    log_exception(
                        logger,
                        f"Exception caught while attempting to recover msg consumer, camera {camera_id}",
                        e,
                    )
                    self.camera_ext_consumers[camera_id].state = ConsumerState.FAULT

    def stop(self):
        self.state = ConsumerState.STOPPED
