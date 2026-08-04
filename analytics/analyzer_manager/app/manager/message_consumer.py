import json
import threading
import time
from queue import Queue, Full, Empty

import pika

from general.analyzer_general import logger, log_exception
from manager.common import (
    FrameData,
    QueueParams,
    ConsumerState,
    gen_connection_channel,
    ManagementMessageData,
    connection_exceptions,
    ManagementMessage,
    ALL_CAMERAS_ID,
    DataType,
)


def image_message_handling(message_data: dict):
    frame_data = FrameData(
        message_data["timestamp"],
        message_data["frameNumber"],
        DataType(message_data.get("messageType", DataType.Image.value)),
    )
    return frame_data


def default_message_handling(message_data: dict):
    return message_data


class ExternalMessageConsumer(threading.Thread):
    connection: pika.BlockingConnection
    channel: pika.adapters.blocking_connection.BlockingChannel
    consumer_tag: str

    def __init__(self, internal_queue: Queue, external_queue: QueueParams, msg_function: callable = None):

        super().__init__(name=f"{external_queue.name}_consumer")
        self.internal_queue = internal_queue
        self.external_queue_params = external_queue
        self.state = ConsumerState.INIT
        self.last_message_index = 0
        self.last_message_timestamp = 0

        if msg_function is None:
            msg_function = default_message_handling
        self.external_message_handling = msg_function
        self.msg_queue = self.external_queue_params.name
        if isinstance(internal_queue, Queue):
            self.drop_outs = int(internal_queue.maxsize * 0.7)
        else:
            self.drop_outs = 1
        self.last_full_message_ts = 0
        self.total_last_dropped = 0

    def _msg_callback(self, channel, method, properties, body):
        if self.state == ConsumerState.STOPPED:
            raise KeyboardInterrupt
        channel.basic_ack(delivery_tag=method.delivery_tag)
        body_data = json.loads(body)
        obj = self.external_message_handling(body_data)
        try:
            self.internal_queue.put(obj, block=False)
        except Full:
            drp = 0
            try:
                for drp in range(self.drop_outs):
                    self.internal_queue.get_nowait()
            except Empty:
                pass  # general protection against races and bad config
            self.total_last_dropped += drp
            if self.last_full_message_ts + 60 < time.time():
                self.last_full_message_ts = time.time()
                logger.warning(f"Queue {self.msg_queue} is full, dropped {self.total_last_dropped} in the last minute")
                self.total_last_dropped = 0

        self.post_message_handling(obj)

    def run(self):
        self.connection, self.channel = gen_connection_channel()
        self.channel.queue_declare(self.msg_queue, arguments={"x-message-ttl": self.external_queue_params.ttl})
        self.consumer_tag = self.channel.basic_consume(self.msg_queue, self._msg_callback)

        self.state = ConsumerState.WORKING
        logger.info(f"start consuming from {self.msg_queue}")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.state = ConsumerState.STOPPED
            logger.info(f"Stopped consuming from {self.msg_queue}")
        except Exception as e:
            if self.state == ConsumerState.WORKING:
                log_exception(logger, "Exception caught in command queue listener", e)
                self.state = ConsumerState.FAULT
        logger.info(f"Thread consuming from {self.msg_queue} Finished")

    def stop(self):
        self.state = ConsumerState.STOPPED

    def post_message_handling(self, obj):
        self.last_message_index += 1
        self.last_message_timestamp = int(time.time() * 1000)

    def force_shutdown(self):
        logger.info(f"Force shutdown of consumer for queue {self.msg_queue}")

        def internal_func():
            self.state = ConsumerState.STOPPED
            #   close the channel
            try:
                self.channel.stop_consuming(self.consumer_tag)
                logger.info("consuming stopped successfully")
            except connection_exceptions:
                logger.info("consuming stopped from expected exception")
            except Exception as e:
                logger.warning(f"exception caught trying to close channel: {str(e)}")

        # run in separate thread to avoid deadlock
        t = threading.Thread(target=internal_func)
        t.start()


class ImagesExternalMessageConsumer(ExternalMessageConsumer):
    def post_message_handling(self, obj):
        obj: FrameData
        self.last_message_index = obj.frame_number
        self.last_message_timestamp = obj.timestamp


def management_message_handling(message_data: dict):
    action = ManagementMessage(message_data["msgAction"])
    camera_id = message_data.get("cameraId", ALL_CAMERAS_ID)
    edge_id = message_data["edgeId"]
    obj = ManagementMessageData(action, camera_id, edge_id, message_data)
    logger.info(f"Received action {action} for {camera_id}")
    return obj
