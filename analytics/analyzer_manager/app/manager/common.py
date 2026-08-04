import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict

import pika


# should reflect enum AnalyticMsgAction  in edge: libs/src/analytics_msg_interface.ts
class ManagementMessage(Enum):
    START = 0
    STOP = 1
    UPDATE_ALERT = 2
    UPDATE_SEARCH = 3
    UPDATE_VARIABLE = 4
    RESTART_ANALYTICS = 5
    UPDATE_CAMERA = 6
    DELETE_CAMERA = 7
    CHECK_ANALYTIC_STATUS = 8
    BIOMETRIC_UPDATE = 9
    EXIT = 10
    EDGE_STATUS = 11
    GET_MODELS_VERSION = 12
    UPDATE_CUSTOM_OBJECT = 13
    ADD_MODEL = 14
    ARM_POLICY = 15
    RELOAD_CONFIG = 16
    RESET_PROPER_FITTING = 17
    UPDATE_ANALYTIC_ASSET = 18
    EXTERNAL_EVENT_VALIDATION = 19
    UPDATE_CONTROL_ZONE = 20
    UPDATE_ZONE_REGIONS = 21

class AnalyticTrainingMode(Enum):
    CUSTOMIZED = 0
    STATIC = 1
    LOCKED = 2

class ConsumerState(Enum):
    INIT = 0
    STOPPED = 1
    WORKING = 2
    FAULT = 3
    INIT_FAULT = 4
    UPDATE_OK = 5
    UPDATE_FAILED = 6


class DataType(Enum):
    Image = 0
    Audio = 1

class ErrorState(Enum):
    CONNECTION_ERROR = "ConnectionError"


class ResponseAction(str, Enum):
    PROGRESS = "Progress"
    COMPONENT_STATUS = "ComponentStatus"
    ERROR = "Error"
    ANALYTIC_DOCKER_FAILURE = "AnalyticDockerFailure"
    NO_THUMBNAIL = "NoThumbnail"
    MODEL_VERSIONS = "ModelVersions"
    ANALYTIC_PROGRESS = "AnalyticProgress"
    UPDATE_STATUS = "UpdateStatus"
    ANALYZER_READY = "AnalyzerReady"


class CameraManagementType(str, Enum):
    UNKNOWN = "Unknown"
    UPDATE = "Update"
    RESOLUTION = "Resolution"
    ADD_MODEL = "AddModel"
    RESET_PROPER_FITTING = "ResetPf"



class ComponentStatus(Enum):
    AVAILABLE = "Available"
    UNAVAILABLE = "Unavailable"


@dataclass
class ManagementMessageData:
    action: ManagementMessage
    camera_id: str
    edge_id: str
    message_body: Dict


@dataclass
class FrameData:
    timestamp: int
    frame_number: int
    message_type: DataType = DataType.Image  # default

@dataclass
class QueueParams:
    name: str
    ttl: int


def gen_connection_channel():
    parameters = pika.ConnectionParameters(host=os.environ.get("RABBIT_HOST", "rabbitMQ"))
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    return connection, channel


connection_exceptions = (
    pika.exceptions.ConnectionClosed,
    pika.exceptions.ChannelClosed,
    pika.exceptions.StreamLostError,
    pika.exceptions.ChannelWrongStateError,
)

ALL_CAMERAS_ID: str = "all_cameras"