import base64
import collections.abc
import enum
import math
import os
import re
import socket
import threading
import traceback
from argparse import Namespace
from typing import Any, Dict

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from requests.auth import HTTPDigestAuth

from general.perf_utils import time_sync
from . import proj
from .logger import get_logger

ANALYTIC_ALERT_TYPE = 0
MULTIPLE_ENUM = 5
ALPR_UNKNOWN = ["unknown"]
ALPR_CAR_TYPES = ["sedan", "suv", "van", "unknown"]
ALPR_TRUCK_TYPES = ["big truck", "pickup truck"]
CAR_TYPES = ["sedan", "suv", "van", "big truck", "pickup truck", "bus"]
GLOBAL_TYPE_KEY = "globalType"
GLOBAL_SUCCESS = "success"
GLOBAL_FAILURE = "failure"
GLOBAL_BLACKLIST = "blacklist"

ROI_SHAPE = (32, 32)
ROUTE_SHAPE = (32, 32)

POSITION_LIST = [0] * (ROI_SHAPE[0] * ROI_SHAPE[1])

VEHICLE_FILTERS = ["make", "model", "colors", "plate"]
PARTIAL_MATCH_FILTERS = ["plate"]
AGE_FILTER = "ageType"
GENDER_FILTER = "genderType"
PLATE_NATIONALITY = "licensePlateNationality"

PERSON_FILTERS = [
    "accessoryType",
    AGE_FILTER,
    "carryingType",
    "footwearColor",
    "footwearType",
    GENDER_FILTER,
    "hairColor",
    "hairType",
    "lowerbodyColor",
    "lowerbodyType",
    "upperbodyColor",
    "upperbodyType",
]
PPE_FILTERS = ["protectiveGearType"]

SUPPORTED_FILTERS = ["type"] + VEHICLE_FILTERS + PERSON_FILTERS + PPE_FILTERS
CUSTOM_OBJECT_FILTER = "custom_object"

CHILD_VALUE = "less_18"
MALE_VALUE = "male"
FEMALE_VALUE = "female"

HTTP_CONNECT_TIMEOUT = 0.5  # sec
MAX_FPS = 10


class SpeakerType(enum.Enum):
    TOA = "TOA"


class SpeakerMsgType(enum.Enum):
    HTTP = 0


DEFAULT_TRT_MAX_BATCH_ALLOWED = 8


# Message acknolage sent when all models are up
class e_MsgUpdate(enum.IntEnum):
    READY = (0,)
    ERR = (1,)
    GotThumbnails = (2,)
    NoThumbnailsToAnalyzer = (3,)
    Stopped = (4,)
    Ack = (5,)
    Progress = (6,)


# Message acknolage sent when all models are up
class e_HTTPMethod(enum.IntEnum):
    GET = (0,)
    POST = (1,)


class e_HTTPCredentials(enum.IntEnum):
    NONE = (0,)
    BASIC = (1,)
    DIGEST = (2,)


class BiometricProperty(str, enum.Enum):
    GENDER = "genderClassification"
    FACE = "faceRecognition"


def bitstring_to_bytes(s):
    # v = int(s, 2)
    # b = bytearray()
    # while v:
    #    b.append(v & 0xff)
    #    v >>= 8
    # return bytes(b[::-1])
    return bytearray([int(s[i : i + 8], 2) for i in range(0, len(s), 8)])


class e_ControllerStatus(enum.Enum):
    NO_CALL_NO_CHANGE = "00"
    CALL_NO_CHANGE = "01"
    NO_CALL_CHANGE = "10"
    CALL_CHANGE = "11"


class PrivacyType(enum.Enum):
    PIXELATE = "pixelate"
    BLACKEN = "blacken"


class InferenceType(str, enum.Enum):

    NONE = ""
    DETECTION = "detection"
    IMAGE_ENCODER = "enc"
    PERSON_REID = "pa-reid"
    PERSON_ATTR = "pa"
    SFACE = "sface"
    YUNET = "yunet"
    WEAPONS_CLASSIFICATION = "wc"
    DOORS_CLASSIFICATION = "doors"
    PPE_ATTR = "ppe"
    LPC_ATTR = "lpc"
    RETINA_FACE = "retinaface"
    CRAFT = "craft"
    CRNN = "crnn"
    VEHICLE = "vehicle"
    RTMPOSE = "rtmpose"
    MOTIONBERT = "motion-bert"
    STGCN = "stgcn"
    MAGFACE = "magface"
    EDIFFIQA = "ediffiqa"
    ARCFACE = "arcface"
    CLIP_ENCODER = "clip"
    TRACKER_PERSON_REID = "tracker-person-reid"
    TRACKER_VEHICLE_REID = "tracker-vehicle-reid"
    HANDS = "hands"
    LICENSE_PLATE = ("license-plate",)
    YOLOV5 = "yolov5"
    YOLOV8 = "yolov8"
    EXPERT_DETECT = "expert-detect"
    WEAPON_EXPERT = "weapon-expert"
    VIOLENCE = "violence"
    VIOLENCE_DETECT = "violence-detect"
    STATE_OBJECT = "state-object"
    WEBFACE = "webface"
    IMAGEQUALITY = "image-quality"
    PHONE = "phone"
    CONTAINER_OCD = "container-ocd"
    CONTAINER_OCR = "container-ocr"


class CameraType(str, enum.Enum):
    CEILING_MOUNTED_FISHEYE = ("ceiling-mounted-fishEye",)
    WALL_MOUNTED_FISHEYE = ("wall-mounted-fishEye",)
    THERMAL = ("thermal-camera",)
    STATIC = ("static-camera",)
    PTZ = ("ptz-camera",)
    MOBILE = ("mobile-camera",)
    CEILING_MOUNTED = "ceiling-mounted"


camera_type_to_location_center = {
    CameraType.CEILING_MOUNTED.value: True,
    CameraType.CEILING_MOUNTED_FISHEYE.value: True,
}

config = proj.load_config()
CROP_GB = 0.05


def JpegCompBytes(image, jpegQuality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpegQuality]
    image_bytes = cv2.imencode(".jpg", image, encode_param)[1].tobytes()
    return image_bytes


def Perimeter(bboxes, size):
    perimeter = {}
    perimeter["topLeft"] = {}
    perimeter["bottomRight"] = {}

    perimeter["topLeft"]["x"] = round(float(bboxes[0]) / size[1], 3)
    perimeter["topLeft"]["y"] = round(float(bboxes[1]) / size[0], 3)
    perimeter["bottomRight"]["x"] = round(float(bboxes[2]) / size[1], 3)
    perimeter["bottomRight"]["y"] = round(float(bboxes[3]) / size[0], 3)

    return perimeter


def getTrafficPath(markedPositions, shape):
    locMarkedPositions = {i: markedPositions[i] for i in markedPositions if i != "positionsForUpdate"}
    sortedPositions = sorted(locMarkedPositions.values(), key=lambda x: x["startTime"])
    sortedIndexes = [pos["index"] for pos in sortedPositions]
    # sortedScaledIndexes =  list(OrderedDict.fromkeys([posConvert(pos,shape,downScale = True)  for pos in sortedIndexes]))
    # trafficPath         =                            [posConvert(pos,shape,downScale = False) for pos in sortedScaledIndexes]
    # return trafficPath
    return sortedIndexes


def roi_pos(idx, shape):
    xidx = int(idx % shape[0])
    yidx = int(idx / shape[1])
    return xidx, yidx


def roi_gen(roi1d):
    roi = np.zeros(ROI_SHAPE, dtype=bool)
    for idx in roi1d:
        xidx, yidx = roi_pos(idx, ROI_SHAPE)
        roi[xidx, yidx] = True
    return roi


def inRoi(filter, position):
    if not (filter.roiFilter):
        return True
    x_pos = int(ROI_SHAPE[0] * position["x"])
    y_pos = int(ROI_SHAPE[1] * position["y"])
    return filter.roi[x_pos, y_pos]


def bboxesInRois(filters, perimeter, obj):
    for filter in filters:
        if bboxesInRoi(filter, perimeter, obj):
            return True
    return False


def bboxesInRoi(filter, perimeter, obj):
    # if its not in teh filter we are not looking for this object
    if not (obj in filter):
        return False

    # if filter is disabled we are looking on entire FOV
    if not (filter[obj].roiFilter):
        return True

    # check that one of the points in roi
    # topLeft
    x_pos = min(int(ROI_SHAPE[0] * perimeter["topLeft"]["x"]), ROI_SHAPE[0] - 1)
    y_pos = min(int(ROI_SHAPE[1] * perimeter["topLeft"]["y"]), ROI_SHAPE[1] - 1)

    if filter[obj].roi[x_pos, y_pos]:
        return True

    # bottomRight
    x_pos = min(int(ROI_SHAPE[0] * perimeter["bottomRight"]["x"]), ROI_SHAPE[0] - 1)
    y_pos = min(int(ROI_SHAPE[1] * perimeter["bottomRight"]["y"]), ROI_SHAPE[1] - 1)
    if filter[obj].roi[x_pos, y_pos]:
        return True

    # bottomRight
    x_pos = min(int(ROI_SHAPE[0] * perimeter["bottomRight"]["x"]), ROI_SHAPE[0] - 1)
    y_pos = min(int(ROI_SHAPE[1] * perimeter["topLeft"]["y"]), ROI_SHAPE[1] - 1)
    if filter[obj].roi[x_pos, y_pos]:
        return True

    # topLeft
    x_pos = min(int(ROI_SHAPE[0] * perimeter["topLeft"]["x"]), ROI_SHAPE[0] - 1)
    y_pos = min(int(ROI_SHAPE[1] * perimeter["topLeft"]["y"]), ROI_SHAPE[1] - 1)
    if filter[obj].roi[x_pos, y_pos]:
        return True

    # if we got here non of the points is inside
    return False


def Position(bboxes, size):
    x = round(float(((bboxes[0] + bboxes[2]) / 2) / size[1]), 3)
    y = round(float(((bboxes[1] + bboxes[3]) / 2) / size[0]), 3)
    Position = {}
    Position["x"] = x
    Position["y"] = y
    return Position


# line crosing
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def SendUDP(UDP_IP, UDP_PORT, MESSAGE):
    t0 = time_sync()
    bytesToSend = str.encode(MESSAGE)
    serverAddressPort = (UDP_IP, int(UDP_PORT))
    # Create a UDP socket at client side
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    # Send to server using created UDP socket
    UDPClientSocket.sendto(bytesToSend, serverAddressPort)
    t1 = time_sync()

    logger.info(f"Sent UDP: {MESSAGE} latency: {int((t1 - t0) * 1000)} [ms]")


def SendTCP(TCP_IP, TCP_PORT, MESSAGE):
    # Create a socket (SOCK_STREAM means a TCP socket)
    t0 = time_sync()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((TCP_IP, TCP_PORT))
        # TODO add timeouts
        sock.sendall(bytes(MESSAGE + "\n", "utf-8"))
        t1 = time_sync()

        # Receive data from the server and shut down
        # TODO: Do we need to get response, maybe add to alert Message?
        received = str(sock.recv(1024), "utf-8")
        sock.close()
        t2 = time_sync()

    logger.info(f"Sent TCP: {MESSAGE} latency: {int((t1 - t0) * 1000)} [ms]")
    logger.info(f"Sent TCP: {received} latency: {int((t2 - t1) * 1000)} [ms]")


def sendHTTP(http, auth=None):
    t0 = time_sync()
    if auth is not None:
        if auth["type"] == e_HTTPCredentials.BASIC.value:
            auth = HTTPBasicAuth(auth["username"], auth["password"])
        elif auth["type"] == e_HTTPCredentials.DIGEST.value:
            auth = HTTPDigestAuth(auth["username"], auth["password"])
        else:
            auth = None

    if http.method == e_HTTPMethod.POST.value:
        r = requests.post(url=http.url, data=http.body, auth=auth, timeout=HTTP_CONNECT_TIMEOUT)
        t1 = time_sync()
        logger.info(f"Sent HTTP POST: {http.url}. Content: {r.text} latency: {int((t1 - t0) * 1000)} [ms]")

    else:
        r = requests.get(url=http.url, data=http.body, auth=auth, timeout=HTTP_CONNECT_TIMEOUT)
        t1 = time_sync()
        logger.info(f"Sent HTTP GET: {http.url}. Content: {r.json()} latency: {int((t1 - t0) * 1000)} [ms]")


def bold_http_post_without(http, alert_info):
    data = http.body
    data["comment"] = alert_info.msg
    t0 = time_sync()
    try:
        response = requests.post(http.url, headers=http.headers, json=data, timeout=HTTP_CONNECT_TIMEOUT)
        t1 = time_sync()
        logger.info(f"Sent HTTP POST: {http.url}. Content: {response.text} latency: {int((t1 - t0) * 1000)} [ms]")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred on {http}: {e}")


def sendBOLD(http, alert_info):
    thread = threading.Thread(target=bold_http_post_without, args=(http, alert_info))
    thread.start()


def SendCEF(msgAction, alert_info):
    cef_logger = msgAction.cef_logger
    cef_message = msgAction.data_header + alert_info.msg + msgAction.data_footer

    # Send the CEF message
    cef_logger.info(cef_message)


def speaker_builder(speaker, pattern):
    speaker_type = speaker["model"]
    if speaker_type == SpeakerType.TOA.value:
        url = f'http://{speaker["ipAddress"]}/api/v1/pattern/play?pattern_number={int(pattern) + 1}'
        auth = HTTPDigestAuth(speaker["username"], speaker["password"])
        msg_type = SpeakerMsgType.HTTP

        msg_type = SpeakerMsgType.HTTP
        action = Namespace(url=url, msg_type=msg_type, auth=auth)
    else:
        action = None
    return action


def playSpeaker(speaker):
    if speaker is None:
        logger.error(f"Speaker msg is None")
        return
    else:
        if speaker.msg_type == SpeakerMsgType.HTTP:
            try:
                logger.info(f"sending speaker msg: {speaker}")
                response = requests.get(speaker.url, auth=speaker.auth)
                response.raise_for_status()
                logger.info(response.text)
            except requests.exceptions.RequestException as e:
                logger.error(f"Speaker error occurred: {e}")
        else:
            logger.error(f"Un supported speaker msg")


def lineCrossingCenter(perimeter, position):
    if position == "bottom":
        x = (perimeter["bottomRight"]["x"] + perimeter["topLeft"]["x"]) / 2
        y = perimeter["bottomRight"]["y"]
    elif position == "top":
        x = (perimeter["bottomRight"]["x"] + perimeter["topLeft"]["x"]) / 2
        y = perimeter["topLeft"]["y"]
    elif position == "right":
        x = perimeter["bottomRight"]["x"]
        y = (perimeter["bottomRight"]["y"] + perimeter["topLeft"]["y"]) / 2
    elif position == "left":
        x = perimeter["topLeft"]["x"]
        y = (perimeter["bottomRight"]["y"] + perimeter["topLeft"]["y"]) / 2
    else:  # center
        x = (perimeter["bottomRight"]["x"] + perimeter["topLeft"]["x"]) / 2
        y = (perimeter["bottomRight"]["y"] + perimeter["topLeft"]["y"]) / 2

    point = Point(x, y)
    return point


def bbxSide(p1, p2, d):
    if d == 0:
        return "center"

    # find line angle
    # order lines per x

    # y is opposite, work with positive angles
    angle = math.degrees(math.atan2((p2.y - p1.y), (p1.x - p2.x))) % 360

    if 0 <= angle <= 45:
        if d == 1:
            return "bottom"
        else:
            return "top"

    elif 45 < angle <= 135:
        if d == 1:
            return "right"
        else:
            return "left"

    elif 135 < angle <= 225:
        if d == 1:
            return "top"
        else:
            return "bottom"

    elif 225 < angle <= 315:
        if d == 1:
            return "left"
        else:
            return "right"

    # similar to starting
    elif 315 < angle:
        if d == 1:
            return "bottom"
        else:
            return "top"

        # Given three collinear points p, q, r, the function checks if


# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y)):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points - on th line
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    return np.sign((float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)))


def recursive_update(d, u):
    """recursive update of dictionary"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
# if d is set than there is a direction test as well
def doIntersect(p1, q1, p2, q2, d):
    # find the first orientation to see if
    # we are on the same direction

    o1 = orientation(p1, q1, p2)
    if not (d == 0):
        # We have orientation test to make
        if (o1 == d) and not (o1 == 0):
            # Starting point isnt on the same side
            return 0

    # if we got here we passed orientation test for general case
    # Find the rest 3 orientations required for
    # the general and special cases
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    direction = int(np.sign(o2 + (-o1)))

    # General case
    if (o1 != o2) and (o3 != o4):
        return direction

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and onSegment(p1, p2, q1):
        # Here we need to make sure that q2 is opposite to direction
        if (d == 0) or (d == o2):
            return direction
        else:
            return 0

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and onSegment(p1, q2, q1):
        return direction

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and onSegment(p2, p1, q2):
        return direction

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and onSegment(p2, q1, q2):
        return direction

    # If none of the cases
    return 0


# Python program to find all
# rectangles filled with 0
def findend(i, j, a, output, index):
    x = len(a)
    y = len(a[0])

    # flag to check column edge case,
    # initializing with 0
    flagc = 0

    # flag to check row edge case,
    # initializing with 0
    flagr = 0

    for m in range(i, x):

        # loop breaks where first 1 encounters
        if a[m][j] == 1:
            flagr = 1  # set the flag
            break

        # pass because already processed
        if a[m][j] == 5:
            pass

        for n in range(j, y):

            # loop breaks where first 1 encounters
            if a[m][n] == 1:
                flagc = 1  # set the flag
                break

            # fill rectangle elements with any
            # number so that we can exclude
            # next time
            a[m][n] = 5

    if flagr == 1:
        output[index].append(m - 1)
    else:
        # when end point touch the boundary
        output[index].append(m)

    if flagc == 1:
        output[index].append(n - 1)
    else:
        # when end point touch the boundary
        output[index].append(n)


def get_rectangle_coordinates(a):
    # retrieving the column size of array
    size_of_array = len(a)

    # output array where we are going
    # to store our output
    output = []

    # It will be used for storing start
    # and end location in the same index
    index = -1

    for i in range(0, size_of_array):
        for j in range(0, len(a[0])):
            if a[i][j] == 0:
                # storing initial position
                # of rectangle
                output.append([i, j])

                # will be used for the
                # last position
                index = index + 1
                findend(i, j, a, output, index)

    return output


def extract_num_classes(filename):
    match = re.search(r"-(\d+)cls", filename)
    if match:
        return int(match.group(1))
    else:
        return None


def convert(arr):
    res = np.zeros((len(arr), len(arr[0])))
    for x in range(len(arr)):
        for y in range(len(arr[0])):
            if not (arr[x][y]):
                res[x][y] = 1
    return res


def encode_debug_data(data: Dict[str, Any]) -> str:
    import msgpack
    import msgpack_numpy as m

    try:
        m.patch()
        serialized = msgpack.packb(data, use_bin_type=True)
    except Exception as e:
        logger.error(f"Serialization failed: {e}")
        raise
    return base64.b64encode(serialized).decode("ascii")


def decode_debug_data(data_str: str) -> Dict[str, Any]:
    """Decode from base64 string and deserialize"""
    import msgpack
    import msgpack_numpy as m

    if not data_str:
        return {}
    try:
        m.patch()
        decoded = base64.b64decode(data_str.encode("ascii"))
        deserialized = msgpack.unpackb(decoded, raw=False)
    except Exception as e:
        logger.error(f"Deserialization failed: {e}")
        raise
    return deserialized


instance_id = str(os.environ.get("INSTANCE_ID", 1))

global logger
logger = get_logger(
    log_dir="/usr/src/app/logs/",
    log_file_name=f"analyzer_manager{instance_id}",
    log_level=config["logger"]["analyticLoggerLevel"],
    logger_days=int(config["logger"]["analyticLoggerDays"][:-1]),
)


def log_exception(log, message: str, ex: Exception):
    exception_traceback = traceback.format_exc()
    log.error(f"{message}: {str(ex)}, traceback: {exception_traceback}")


DEFAULT_DETECTOR_IM_SIZE = [384, 640]


def max_per_id_mask(ids: np.ndarray, conf: np.ndarray) -> np.ndarray:
    # Map each id to a group index
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    # Find max confidence per group
    max_conf = np.full(unique_ids.shape, -np.inf)
    np.maximum.at(max_conf, inverse, conf)
    # Find mask where conf equals group max
    mask = (conf == max_conf[inverse]).astype(int)
    return mask
