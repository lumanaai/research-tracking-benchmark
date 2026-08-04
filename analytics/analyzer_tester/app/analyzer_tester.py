# limit the number of cpus used by high performance libraries
import shutil
from audioop import ratecv
import json
from pathlib import Path

import numpy as np
import pika
import time
import numpy
import threading
import sys
import os
import struct
import cv2
import argparse
import glob
import enum
import subprocess
import base64


# Message acknolage sent when all models are up
class e_CameraStatus(enum.IntEnum):
    Online = 0
    Offline = 1
    Stopped = 2
    Unknown = 3
    Init = 4


class e_ComponentStatus (enum.Enum):
    Available = "Available",
    Unavailable = "Unavailable"


#Config stage
config         = json.load(open('/usr/src/app/configs/config.json'))
configAnalytic = json.load(open('/usr/src/app/configs/configAnalytic.json'))

DeviceID  = 'test_device'
CameraID  = 'test_camera'
ImageName = 'yuv_images'
ThumbName = 'thumb_images'
JsonName  = 'analytic'
MotName   = 'motion'
TrainName = 'training'
VarName   = 'variables'
variables = {}

ImageQueueName  = ImageName + '_' + DeviceID +'_' + CameraID
ThumbQueueName  = ThumbName + '_' + DeviceID + '_' + CameraID
JsonQueueName   = JsonName  + '_' + DeviceID +'_' + CameraID
MotQueueName    = MotName   + '_' + DeviceID +'_' + CameraID
TraQueueName    = TrainName + '_' + DeviceID +'_' + CameraID
variablesQueue  = VarName   + '_' + DeviceID +'_' + CameraID

parameters = pika.ConnectionParameters(host='rabbitMQ')
connection = pika.BlockingConnection(parameters)
channel = connection.channel()


burst_test = False
gpu_pointer = False
jetson = False
if os.getenv('BURST_TEST'):  burst_test  = (os.getenv('BURST_TEST')=='True')
if os.getenv('GPU_POINTER_PIPELINE'): gpu_pointer  = (os.getenv('GPU_POINTER_PIPELINE') == 'True')
if os.getenv('JETSON'): jetson  = (os.getenv('JETSON') == 'True')

SentImages = 0
RcvdImages = 0
RcvdTrain = 0
RcvdJson   = 0
RcvdMotion = 0
resultpath = ''

info_list = []
objects_list = []
motion_list = []

topRightROI = [54,55,56,57,58,59,60,61,62,63,86,87,88,89,90,91,92,93,94,95,118,119,120,121,122,123,124,125,126,127,150,151,152,153,154,155,156,157,158,159,182,183,184,185,186,187,188,189,190,191,214,215,216,217,218,219,220,221,222,223,246,247,248,249,250,251,252,253,254,255,278,279,280,281,282,283,284,285,286,287,310,311,312,313,314,315,316,317,318,319,342,343,344,345,346,347,348,349,350,351,374,375,376,377,378,379,380,381,382,383,406,407,408,409,410,411,412,413,414,415,438,439,440,441,442,443,444,445,446,447,469,470,471,472,473,474,475,476,477,478,479,501,502,503,504,505,506,507,508,509,510,511,541,542,543]


objectNames = enum.Enum(
    "objectNames",
    [
        "person",
        "vehicle",
        "pet",
        "gun"
    ],
    start=0,
)
names = enum.Enum('names',['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush', 'vehicle', 'pet'],start=0)  # class names


alertsType = enum.Enum(
    "alerts",
    ["appearance", "disappeare", "loitering", "lineCrossing", "lpr", "faceDetection", "videoTampering", "motionScore", "occupancy", "movement", "trafficControl", "tailgating"],
    start=0,
)

alertsActions = enum.Enum(
    "alertsActions",
    ["add", "remove", "update"],
    start=0,
)


class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y


def CreateZone(ul,dr,w,h):
	z = numpy.zeros((w,h), dtype=bool)
	z[ul.x:dr.x,dr.y:ul.y] = numpy.ones((dr.x-ul.x,ul.y - dr.y), dtype=bool)
	z = z.tolist()
	return z

esp_alert_l = {'action': 2, '_id': '63e2bd8d84ff2e2945fb53de', 'alertType': 0, 'selectedCamera': {'locationId': '63b74bfcfaf2c60c23b8fdcf', 'cameraId': '63e2aa4284ff2e2945fb53db', 'edgeId': '635730eef5dee0ccdd0c7a63', 'timezone': 'America/Chicago'}, 'configuration': {'object': 1, 'detection': 4, 'detectionAdditionalAttributes': {'direction': 'above', 'count': 0, 'sensitivity': 0.5}, 'filters': {'greenList': ' ', 'redList': '', 'unrecognized': True}, 'tresholdTime': 0}, 'settings': {'schedule': {'monday': {'from': '12:00 AM', 'to': '11:59 PM'}, 'tuesday': {'from': '12:00 AM', 'to': '11:59 PM'}, 'wednesday': {'from': '12:00 AM', 'to': '11:59 PM'}, 'thursday': {'from': '12:00 AM', 'to': '11:59 PM'}, 'friday': {'from': '12:00 AM', 'to': '11:59 PM'}, 'saturday': {'from': '12:00 AM', 'to': '11:59 PM'}, 'sunday': {'from': '12:00 AM', 'to': '11:59 PM'}}, 'frequency': 0, 'additionalOptions': False, 'pushAlert': True, 'blockNotificationPeriod': None, 'enableNotificationSound': False, 'alertThumbnail': True, 'alertZoomThumbnail': True}, 'markedIdx': [363, 364, 365, 366, 367, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 747, 748, 749, 750, 751, 752, 753, 754, 755, 780, 781, 782, 783, 784, 785, 786, 787, 813, 814, 815, 816, 817, 818, 846, 847, 848, 849, 850, 880, 881], 'zones': {'{"x":0.07314629258517034,"y":0.5035460992907801}': {'name': '', 'color': 'green', 'selection': [{'x': 0.07314629258517034, 'y': 0.5035460992907801}, {'x': 0.5360721442885772, 'y': 0.8971631205673759}, {'x': 0.8016032064128257, 'y': 0.4716312056737589}, {'x': 0.40080160320641284, 'y': 0.3280141843971631}], 'markedIdx': [363, 364, 365, 366, 367, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 747, 748, 749, 750, 751, 752, 753, 754, 755, 780, 781, 782, 783, 784, 785, 786, 787, 813, 814, 815, 816, 817, 818, 846, 847, 848, 849, 850, 880, 881]}}, 'enabled': True, 'actions': {'gpioActions': [], 'msgActions': [], 'directActions': []}, 'measureCrossZones': False}

#occupancy_2 = {"_id":"640a1776733835ddbebbaca0","name":"crosswalk","alertType":0,"configuration":{"object":0,"offender":None,"detection":8,"detectionAdditionalAttributes":{"direction":"above","count":1,"sensitivity":0.5},"filters":{"ageType":[],"carryingType":[],"lowerbodyType":[],"upperbodyType":[],"accessoryType":[],"footwearType":[],"hairType":[],"genderType":[],"upperbodyColor":[],"lowerbodyColor":[],"hairColor":[],"footwearColor":[]},"tresholdTime":0},"settings":{"schedule":{"monday":{"from":"12:00 AM","to":"11:59 PM"},"tuesday":{"from":"12:00 AM","to":"11:59 PM"},"wednesday":{"from":"12:00 AM","to":"11:59 PM"},"thursday":{"from":"12:00 AM","to":"11:59 PM"},"friday":{"from":"12:00 AM","to":"11:59 PM"},"saturday":{"from":"12:00 AM","to":"11:59 PM"},"sunday":{"from":"12:00 AM","to":"11:59 PM"}},"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":1,"enableNotificationSound":True,"alertThumbnail":True,"alertZoomThumbnail":True},"notifications":{"orgUsers":[{"id":"640a0a13733835ddbebbac96","firstname":"Martin","lastname":"Avegno","email":"martin@activesolutionsusa.com","phone":None,"roles":["owner"],"status":0}],"manualUsers":[],"notificationMethods":{"640a0a13733835ddbebbac96":"email"}},"timezone":"America/Chicago","actions":{"gpioActions":[{"id":0,"action":False}],"msgActions":[],"directActions":[]},"zones":{"{\"x\":0.3778307508939213,\"y\":0.6908602150537635}":{"name":"crosswalk1","color":"green","selection":[{"x":0.3778307508939213,"y":0.6908602150537635},{"x":0.6924910607866508,"y":0.6639784946236559},{"x":0.5375446960667462,"y":0.1827956989247312},{"x":0.40762812872467225,"y":0.20698924731182797}],"markedIdx":[205,206,207,208,237,238,239,240,241,269,270,271,272,273,301,302,303,304,305,333,334,335,336,337,338,365,366,367,368,369,370,397,398,399,400,401,402,429,430,431,432,433,434,435,461,462,463,464,465,466,467,492,493,494,495,496,497,498,499,524,525,526,527,528,529,530,531,532,556,557,558,559,560,561,562,563,564,588,589,590,591,592,593,594,595,596,620,621,622,623,624,625,626,627,628,629,652,653,654,655,656,657,658,659,660,661,684,685,686,687,688,689,690,691]},"{\"x\":0.7661764705882353,\"y\":0.6701298701298701}":{"name":"","color":"blue","selection":[{"x":0.7661764705882353,"y":0.6701298701298701},{"x":0.8779411764705882,"y":0.987012987012987},{"x":0.9897058823529412,"y":0.974025974025974},{"x":0.8823529411764706,"y":0.5948051948051948}],"markedIdx":[635,666,667,668,696,697,698,699,700,729,730,731,732,761,762,763,764,765,794,795,796,797,826,827,828,829,858,859,860,861,891,892,893,894,923,924,925,926,955,956,957,958,988,989,990,991,1020,1021]},"{\"x\":0.7911764705882353,\"y\":0.11948051948051948}":{"name":"","color":"yellow","selection":[{"x":0.7911764705882353,"y":0.11948051948051948},{"x":0.7573529411764706,"y":0.12207792207792208}]},"{\"x\":0.4411764705882353,\"y\":0.09090909090909091}":{"name":"","color":"purple","selection":[{"x":0.4411764705882353,"y":0.09090909090909091}]},"{\"x\":0.5029411764705882,\"y\":0.11688311688311688}":{"name":"","color":"cyan","selection":[{"x":0.5029411764705882,"y":0.11688311688311688}]},"{\"x\":0.4411764705882353,\"y\":0.13506493506493505}":{"name":"","color":"green","selection":[{"x":0.4411764705882353,"y":0.13506493506493505},{"x":0.47352941176470587,"y":0.07532467532467532},{"x":0.4985294117647059,"y":0.08051948051948052},{"x":0.4808823529411765,"y":0.13246753246753246}],"markedIdx":[79,110,111]}},"definedZones":True,"markedIdx":[79,110,111,121,143,205,206,207,208,237,238,239,240,241,269,270,271,272,273,301,302,303,304,305,333,334,335,336,337,338,365,366,367,368,369,370,397,398,399,400,401,402,429,430,431,432,433,434,435,460,461,462,463,464,465,466,467,492,493,494,495,496,497,498,499,524,525,526,527,528,529,530,531,532,556,557,558,559,560,561,562,563,564,588,589,590,591,592,593,594,595,596,620,621,622,623,624,625,626,627,628,629,635,652,653,654,655,656,657,658,659,660,661,666,667,668,684,685,686,687,688,689,690,691,696,697,698,699,700,729,730,731,732,761,762,763,764,765,794,795,796,797,826,827,828,829,858,859,860,861,891,892,893,894,923,924,925,926,955,956,957,958,988,989,990,991,1020,1021,1023],"measureCrossZones":False,"selectedCamera":{"locationId":"640a0d96733835ddbebbac99","cameraId":"640a13ea733835ddbebbac9f","edgeId":"63ceb52d036b9d5e670d3bac","timezone":"America/Chicago"},"synced":True,"action":2,"orgId":"640a0a36733835ddbebbac97","enabled":True,"groupId":"71256e8e-c89a-476d-b730-a1fbc5c40217","version":"0.1.55"}
occupancy_alert = {"_id":"personDetectionAbove5_ID",
                    "action":   alertsActions.add.value,
                    "configuration": {
                         "object": objectNames.person.value,
                         "detection":alertsType.occupancy.value,
                         "detectionAdditionalAttributes": {
                            #"direction": "above",
                            "count":5
                            }
                    },
                    "settings": {
                        "blockNotificationPeriod": 5
                    }
}
multi_occupancy_alert = {"_id":"personDetectionAbove5_ID",
                    "action":   alertsActions.add.value,
                    "configuration": {
                         "objects": [{"object": objectNames.person.value},{"object": objectNames.vehicle.value}],
                         "detection":alertsType.occupancy.value,
                         "detectionAdditionalAttributes": {
                            #"direction": "above",
                            "count":5
                            }
                    },
                    "settings": {
                        "blockNotificationPeriod": 5
                    }
}

linecross_1_alert = {"action": 0,
                            "_id": 'linecross_1_alert',
                            "action":   alertsActions.add.value,
                            "alertType": 0,
                            "configuration":
                                {'object': objectNames.person.value,
                                'detection': alertsType.lineCrossing.value,
                                "detectionAdditionalAttributes": {
                                    "direction": "below",
                                    "count":3
                                    }
                                },
                            'settings': {
                                'additionalOptions': False,
                                "alertThumbnail":True,"alertZoomThumbnail":True,
                                'blockNotificationPeriod': None},
                            'lineCrossing': {"p1":{"x":0.08,"y":0.42},"p2":{"x":0.92,"y":0.41},"d":0,"state":2},
                            "enabled": True
                            }

linecross_2_alert = {"action": 0,
                            "_id": 'linecross_2_alert',
                            "action":   alertsActions.add.value,
                            "alertType": 0,
                            "configuration":
                                {'object': objectNames.person.value,
                                'detection': alertsType.lineCrossing.value,
                                "detectionAdditionalAttributes": {
                                    "direction": "below",
                                    "count":3
                                    }
                                },
                            'settings': {
                                'additionalOptions': False,
                                'blockNotificationPeriod': None,"alertThumbnail":True,"alertZoomThumbnail":True},
                            'lineCrossing': {"p1":{"x":0.1,"y":0.82},"p2":{"x":0.4,"y":0.81},"d":2,"state":2},
                            "enabled": True
                            }
#line_3 = {"_id":"643f7f22589eec72287dfadd","name":"temp","alertType":0,"configuration":{"objects":[0],"offender":None,"detection":3,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},"filters":{},"tresholdTime":0},"settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":None,"enableNotificationSound":False,"alertThumbnail":True,"alertZoomThumbnail":True},"notifications":{"orgUsers":[],"manualUsers":[],"notificationMethods":{}},"timezone":"Asia/Beirut","actions":{"gpioActions":[],"msgActions":[],"directActions":[]},"lineCrossing":{"p1":{"x":0.14,"y":0.68},"p2":{"x":0.72,"y":0.65},"d":2,"state":2},"selectedCamera":{"locationId":"643e3760589eec72287dfab4","cameraId":"643e4982589eec72287dfac0","edgeId":"643e34b2589eec72287dfab3","timezone":"Asia/Beirut"},"synced":True,"action":0,"orgId":"63bd6ed3d67d82346855d43a","enabled":True,"groupId":"48d63d8a-428b-4ad3-bed2-49b42d681038","version":"1.0.0"}


simultanous_alert = {"_id":"simulatananana",
                    "action":   alertsActions.add.value,
                    "configuration": {
                         "object": objectNames.person.value,
                         "detection":alertsType.appearance.value,
                         "objects": [
                             {
                                 "object": objectNames.person.value
                             },
                             {
                                 "object": objectNames.vehicle.value
                             },
                         ]

                    },
                    "settings": {
                        "blockNotificationPeriod": 0,
                        "alertZoomThumbnail": True,
                        "alertThumbnail": True
                    },
#"zones":{"{\"x\":0.12625250501002003,\"y\":0.49113475177304966}":{"name":"","color":"green","selection":[{"x":0.12625250501002003,"y":0.49113475177304966},{"x":0.24048096192384769,"y":0.2854609929078014},{"x":0.6973947895791583,"y":0.28191489361702127},{"x":0.7905811623246493,"y":0.45567375886524825},{"x":0.7985971943887775,"y":0.4858156028368794}],"markedIdx":[295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505]}},"definedZones":True,"markedIdx":[295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505],"measureCrossZones":False,"selectedCamera":{"locationId":"63869dfe608d754ab8932e9e","cameraId":"63876c93608d754ab8932ec0","edgeId":"6356f7b0f5dee0ccdd0c7a61","timezone":"US/Central"},"synced":True,"action":0,"orgId":"63869da3608d754ab8932e9d","enabled":True,"groupId":"a01521b6-b196-4238-baf5-5f3b399baf2f","version":"0.1.51"}
"zones":{"{\"x\":0.05911823647294589,\"y\":0.43439716312056736}":{"name":"","color":"green","selection":[{"x":0.05911823647294589,"y":0.43439716312056736},{"x":0.17234468937875752,"y":0.3120567375886525},{"x":0.9859719438877755,"y":0.2907801418439716},{"x":0.9909819639278558,"y":0.4734042553191489}],"markedIdx":[311,312,313,314,315,316,317,318,319,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479]}},"definedZones":True,"markedIdx":[310,311,312,313,314,315,316,317,318,319,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479],"measureCrossZones":False,"selectedCamera":{"locationId":"63869dfe608d754ab8932e9e","cameraId":"63876c93608d754ab8932ec0","edgeId":"6356f7b0f5dee0ccdd0c7a61","timezone":"US/Central"},"synced":True,"action":0,"orgId":"63869da3608d754ab8932e9d","enabled":True,"groupId":"797bc0c7-cb02-43b8-b6bf-54a73f6d4770","version":"0.1.51"}

simultanous_alert = {'action': 2, '_id': '63f4835beadf3e1f48a54bb7', 'alertType': 0, 'selectedCamera': {'locationId': '63627fbc28f6e6f856aa4884', 'cameraId': '63f48060eadf3e1f48a54bb6', 'edgeId': '63e236d45951bba3f62f14c8', 'timezone': 'Asia/Jerusalem'}, 'configuration': {'object': 0, 'objects': [{'object': 0}, {'object': 1}], 'detection': 0, 'detectionAdditionalAttributes': {'direction': 'above', 'count': 0, 'sensitivity': 0.5}, 'tresholdTime': 0}, 'settings': {'schedule': {'monday': None, 'tuesday': None, 'wednesday': None, 'thursday': None, 'friday': None, 'saturday': None, 'sunday': None}, 'frequency': 0, 'additionalOptions': False, 'pushAlert': True, 'blockNotificationPeriod': None, 'enableNotificationSound': False, 'alertThumbnail': True, 'alertZoomThumbnail': True}, 'markedIdx': [492, 522, 523, 524, 525, 526, 552, 553, 554, 555, 556, 557, 558, 559, 560, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 1010, 1011], 'zones': {'{"x":0.251503006012024,"y":0.5475113122171946}': {'name': '', 'color': 'green', 'selection': [{'x': 0.251503006012024, 'y': 0.5475113122171946}, {'x': 0.39579158316633267, 'y': 0.4841628959276018}, {'x': 0.42985971943887774, 'y': 0.48868778280542985}, {'x': 0.4759519038076152, 'y': 0.5294117647058824}, {'x': 0.501002004008016, 'y': 0.5407239819004525}, {'x': 0.5210420841683366, 'y': 0.5361990950226244}, {'x': 0.5370741482965932, 'y': 0.5475113122171946}, {'x': 0.5621242484969939, 'y': 0.5859728506787331}, {'x': 0.5801603206412825, 'y': 0.6470588235294118}, {'x': 0.6022044088176353, 'y': 0.669683257918552}, {'x': 0.6322645290581163, 'y': 0.6832579185520362}, {'x': 0.6282565130260521, 'y': 0.8212669683257918}, {'x': 0.6232464929859719, 'y': 0.9276018099547512}, {'x': 0.6102204408817635, 'y': 0.9841628959276018}, {'x': 0.21342685370741482, 'y': 0.9592760180995475}], 'markedIdx': [492, 522, 523, 524, 525, 526, 552, 553, 554, 555, 556, 557, 558, 559, 560, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 1010, 1011]}}, 'enabled': True, 'actions': {'gpioActions': [], 'msgActions': [], 'directActions': []}, 'measureCrossZones': False}

simultanous_alert = {'action': 2, '_id': '63f4835beadf3e1f48a54bb7', 'alertType': 0, 'selectedCamera': {'locationId': '63627fbc28f6e6f856aa4884', 'cameraId': '63f48060eadf3e1f48a54bb6', 'edgeId': '63e236d45951bba3f62f14c8', 'timezone': 'Asia/Jerusalem'}, 'configuration': {'objects': [{'object': 0}, {'object': 1}], 'detection': 0, 'detectionAdditionalAttributes': {'direction': 'above', 'count': 0, 'sensitivity': 0.5}, 'tresholdTime': 0}, 'settings': {'schedule': {'monday': None, 'tuesday': None, 'wednesday': None, 'thursday': None, 'friday': None, 'saturday': None, 'sunday': None}, 'frequency': 0, 'additionalOptions': False, 'pushAlert': True, 'blockNotificationPeriod': None, 'enableNotificationSound': False, 'alertThumbnail': True, 'alertZoomThumbnail': True}, 'markedIdx': [355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 735], 'zones': {'{"x":0.033066132264529056,"y":0.36425339366515835}': {'name': '', 'color': 'green', 'selection': [{'x': 0.033066132264529056, 'y': 0.36425339366515835}, {'x': 0.036072144288577156, 'y': 0.6334841628959276}, {'x': 0.9859719438877755, 'y': 0.6990950226244343}, {'x': 0.9849699398797596, 'y': 0.34841628959276016}], 'markedIdx': [357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703]}}, 'enabled': True, 'actions': {'gpioActions': [], 'msgActions': [], 'directActions': []}, 'measureCrossZones': False}

simultanous_alert = {"_id":"63f4ae32b76393d031bbb8f2","name":"Street hazard","alertType":0,"configuration":{"object":0,"offender":0,"objects":[{"object":0},{"object":1}],"detection":0,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},"tresholdTime":0},"settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":None,"enableNotificationSound":False,"alertThumbnail":True,"alertZoomThumbnail":True},"notifications":{"orgUsers":[],"manualUsers":[],"notificationMethods":{}},"timezone":"Asia/Jerusalem","actions":{"gpioActions":[],"msgActions":[],"directActions":[]},"zones":{"{\"x\":0.1312625250501002,\"y\":0.36199095022624433}":{"name":"","color":"green","selection":[{"x":0.1312625250501002,"y":0.36199095022624433},{"x":0.5921843687374749,"y":0.3755656108597285},{"x":0.5961923847695391,"y":0.40271493212669685},{"x":0.5991983967935872,"y":0.42081447963800905},{"x":0.10320641282565131,"y":0.4253393665158371}],"markedIdx":[388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434]}},"definedZones":True,"markedIdx":[356,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434],"measureCrossZones":False,"selectedCamera":{"locationId":"63627fbc28f6e6f856aa4884","cameraId":"63f4ad6bb76393d031bbb8f1","edgeId":"63e236d45951bba3f62f14c8","timezone":"Asia/Jerusalem"},"synced":True,"action":0,"orgId":"636279a628f6e6f856aa4881","enabled":True,"groupId":"8bb74144-a0e4-4416-8875-01a4643bea02","version":"0.1.861"}
temp_person = {'action': alertsActions.add.value, '_id': '636de0b623b1bb89af877658', 'alertType': 0, 'selectedCamera': {'locationId': '6314b194eb9c1420f9bf0c30', 'cameraId': '6314b1dbeb9c1420f9bf0c32', 'edgeId': '630eeca5ab67fdfefd359dcb',
  'timezone': 'Asia/Jerusalem'}, 'configuration': {'object': objectNames.person.value, 'detection': 0, 'detectionAdditionalAttributes': {'direction': 'above', 'count': 0}},
  'settings': {'schedule': {'monday': None, 'tuesday': None, 'wednesday': None, 'thursday': None, 'friday': None, 'saturday': None, 'sunday': None}, 'frequency': 0, 'additionalOptions': False, 'blockNotificationPeriod': None, 'enableNotificationSound': False,
  'alertThumbnail': True, 'alertZoomThumbnail': True},
  'markedIdx': [], 'lineCrossing': None, 'trafficControl': None,
  'actions': {'gpioActions': [], 'msgActions': [{'cloudHttp': '', 'localHttp': '', 'direct': {'protocol': 1, 'address': '192.168.86.60', 'port': '2222', 'msg': 'Lumix Message: Person in the yard', 'body': ''}}]}, 'enabled': True}


lubbokTest = {"_id":"63e1658684ff2e2945fb53c6","name":"Detector","alertType":0,"configuration":{"object":1,"detection":8,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},"filters":{},"tresholdTime":0},"settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},"frequency":0,"additionalOptions":False,"pushAlert":False,"blockNotificationPeriod":None,"enableNotificationSound":False,"alertThumbnail":False,"alertZoomThumbnail":True},"notifications":{"orgUsers":[],"manualUsers":[],"notificationMethods":{}},"timezone":"US/Central","actions":{"gpioActions":[],"msgActions":[],"directActions":[
    {"address":"192.168.86.110","port":"5555","protocol":1,"trafficController":True,"msg":"","body":""}]},
    "zones":{"{\"x\":0.4082901554404145,\"y\":0.2473444613050076}":{"name":"9","color":"green","selection":[{"x":0.4082901554404145,"y":0.2473444613050076},{"x":0.33264248704663213,"y":0.4218512898330804},{"x":0.49430051813471504,"y":0.4400606980273141},{"x":0.5139896373056995,"y":0.26251896813353565}],"markedIdx":[269,270,271,300,301,302,303,332,333,334,335,363,364,365,366,367,395,396,397,398,399,427,428,429,430,431]}},"definedZones":True,"markedIdx":[269,270,271,272,300,301,302,303,332,333,334,335,363,364,365,366,367,395,396,397,398,399,426,427,428,429,430,431],"measureCrossZones":False,"selectedCamera":{"locationId":"63869dfe608d754ab8932e9e","cameraId":"639917ab86fd40b13706dfaa","edgeId":"6351e2a10b88a213c0bb07da","timezone":"US/Central"},"synced":True,"action":0,"orgId":"63869da3608d754ab8932e9d","enabled":True,"groupId":"eedd0bd1-c032-4f7a-b312-9e3a4839322d","version":"0.1.50"}
gun_alert = {"_id":"GunAlert",
                    "action":   alertsActions.add.value,
                    "configuration": {
                         "object": objectNames.gun.value,
                         "detection":alertsType.appearance.value,
                         "detectionAdditionalAttributes": None,

                         #"filters":{
                                    #"genderType": ["male"],
                                    #"ageType": [],
                                    #"upperbodyColor": ["grey"],
                                    #"lowerbodyColor": ["blue"],
                         #}
                    },
                    'selectedCamera': {'locationId': '633994ad5c2e03efa5dce6ff', 'cameraId': '63468c0885203eb5c7a133d6', 'edgeId': '633998434b80396a8cccc9a7', 'timezone': 'Asia/Jerusalem'},
                    #"settings": {"schedule":{"monday":{"from":"01:00 AM","to":"04:30 AM"},"wednesday":{"from":"01:00 PM","to":"04:30 PM"},"tuesday":{"from":"01:00 AM","to":"04:30 AM"},"thursday":{"from":"09:00 AM","to":"10:30 AM"},"friday":{"from":"01:00 AM","to":"04:30 AM"},"saturday":{"from":"01:00 AM","to":"04:30 AM"},"sunday":{"from":"01:00 AM","to":"04:30 AM"}}}
                    "settings": {
                                'additionalOptions': False,
                                'blockNotificationPeriod': None,
                                'alertThumbnail': True,
                                'alertZoomThumbnail': True
                                }

}


motion_v2 = {"selectedFlow":{"category":0,"flowType":0,"formValue":{"sensitivity":80},"stepsLength":3,"currentStep":3},
"_id":"647c6bd0ab929700c7eac960",
"selectedCamera":{"locationId":"63b2a074f597c5e4a776b874","edgeId":"6445186c2bcc3fc3ff2ffe37",
"cameraId":"647c6781a3b47de785fc2439","zones":{"{\"x\":0.42950391644908614,\"y\":0.4375}":{"name":"","color":"green","selection":[{"x":0.42950391644908614,"y":0.4375},{"x":0.5313315926892951,"y":0.3888888888888889},{"x":0.577023498694517,"y":0.41898148148148145},{"x":0.5953002610966057,"y":0.49074074074074076},
{"x":0.6005221932114883,"y":0.6041666666666666},{"x":0.5613577023498695,"y":0.6574074074074074},{"x":0.45691906005221933,"y":0.6666666666666666},{"x":0.412532637075718,"y":0.5671296296296297}],
"markedIdx":[431,432,433,434,462,463,464,465,466,494,495,496,497,498,525,526,527,528,529,530,557,558,559,560,561,562,589,590,591,592,593,594,622,623,624,625,626,654,655,656,657]}},
"markedIdx":[431,432,433,434,462,463,464,465,466,493,494,495,496,497,498,525,526,527,528,529,530,557,558,559,560,561,562,589,590,591,592,593,594,622,623,624,625,626,654,655,656,657,686],"timezone":"Asia/Jerusalem"},
"settings":{"sound":True,"priority":2,"confidence":2,
"autoArchive":{"enabled":True,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":"0",
"formValue":{"notifications":{"orgUsers":[{"id":"63bd2a77d67d82346855d437","firstname":None,"lastname":None,"email":"qa@lumix.ai","phone":None,"roles":["owner","user"],"status":0,"timezone":None}],
"manualUsers":[],"notificationMethods":{"63bd2a77d67d82346855d437":"email"}}}}],"action":0,"version":"2.0.0"}


tampering_v2 = {"version": "2.0.0","_id":"646e0f69f8cb2f0e412624b8","enabled":True,"name":"Tampering outdoor","selectedFlow":{"category":0,"flowType":1,"formValue":{"location":"1","duration":10,
"camera":[{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646b6567bd9ed9b71a2ecfa8"}],
"schedule":""},"stepsLength":4,"currentStep":3},"actions":[{"actionType":None}],"settings":{"priority":2,"confidence":2,"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None}},
"selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646b6567bd9ed9b71a2ecfa8"},"synced":True,"action":0,"orgId":"63146239eb9c1420f9bf0bfd","groupId":"246eb178-162d-4c74-b8ba-118095f5369f","version":"2.0.0"}

udp_v2 = {"version": "2.0.0",'action': 0, 'selectedFlow': {'category': 0, 'flowType': 0, 'formValue': {'sensitivity': 58, 'camera': [{'locationId': '63146417eb9c1420f9bf0c01', 'edgeId': '643fb29b9b5bdc5b7ebac860', 'cameraId': '646c88f3c5375c0233448a47'}], 'schedule': [{'day': 0, 'allDay': True, 'from': '12:00 AM', 'to': '11:59 PM'}, {'day': 1, 'allDay': True, 'from': '12:00 AM', 'to': '11:59 PM'}, {'day': 2, 'allDay': True, 'from': '12:00 AM', 'to': '11:59 PM'}, {'day': 3, 'allDay': True, 'from': '12:00 AM', 'to': '11:59 PM'}, {'day': 4, 'allDay': True, 'from': '12:00 AM', 'to': '11:59 PM'}, {'day': 5, 'allDay': True, 'from': '12:00 AM', 'to': '11:59 PM'}, {'day': 6, 'allDay': True, 'from': '12:00 AM', 'to': '11:59 PM'}]}, 'stepsLength': 3, 'currentStep': 2}, '_id': '64731d5c7bae6a180a634806', 'selectedCamera': {'locationId': '63146417eb9c1420f9bf0c01', 'edgeId': '643fb29b9b5bdc5b7ebac860', 'cameraId': '646c88f3c5375c0233448a47'}, 'settings': {'priority': 2, 'confidence': 2, 'schedule': {'monday': {'from': '12:00 AM', 'to': '11:59 PM'}, 'tuesday': {'from': '12:00 AM', 'to': '11:59 PM'}, 'wednesday': {'from': '12:00 AM', 'to': '11:59 PM'}, 'thursday': {'from': '12:00 AM', 'to': '11:59 PM'}, 'friday': {'from': '12:00 AM', 'to': '11:59 PM'}, 'saturday': {'from': '12:00 AM', 'to': '11:59 PM'}, 'sunday': {'from': '12:00 AM', 'to': '11:59 PM'}}}, 'enabled': True, 'timezone': 'Asia/Jerusalem', 'actions': [{'actionType': '1', 'formValue': {'protocol': '2', 'method': '1', 'address': '192.168.101.66:2222', 'port': '', 'msg': '', 'body': ''}}]}

tailgatin_v2 = {"version": "2.0.0","_id":"64732be47bae6a180a63480a","enabled":True,"name":"tailgating","selectedFlow":{"category":0,"flowType":5,"formValue":{"objects":[{"type":0,"filters":{},"strict":True}],"duration":3,"schedule":[{"day":0,"allDay":True,"from":"12:00 AM","to":"11:59 PM"},{"day":1,"allDay":True,"from":"12:00 AM","to":"11:59 PM"},{"day":2,"allDay":True,"from":"12:00 AM","to":"11:59 PM"},{"day":3,"allDay":True,"from":"12:00 AM","to":"11:59 PM"},{"day":4,"allDay":True,"from":"12:00 AM","to":"11:59 PM"},{"day":5,"allDay":True,"from":"12:00 AM","to":"11:59 PM"},{"day":6,"allDay":True,"from":"12:00 AM","to":"11:59 PM"}]},"stepsLength":3,"currentStep":2},"actions":[{"actionType":None}],"settings":{"priority":2,"confidence":2,"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None}},"selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646cc8afe2947aa75fea1f9d",'lineCrossing': {'p1': {'x': 0.73, 'y': 0.54}, 'p2': {'x': 0.21, 'y': 0.76}, 'd': 1}},"synced":True,"action":0,"orgId":"63146239eb9c1420f9bf0bfd","groupId":"7eb31346-7efa-402e-b1ef-b5dbc32cda95","version":"2.0.0"}

weapon_v2 = {"version": "2.0.0","_id":"64736acb60f600ec270ceed1","enabled":True,"name":"weapon","selectedFlow":{"category":0,"flowType":3,"formValue":{"camera":[{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646a0e9c7b107727068a8313"}],"schedule":""},"stepsLength":2,"currentStep":1},"actions":[{"actionType":None}],"settings":{"reactivationTh":100,"sound":True,"priority":2,"confidence":2,"autoAck":10,"reactivationTh":10,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646a0e9c7b107727068a8313"},"synced":True,"action":0,"orgId":"63146239eb9c1420f9bf0bfd","groupId":"064f832b-b3ab-4d7a-85e6-bf1a0ab929c6","version":"2.0.0"}

proximity_v4 = {'action': 0, 'selectedFlow': {'category': 0, 'flowType': 2, 'formValue': {'proximity': 1, 'objects': [{'type': 0, 'filters': {}}, {'type': 1, 'filters': {'type': ['motorcycle']}}], 'duration': 0, 'schedule': 0}, 'stepsLength': 5, 'currentStep': 3}, '_id': '6474ab3521e6f2d1cdaac850', 'selectedCamera': {'locationId': '63146417eb9c1420f9bf0c01', 'edgeId': '643fb29b9b5bdc5b7ebac860', 'cameraId': '646b5f5be8c942202a36bf1d', 'timezone': 'Asia/Jerusalem'}, 'settings': {'sound': True, 'priority': 2, 'confidence': 2, 'autoArchive': {'enabled': False, 'duration': 10}, 'reactivationTh': 0, 'display': 2, 'titleColor': 0, 'picInPic': 0, 'picInPicPos': 1, 'schedule': None}, 'enabled': True, 'timezone': 'Asia/Jerusalem', 'actions': [{'actionType': None}], 'version': '2.0.0'}


zone_protection = {"selectedFlow":{"category":0,"flowType":6,"formValue":{"objects":[{"type":0,"filters":{},"strict":True}],"schedule":0},"stepsLength":3,"currentStep":2},"_id":"6475a825dce78b0b6dce05d8","selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646a0e9c7b107727068a8313","zones":{"{\"x\":0.5013054830287206,\"y\":0.5578703703703703}":{"name":"","color":"green","selection":[{"x":0.5013054830287206,"y":0.5578703703703703},{"x":0.7258485639686684,"y":0.6064814814814815},{"x":0.7258485639686684,"y":0.8842592592592593},{"x":0.36945169712793735,"y":0.8703703703703703}],"markedIdx":[592,593,594,623,624,625,626,627,628,629,630,655,656,657,658,659,660,661,662,686,687,688,689,690,691,692,693,694,718,719,720,721,722,723,724,725,726,750,751,752,753,754,755,756,757,758,781,782,783,784,785,786,787,788,789,790,813,814,815,816,817,818,819,820,821,822,844,845,846,847,848,849,850,851,852,853,854,876,877,878,879,880,881,882,883,884,885,886]},"{\"x\":0.5391644908616188,\"y\":0.4351851851851852}":{"name":"","color":"blue","selection":[{"x":0.5391644908616188,"y":0.4351851851851852},{"x":0.7180156657963447,"y":0.4675925925925926},{"x":0.7232375979112271,"y":0.6018518518518519},{"x":0.49869451697127937,"y":0.5509259259259259}],"markedIdx":[465,466,467,468,497,498,499,500,501,502,528,529,530,531,532,533,534,560,561,562,563,564,565,566,596,597,598]}},"markedIdx":[465,466,467,468,497,498,499,500,501,502,528,529,530,531,532,533,534,560,561,562,563,564,565,566,592,593,594,595,596,597,598,623,624,625,626,627,628,629,630,655,656,657,658,659,660,661,662,686,687,688,689,690,691,692,693,694,718,719,720,721,722,723,724,725,726,750,751,752,753,754,755,756,757,758,781,782,783,784,785,786,787,788,789,790,813,814,815,816,817,818,819,820,821,822,844,845,846,847,848,849,850,851,852,853,854,876,877,878,879,880,881,882,883,884,885,886],"timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}

loitering_v2 = {"selectedFlow":{"category":2,"flowType":2,"formValue":{"duration":6,"objects":[{"type":0,"filters":{"genderType": ["male"]},"strict":True}],"schedule":0},"stepsLength":4,"currentStep":3},"_id":"6475b63b1eccf6233687ebc7","selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646a0e9c7b107727068a8313","timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}

qa_alert_1 = {"selectedFlow":{"category":2,"flowType":5,"formValue":{"objects":[{"type":1,"filters":{"colors":["white"]},"strict":True}]},"stepsLength":3,"currentStep":3},"_id":"6458e65c22fce9a9ebb4075d","selectedCamera":{"locationId":"63b2a074f597c5e4a776b874","cameraId":"6458defb22fce9a9ebb40746","edgeId":"6445186c2bcc3fc3ff2ffe37","timezone":"Asia/Jerusalem","lineCrossing":None,"trafficControl":{"state":0,"lines":[{"p1":{"x":0.74,"y":0.33},"p2":{"x":0.88,"y":0.31},"d":0,"color":"blue"},{"p1":{"x":0.94,"y":0.42},"p2":{"x":0.85,"y":0.55},"d":0,"color":"green"}],"distance":4,"distanceUnits":"meter"},"markedIdx":None,"zones":None},"settings":{"sound":True,"autoArchive":None,"priority":2,"confidence":2,"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}

lpr_v2 = {"selectedFlow":{"category":1,"flowType":1,"formValue":{"plates":{"list":[{"name":"test","plate":"1037056"}],"appears":True,"unrecognized":True},"schedule":0},"stepsLength":3,"currentStep":1},"_id":"6475dce2008359f10550f5ca","selectedCamera":{
    'zones': {'{"x":0.006012024048096192,"y":0.9946808510638298}': {'name': '', 'color': 'green', 'selection': [{'x': 0.006012024048096192, 'y': 0.9946808510638298}, {'x': 0.5150300601202404, 'y': 0.9840425531914894}, {'x': 0.4969939879759519, 'y': 0.49645390070921985}, {'x': 0.004008016032064128, 'y': 0.46631205673758863}], 'markedIdx': [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]}}, 'markedIdx': [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
    "locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646a0e9c7b107727068a8313","timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}

line_crossing_v2 = {"selectedFlow":{"category":2,"flowType":3,"formValue":{"objects":[{"type":0,"filters":{},"strict":None}],"duration":0,"schedule":0},"stepsLength":3,"currentStep":2},"_id":"64760a4f9edb2cc98f4eab2b","selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646a0ecb7b107727068a8315","lineCrossing":{"p1":{"x":0.28,"y":0.7},"p2":{"x":0.57,"y":0.69},"d":2,"state":2},"timezone":"Asia/Jerusalem"},"settings":{"sound":None,"priority":2,"confidence":2,"autoArchive":{"enabled":None,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":None,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}

lpr_maor = {'action': 0, '_id': '6458e62a22fce9a9ebb4075c', 'alertType': 0, 'selectedCamera': {'locationId': '63b2a074f597c5e4a776b874', 'cameraId': '6458defb22fce9a9ebb40746', 'edgeId': '6445186c2bcc3fc3ff2ffe37', 'timezone': 'Asia/Jerusalem'},
'configuration': {'object': 1, 'detection': 4, 'detectionAdditionalAttributes': {'direction': 'above', 'count': 0, 'sensitivity': 0.5}, 'filters': {'greenList': '', 'redList': '1037056', 'unrecognized': False}, 'tresholdTime': 0},
'settings': {'schedule': {'monday': None, 'tuesday': None, 'wednesday': None, 'thursday': None, 'friday': None, 'saturday': None, 'sunday': None}, 'frequency': 0, 'additionalOptions': False, 'pushAlert': True, 'blockNotificationPeriod': None, 'enableNotificationSound': True,
'alertThumbnail': True, 'alertZoomThumbnail': True}, 'enabled': True, 'actions': {'gpioActions': [], 'msgActions': [], 'directActions': []}, 'measureCrossZones': False,
'zones': {'{"x":0.006012024048096192,"y":0.9946808510638298}': {'name': '', 'color': 'green', 'selection': [{'x': 0.006012024048096192, 'y': 0.9946808510638298}, {'x': 0.5150300601202404, 'y': 0.9840425531914894}, {'x': 0.4969939879759519, 'y': 0.49645390070921985}, {'x': 0.004008016032064128, 'y': 0.46631205673758863}], 'markedIdx': [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]}}, 'markedIdx': [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]}

lpr_v3 = {"selectedFlow":{"category":1,"flowType":1,"formValue":{"plates":{"list":[{"name":"","plate":"1037056"}],"appears":True,"unrecognized":False}},"stepsLength":3,"currentStep":3},"_id":"64785fd258fb0358309ad92e","selectedCamera":{"locationId":"63b2a074f597c5e4a776b874","edgeId":"6445186c2bcc3fc3ff2ffe37","cameraId":"6458defb22fce9a9ebb40746","zones":{"{\"x\":0.0026109660574412533,\"y\":0.49074074074074076}":{"name":"","color":"green","selection":[{"x":0.0026109660574412533,"y":0.49074074074074076},{"x":0.0026109660574412533,"y":0.9884259259259259},{"x":0.9947780678851175,"y":0.9907407407407407},{"x":0.9882506527415144,"y":0.49074074074074076}],"markedIdx":[512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976,977,978,979,980,981,982,983,984,985,986,987,988,989,990,991,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023]}},"markedIdx":[511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976,977,978,979,980,981,982,983,984,985,986,987,988,989,990,991,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023],"timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}

traffic_v2 = {"selectedFlow":{"category":2,"flowType":5,"formValue":{"objects":[{"type":0,"filters":{},"strict":True}],"duration":0,"schedule":0},"stepsLength":3,"currentStep":1},"_id":"64760be09edb2cc98f4eab2c","selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646a0e9c7b107727068a8313","trafficControl":{"lines":[{"p1":{"x":0.56,"y":0.31},"p2":{"x":0.75,"y":0.35},"d":1,"color":"blue"},{"p1":{"x":0.48,"y":0.6},"p2":{"x":0.75,"y":0.65},"d":1,"color":"green"}],"distance":None,"distanceUnits":"meter","state":0},"timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}
traffic_v2 = {"selectedFlow":{"category":2,"flowType":5,"formValue":{"objects":[{"type":0,"filters":{},"strict":True}],"duration":0},"stepsLength":3,"currentStep":2},"_id":"647614529edb2cc98f4eab2f","selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646a0e9c7b107727068a8313","trafficControl":{"lines":[{"p1":{"x":0.49,"y":0.59},"p2":{"x":0.75,"y":0.68},"d":1,"color":"green"},{"p1":{"x":0.51,"y":0.51},"p2":{"x":0.74,"y":0.58},"d":1,"color":"blue"}],"distance":None,"distanceUnits":"meter","state":0},"timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}
traffic_v3 = {"selectedFlow":{"category":2,"flowType":5,"formValue":{"objects":[{"type":0,"filters":{},"strict":True}],"duration":0},"stepsLength":3,"currentStep":2},"_id":"64765cbf9edb2cc98f4eab30","selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646a0ecb7b107727068a8315","trafficControl":{"lines":[{"p1":{"x":0.3,"y":0.65},"p2":{"x":0.54,"y":0.61},"d":2,"color":"blue"},{"p1":{"x":0.28,"y":0.79},"p2":{"x":0.58,"y":0.75},"d":2,"color":"green"}],"distance":None,"distanceUnits":"meter","state":0},"timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}
speed_v2 = {"selectedFlow":{"category":0,"flowType":7,"formValue":{"objects":[{"type":0,"filters":{},"strict":True}],"duration":4},"stepsLength":3,"currentStep":2},"_id":"64765cbf9edb2cc98f4eab30","selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646a0ecb7b107727068a8315","trafficControl":{"lines":[{"p1":{"x":0.3,"y":0.65},"p2":{"x":0.54,"y":0.61},"d":2,"color":"blue"},{"p1":{"x":0.28,"y":0.79},"p2":{"x":0.58,"y":0.75},"d":2,"color":"green"}],"distance":None,"distanceUnits":"meter","state":0},"timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}

occupancy_2     = {"selectedFlow":{"category":2,"flowType":4,"formValue":{"amount":4,"objects":[{"type":0,"filters":{},"strict":True}]},"stepsLength":4,"currentStep":4},                                                   "_id":"6478a3a758fb0358309ad93a","selectedCamera":{"locationId":"63b2a074f597c5e4a776b874","edgeId":"6445186c2bcc3fc3ff2ffe37","cameraId":"6458defb22fce9a9ebb40746","zones":{"{\"x\":0.5026109660574413,\"y\":0.9953703703703703}":{"name":"","color":"green","selection":[{"x":0.5026109660574413,"y":0.9953703703703703},{"x":0.4960835509138381,"y":0.011574074074074073},{"x":0.006527415143603133,"y":0.016203703703703703},{"x":0.01174934725848564,"y":0.9907407407407407}],
"markedIdx":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007]}},
"markedIdx":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007],
"timezone":"Asia/Jerusalem"},
"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}
mixed_occupancy = {"selectedFlow":{"category":2,"flowType":4,"formValue":{"amount":1,"objects":[{"type":0,"filters":{},"strict":True},{"type":2,"filters":{},"strict":True}],"schedule":0},"stepsLength":4,"currentStep":2},"_id":"64759b81dce78b0b6dce05d5","selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"643fb29b9b5bdc5b7ebac860","cameraId":"646a0ecb7b107727068a8315","timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}
proximity = {"selectedFlow":{"category":0,"flowType":2,"formValue":{"proximity":1,"objects":[{"type":0,"filters":{},"strict":True},{"type":1,"filters":{},"strict":True}],"duration":30},"stepsLength":5,"currentStep":5},"_id":"64c09b5c17fd0609de3d13f1","selectedCamera":{"locationId":"63582c5af5dee0ccdd0c7a68","edgeId":"645d27121f67833a07945296","cameraId":"64640bdf462aa13c0f1787a4","zones":{"{\"x\":0.6452464788732394,\"y\":0.315625}":{"name":"","color":"green","selection":[{"x":0.6452464788732394,"y":0.315625},{"x":0.9621478873239436,"y":0.465625},{"x":0.9498239436619719,"y":0.9859375},{"x":0.511443661971831,"y":0.8265625}],"markedIdx":[341,372,373,374,375,404,405,406,407,408,409,436,437,438,439,440,441,442,443,467,468,469,470,471,472,473,474,475,476,477,499,500,501,502,503,504,505,506,507,508,509,510,531,532,533,534,535,536,537,538,539,540,541,542,563,564,565,566,567,568,569,570,571,572,573,574,594,595,596,597,598,599,600,601,602,603,604,605,606,626,627,628,629,630,631,632,633,634,635,636,637,638,658,659,660,661,662,663,664,665,666,667,668,669,670,690,691,692,693,694,695,696,697,698,699,700,701,702,721,722,723,724,725,726,727,728,729,730,731,732,733,734,753,754,755,756,757,758,759,760,761,762,763,764,765,766,785,786,787,788,789,790,791,792,793,794,795,796,797,798,817,818,819,820,821,822,823,824,825,826,827,828,829,830,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,883,884,885,886,887,888,889,890,891,892,893,894,918,919,920,921,922,923,924,925,953,954,955,956,957,987,988,989]}},"markedIdx":[340,341,372,373,374,375,404,405,406,407,408,409,436,437,438,439,440,441,442,443,467,468,469,470,471,472,473,474,475,476,477,499,500,501,502,503,504,505,506,507,508,509,510,531,532,533,534,535,536,537,538,539,540,541,542,563,564,565,566,567,568,569,570,571,572,573,574,594,595,596,597,598,599,600,601,602,603,604,605,606,626,627,628,629,630,631,632,633,634,635,636,637,638,658,659,660,661,662,663,664,665,666,667,668,669,670,690,691,692,693,694,695,696,697,698,699,700,701,702,721,722,723,724,725,726,727,728,729,730,731,732,733,734,753,754,755,756,757,758,759,760,761,762,763,764,765,766,785,786,787,788,789,790,791,792,793,794,795,796,797,798,817,818,819,820,821,822,823,824,825,826,827,828,829,830,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,883,884,885,886,887,888,889,890,891,892,893,894,918,919,920,921,922,923,924,925,926,953,954,955,956,957,987,988,989,1022],"timezone":"PST8PDT"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":1,"titleColor":"#0000FF","picInPic":0,"picInPicPos":2},"enabled":True,"timezone":"PST8PDT","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}

family_v2 = {"selectedFlow":{"category":2,"flowType":0,"formValue":{"objects":[{"type":1,"filters":{"model":["G","G 63 AMG","Range Rover","Range Rover (L322)","Range Rover (L405)","Range Rover (P38A)","Range Rover Classic"],"make":["Land Rover","Mercedes-AMG","Mercedes-Benz"]},"strict":True}]},"stepsLength":3,"currentStep":3},"_id":"6441d081a26739c780ae177c","selectedCamera":{"locationId":"64417234a26739c780ae1762","cameraId":"644192c2a26739c780ae1766","edgeId":"6438ad15599deba77930bb53","timezone":"America/New_York","lineCrossing":None,"trafficControl":None,"markedIdx":[],"zones":{}},"settings":{"sound":True,"autoArchive":{"enabled":True,"duration":30},"priority":2,"confidence":2,"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"America/New_York","actions":[{"actionType":"0","formValue":{"notifications":{"orgUsers":[{"id":"6441bc5aa26739c780ae1778","firstname":"Ariel","lastname":"Shlezinger","email":"ariel.shlezinger@fon-llc.com","phone":None,"roles":["admin"],"status":0}],"manualUsers":[],"notificationMethods":{"6441bc5aa26739c780ae1778":"email"}}}}],"action":0,"version":"2.0.0"}


lpr_v3 = {"selectedFlow":{"category":1,"flowType":1,"formValue":{"plates":{"list":[{"name":"","plate":" "}],"appears":False,"unrecognized":True}},"stepsLength":3,"currentStep":3},"_id":"647c9ba6dbdaffd9873a53b5","selectedCamera":{"locationId":"63582c5af5dee0ccdd0c7a68","edgeId":"645d26cb1f67833a07945295","cameraId":"6464064c462aa13c0f1787a0","timezone":"PST8PDT"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":2,"titleColor":0,"picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"PST8PDT","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}

motion_test = {"selectedFlow":{"category":0,"flowType":0,"formValue":{"sensitivity":50},"stepsLength":3,"currentStep":3},"_id":"649020f0c18f603a238114bb","selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"62f4e3ddfd167d8aec2008c8","cameraId":"648ff2d45305b5786438006e","timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False},"reactivationTh":0,"display":1,"titleColor":"#fff700","picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}
motion_test = {"selectedFlow":{"category":0,"flowType":0,"formValue":{"sensitivity":30},"stepsLength":3,"currentStep":3},"_id":"6491d767e1cdedc4aaaa461a","selectedCamera":{"locationId":"648e142b959ddaaa029d4efa","edgeId":"643753f2acc3bcf72f1d089a","cameraId":"648e146b959ddaaa029d4efc","zones":{"{\"x\":0.18330733229329174,\"y\":0.1683991683991684}":{"name":"","color":"green","selection":[{"x":0.18330733229329174,"y":0.1683991683991684},{"x":0.14508580343213728,"y":0.2681912681912682},{"x":0.17160686427457097,"y":0.39293139293139295},{"x":0.17706708268330734,"y":0.5582120582120582},{"x":0.2059282371294852,"y":0.7245322245322245},{"x":0.16926677067082682,"y":0.8128898128898129},{"x":0.09126365054602184,"y":0.8596673596673596},{"x":0.1294851794071763,"y":0.9459459459459459},{"x":0.25663026521060844,"y":0.9844074844074844},{"x":0.4282371294851794,"y":0.9823284823284824},{"x":0.4797191887675507,"y":0.9158004158004158},{"x":0.517160686427457,"y":0.8232848232848233},{"x":0.5663026521060842,"y":0.7692307692307693},{"x":0.6224648985959438,"y":0.6891891891891891},{"x":0.6950078003120125,"y":0.6278586278586279},{"x":0.6809672386895476,"y":0.5187110187110187},{"x":0.6318252730109204,"y":0.5239085239085239},{"x":0.5280811232449298,"y":0.58004158004158},{"x":0.47581903276131043,"y":0.5945945945945946},{"x":0.4391575663026521,"y":0.5488565488565489},{"x":0.39391575663026523,"y":0.5187110187110187},{"x":0.38845553822152884,"y":0.5966735966735967},{"x":0.3205928237129485,"y":0.5457380457380457},{"x":0.29797191887675506,"y":0.4896049896049896},{"x":0.26521060842433697,"y":0.4282744282744283},{"x":0.24882995319812792,"y":0.3804573804573805},{"x":0.20436817472698907,"y":0.38253638253638256},{"x":0.19578783151326054,"y":0.38253638253638256},{"x":0.18252730109204368,"y":0.3180873180873181},{"x":0.1684867394695788,"y":0.25467775467775466},{"x":0.1809672386895476,"y":0.20893970893970895},{"x":0.20826833073322934,"y":0.183991683991684}],"markedIdx":[197,229,261,293,325,357,389,390,391,422,423,454,455,456,486,487,488,518,519,520,521,550,551,552,553,557,563,564,565,582,583,584,585,586,587,588,589,590,593,594,595,596,597,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,710,711,712,713,714,715,716,717,718,719,720,721,722,723,742,743,744,745,746,747,748,749,750,751,752,753,754,774,775,776,777,778,779,780,781,782,783,784,785,806,807,808,809,810,811,812,813,814,815,816,837,838,839,840,841,842,843,844,845,846,847,848,867,868,869,870,871,872,873,874,875,876,877,878,879,899,900,901,902,903,904,905,906,907,908,909,910,911,932,933,934,935,936,937,938,939,940,941,942,965,966,967,968,969,970,971,972,973,1000,1001,1002]}},"markedIdx":[166,197,198,229,260,261,293,325,357,358,389,390,391,421,422,423,424,453,454,455,456,485,486,487,488,489,517,518,519,520,521,524,549,550,551,552,553,554,556,557,562,563,564,565,581,582,583,584,585,586,587,588,589,590,592,593,594,595,596,597,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,710,711,712,713,714,715,716,717,718,719,720,721,722,723,742,743,744,745,746,747,748,749,750,751,752,753,754,774,775,776,777,778,779,780,781,782,783,784,785,805,806,807,808,809,810,811,812,813,814,815,816,817,835,836,837,838,839,840,841,842,843,844,845,846,847,848,867,868,869,870,871,872,873,874,875,876,877,878,879,899,900,901,902,903,904,905,906,907,908,909,910,911,931,932,933,934,935,936,937,938,939,940,941,942,943,964,965,966,967,968,969,970,971,972,973,974,999,1000,1001,1002,1003,1004,1005],"timezone":"America/Los_Angeles"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":True,"duration":20},"reactivationTh":0,"display":1,"titleColor":"#0000FF","picInPic":0,"picInPicPos":2},"enabled":True,"timezone":"America/Los_Angeles","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}
motion_test = {"selectedFlow":{"category":0,"flowType":0,"formValue":{"sensitivity":4,"schedule":[{"day":0,"enabled":True,"times":[{"from":"09:10 PM","to":"09:12 PM"}]},{"day":1,"enabled":True,"times":[{"from":"09:10 PM","to":"09:12 PM"}]},{"day":2,"enabled":True,"times":[{"from":"04:32 PM","to":"09:12 PM"}]},{"day":3,"enabled":True,"times":[{"from":"12:00 AM","to":"12:00 AM"}]},{"day":4,"enabled":True,"times":[{"from":"12:00 AM","to":"12:00 AM"}]},{"day":5,"enabled":True,"times":[{"from":"12:00 AM","to":"12:00 AM"}]},{"day":6,"enabled":True,"times":[{"from":"12:00 AM","to":"12:00 AM"}]}]},"stepsLength":3,"currentStep":3},"_id":"64c00fcf42ba0a7ee1a9590e","selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","edgeId":"64bd31a5f377479bcdc008af","cameraId":"64bfb8f607b61f3cba4ab148","timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":1,"titleColor":"#0000FF","picInPic":0,"picInPicPos":2},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}
testtset    = {"selectedFlow":{"category":0,"flowType":0,"formValue":{"sensitivity":1,"schedule":[{"day":0,"allDay":False,"from":"01:00 AM","to":"07:00 AM"},{"day":1,"allDay":False,"from":"01:00 AM","to":"07:00 AM"},{"day":2,"allDay":False,"from":"01:00 AM","to":"07:00 AM"},{"day":3,"allDay":False,"from":"01:00 AM","to":"07:00 AM"},{"day":4,"allDay":False,"from":"01:00 AM","to":"07:00 AM"},{"day":5,"allDay":False,"from":"01:00 AM","to":"07:00 AM"},{"day":6,"allDay":False,"from":"01:00 AM","to":"07:00 AM"}]},"stepsLength":4,"currentStep":4},"_id":"64b957730f623d5c85d4705e","selectedCamera":{"locationId":"6499d4db29afa322e8259090","edgeId":"645d24a71f67833a07945294","cameraId":"6499d58929afa322e8259092","timezone":"America/New_York"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":1,"titleColor":"#0000FF","picInPic":0,"picInPicPos":2},"enabled":True,"timezone":"America/New_York","actions":[{"actionType":"0","formValue":{"notifications":{"orgUsers":[{"id":"649c87a229afa322e82590bf","firstname":"Tom","lastname":"Bragg","email":"tbragg@hyde.edu","phone":None,"roles":["admin"],"status":0},{"id":"64a56a8d66a4ee0d667b0451","firstname":"Richard","lastname":"Truluck","email":"rtruluck@hyde.edu","phone":None,"roles":["admin"],"status":0}],"manualUsers":[],"notificationMethods":{"649c87a229afa322e82590bf":"email","64a56a8d66a4ee0d667b0451":"email"}}}}],"action":0,"version":"2.0.0"}
testtset    = {"selectedFlow":{"category":0,"flowType":0,"formValue":{"sensitivity":1,"schedule":0},"stepsLength":4,"currentStep":4},"_id":"64b957730f623d5c85d4705e","selectedCamera":{"locationId":"6499d4db29afa322e8259090","edgeId":"645d24a71f67833a07945294","cameraId":"6499d58929afa322e8259092","timezone":"America/New_York"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":1,"titleColor":"#0000FF","picInPic":0,"picInPicPos":2},"enabled":True,"timezone":"America/New_York","actions":[{"actionType":"0","formValue":{"notifications":{"orgUsers":[{"id":"649c87a229afa322e82590bf","firstname":"Tom","lastname":"Bragg","email":"tbragg@hyde.edu","phone":None,"roles":["admin"],"status":0},{"id":"64a56a8d66a4ee0d667b0451","firstname":"Richard","lastname":"Truluck","email":"rtruluck@hyde.edu","phone":None,"roles":["admin"],"status":0}],"manualUsers":[],"notificationMethods":{"649c87a229afa322e82590bf":"email","64a56a8d66a4ee0d667b0451":"email"}}}}],"action":0,"version":"2.0.0"}
ofir_test = {"selectedFlow":{"category":1,"flowType":0,"formValue":{"people":{"list":[{"id":3529710967,"name":"ofir","zoomImageHex":"64e0953d9432063e02eae23c3f60053eb7356cbcec098f3cd395aabd475891befedc39bd506c45bec6a1803db9a100bdceab9a3d2feb773d1481b23beefdacbc098483be7e39ca3dab7faebb13b6cdbde385c83ca7710dbc4efe19be6fd5bebca246c9bdadecce3db8e48abca25030bedc3a05bdc0845fbc2ef689bbc50717be79ad1abca73682bd700b5fbd3f735e3c0b77ae3cfae11c3de742d1bd899b003d64a573bd1d41cbbdc837d03c74019c3d4083ff3d180303be149d9c3cfee4393efd2fdebb34cbce3cc55a983d3ce4b13d557b95bd16916cbdb6cee63c14c53abdf261aabdb03ad93b3453e2bcf06c7f3deb3897bd8d4391bd82d435bc27c464bcffcf77be860b873c4fa1d3bc3aab84bd852a583e9d3afebc3267063e1926a53d1aa1a23cbe5804bd74b3a0bd40738ebd3c1fb13c319d85bc095cbabdcdd09abcd456be3de9fabc3d67940fbdf2b5af3d01ef29bd8839ebbce2b8433c1289babde0801dbeb9f9da3d8c53a33cd109043d3436febd56dfb9bc3f8a693d06d12fbcc2eb773d1333d73d7835a7bb8085b4bd5355603db2d6f9bcd9fb92bd1dfa003d4b3335bd420246bee53db2bb5be823bd5686173d85198e3b86fed23d6312b6bd03b1eb3b495f023ddb50073e290fd8bd5b87e63d3475533d6631183ed289a13dcb49a73b5e4a0fbd5a6bee3da524df3da7c9d0bc5c374abb035c8b3d3993413e"}],"appears":True,"unrecognized":False}},"stepsLength":3,"currentStep":3},"_id":"64bd0dae72fc14bf68697910","selectedCamera":{"locationId":"63b2a074f597c5e4a776b874","edgeId":"64b65887192619a71ac85039","cameraId":"64b65b1f192619a71ac8503d","timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":1,"titleColor":"#0000FF","picInPic":0,"picInPicPos":2},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}
ofir_test = {"selectedFlow":{"category":0,"flowType":4,"formValue":{"objects":[{"type":0,"filters":{},"strict":True}]},"stepsLength":3,"currentStep":3},"_id":"64c778f039cb7015c825a076","selectedCamera":{"locationId":"643e3760589eec72287dfab4","edgeId":"64c69f228d895b9afa168109","cameraId":"64c6d1838d895b9afa16810e","lineCrossing":{"p1":{"x":0.15,"y":0.51},"p2":{"x":0.75,"y":0.44},"d":0,"state":2},"timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":1,"titleColor":"#0000FF","picInPic":0,"picInPicPos":2},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{'actionType': '1', 'formValue': {'method': 1, 'protocol': 0, 'address': '192.168.100.150', 'msg': 0, 'body': 0}},{"actionType":None}],"action":0,"version":"2.0.0"}
ofir_test = {"selectedFlow":{"category":0,"flowType":5,"formValue":{"objects":[{"type":0,"filters":{},"strict":True}],"duration":6},"stepsLength":4,"currentStep":4},"_id":"647c91ff411573d72b11b953","selectedCamera":{"locationId":"63b2a074f597c5e4a776b874","edgeId":"6445186c2bcc3fc3ff2ffe37","cameraId":"6458defb22fce9a9ebb40746","lineCrossing":{"p1":{"x":0.24,"y":0.35},"p2":{"x":0.42,"y":0.32},"d":0,"state":2},"timezone":"Asia/Jerusalem"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False},"reactivationTh":0,"display":1,"titleColor":"#948489","picInPic":0,"picInPicPos":1},"enabled":True,"timezone":"Asia/Jerusalem","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}
first_test = {"selectedFlow":{"category":2,"flowType":2,"formValue":{"duration":60,"durationUnit":0,"objects":[{"type":0,"filters":{},"strict":True}]},"stepsLength":5,"currentStep":5},"_id":"64dfad9e90e220d17209f7df","selectedCamera":{"locationId":"641878adf76a57af5b2cb973","edgeId":"63e1ac2384ff2e2945fb53cd","cameraId":"641ccab6f76a57af5b2cb9b1","zones":{"{\"x\":0.0684931506849315,\"y\":0.027777777777777776}":{"name":"","color":"green","selection":[{"x":0.0684931506849315,"y":0.027777777777777776},{"x":0.0136986301369863,"y":0.3038194444444444},{"x":0.9608610567514677,"y":0.3489583333333333},{"x":0.9765166340508806,"y":0.1909722222222222}],"markedIdx":[34,35,36,37,66,67,68,69,70,71,72,73,74,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350]}},"markedIdx":[2,3,4,34,35,36,37,38,39,40,41,65,66,67,68,69,70,71,72,73,74,75,76,77,78,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,191,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,380,381,382],"timezone":"America/New_York"},"settings":{"sound":True,"priority":2,"confidence":2,"autoArchive":{"enabled":False,"duration":10},"reactivationTh":0,"display":1,"titleColor":"#0000FF","picInPic":0,"picInPicPos":2},"enabled":True,"timezone":"America/New_York","actions":[{"actionType":None}],"action":0,"version":"2.0.0"}
schedule_t = {'action': 0, 'selectedFlow': {'category': 0, 'flowType': 0, 'formValue': {'sensitivity': 23, 'schedule': [{'day': 0, 'enabled': True, 'times': [{'from': '12:00 AM', 'to': '12:00 AM'}]}, {'day': 1, 'enabled': True, 'times': [{'from': '12:00 AM', 'to': '12:00 AM'}]}, {'day': 2, 'enabled': True, 'times': [{'from': '12:00 AM', 'to': '12:00 AM'}]}, {'day': 3, 'enabled': True, 'times': []}, {'day': 4, 'enabled': True, 'times': [{'from': '12:00 AM', 'to': '12:00 AM'}]}, {'day': 5, 'enabled': True, 'times': [{'from': '12:00 AM', 'to': '12:00 AM'}]}, {'day': 6, 'enabled': True, 'times': [{'from': '12:00 AM', 'to': '12:00 AM'}]}]}, 'stepsLength': 3, 'currentStep': 3}, '_id': '64ef8d4b63268407a668f61f', 'selectedCameras': [{'locationId': '63146417eb9c1420f9bf0c01', 'edgeId': '64eefe8c062c9aaa00bd64ed', 'cameraId': '64ef0fc05cc73624ae406805'}, {'locationId': '63146417eb9c1420f9bf0c01', 'edgeId': '64eefe8c062c9aaa00bd64ed', 'cameraId': '64ef0fc05cc73624ae406808'}], 'settings': {'sound': True, 'priority': 2, 'confidence': 2, 'autoArchive': {'enabled': False, 'duration': 10}, 'reactivationTh': 0, 'display': 1, 'titleColor': '#0000FF', 'picInPic': 0, 'picInPicPos': 2, 'schedule': [{'day': 0, 'enabled': True, 'times': [{'from': '12:00 AM', 'to': '12:00 AM'}]}, {'day': 1, 'enabled': True, 'times': [{'from': '12:00 AM', 'to': '12:00 AM'}]}, {'day': 2, 'enabled': False, 'times': []}, {'day': 3, 'enabled': True, 'times': []}, {'day': 4, 'enabled': True, 'times': [{'from': '12:00 AM', 'to': '12:00 AM'}]}, {'day': 5, 'enabled': True, 'times': [{'from': '12:00 AM', 'to': '12:00 AM'}]}, {'day': 6, 'enabled': True, 'times': [{'from': '12:00 AM', 'to': '12:00 AM'}]}]}, 'enabled': True, 'timezone': 'Asia/Jerusalem', 'actions': [{'actionType': None}], 'version': '2.0.0', 'selectedCamera': {'locationId': '63146417eb9c1420f9bf0c01', 'edgeId': '64eefe8c062c9aaa00bd64ed', 'cameraId': '64ef0fc05cc73624ae406808'}}
protective = {'action': 0, '_id': '64fec608dbd4dd81a8bf6aad', 'selectedFlow': {'category': 4, 'flowType': 0, 'formValue': {'wear': 0, 'duration': 1, 'durationUnit': 0}, 'stepsLength': 5, 'currentStep': 5}, 'selectedCameras': [{'locationId': '63146417eb9c1420f9bf0c01', 'edgeId': '64eefe8c062c9aaa00bd64ed', 'cameraId': '64fdc143ef834f99ea6bd9d1'}], 'settings': {'sound': True, 'priority': 2, 'confidence': 2, 'autoArchive': {'enabled': False, 'duration': 10}, 'reactivationTh': 0, 'display': 1, 'titleColor': '#0000FF', 'picInPic': 0, 'picInPicPos': 2, 'schedule': None}, 'enabled': True, 'timezone': 'Asia/Jerusalem', 'actions': [{'actionType': None}], 'version': '2.0.0', 'selectedCamera': {'locationId': '63146417eb9c1420f9bf0c01', 'edgeId': '64eefe8c062c9aaa00bd64ed', 'cameraId': '64fdc143ef834f99ea6bd9d1'}}

appearance_alert = {"_id":"Appearance_ID",
                    "action":   alertsActions.add.value,
                    "configuration": {
                         #"object": objectNames.person.value,
                         "objects": [{"object": objectNames.person.value},{"object":objectNames.vehicle.value}],

                         "detection":alertsType.appearance.value,
                         "detectionAdditionalAttributes": None,
                         "filters":{
                                    #"genderType": ["male"],
                                    #"ageType": [],
                                    #"upperbodyColor": ["grey"],
                                    #"lowerbodyColor": ["blue"],
                         }
                    },
                    'selectedCamera': {'locationId': '633994ad5c2e03efa5dce6ff', 'cameraId': '63468c0885203eb5c7a133d6', 'edgeId': '633998434b80396a8cccc9a7', 'timezone': 'Asia/Jerusalem'},
                     'settings': {
                                'additionalOptions': False,
                                'blockNotificationPeriod': None,"alertThumbnail":True,"alertZoomThumbnail":True},
                    #"settings": {"schedule":{"monday":{"from":"01:00 AM","to":"04:30 AM"},"wednesday":{"from":"01:00 PM","to":"04:30 PM"},"tuesday":{"from":"01:00 AM","to":"04:30 AM"},"thursday":{"from":"09:00 AM","to":"10:30 AM"},"friday":{"from":"01:00 AM","to":"04:30 AM"},"saturday":{"from":"01:00 AM","to":"04:30 AM"},"sunday":{"from":"01:00 AM","to":"04:30 AM"}}}
                    'actions': {'gpioActions': [], 'msgActions': [{'cloudHttp': '', 'localHttp': '', 'direct': {'protocol': 1, 'address': '192.168.86.60', 'port': '2222', 'msg': 'Lumix Message: Dog in the yard', 'body': ''}}]}

}
fadicha = {"_id":"6441da51a26739c780ae177f","name":"Tunnel","alertType":0,"configuration":{"object":0,"offender":None,"detection":0,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},"filters":{},"tresholdTime":0},"settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":None,"enableNotificationSound":True,"alertThumbnail":True,"alertZoomThumbnail":True},"notifications":{"orgUsers":[],"manualUsers":[],"notificationMethods":{}},"timezone":"America/New_York","actions":{"gpioActions":[],"msgActions":[],"directActions":[]},"zones":{},"definedZones":False,"markedIdx":[],"measureCrossZones":False,"selectedCamera":{"locationId":"64417234a26739c780ae1762","cameraId":"64419908a26739c780ae176b","edgeId":"6438ad15599deba77930bb53","timezone":"America/New_York"},"synced":True,"action":2,"orgId":"644171a8a26739c780ae175e","enabled":True,"groupId":"69a6bd39-9c37-4d86-88e3-8c0d0fc73c44","version":"1.0.0"}
appear_home = {'action': 0, '_id': '6442e81c591e3063542ec8ab', 'alertType': 0, 'selectedCamera': {'locationId': '63bd6f4fd67d82346855d43b', 'cameraId':
'64184fb03522bcf6cf849825', 'edgeId': '63bd6a94d67d82346855d439', 'timezone': 'Asia/Jerusalem'}, 'configuration': {'object': 0, 'detection': 0, 'detectionAdditionalAttributes': {'direction': 'above', 'count': 0, 'sensitivity': 0.5}, 'filters': {}, 'tresholdTime': 0}, 'settings': {'schedule': {'monday': None, 'tuesday': None, 'wednesday': None, 'thursday': None, 'friday'
: None, 'saturday': None, 'sunday': None}, 'frequency': 0, 'additionalOptions': False, 'pushAlert': True, 'blockNotificationPeriod': None, 'enableNotificationSound': False, 'alertThumbnail': True, 'alertZoomThumbnail': True}, 'enabled': True, 'timezone': 'Asia/Jerusalem', 'actions': {'gpioActions': [], 'msgActions': [], 'directActions': []}, 'measureCrossZones': False}

multi_appearance_alert = {"_id":"Multi_Appearance_ID",
                    "action":   alertsActions.add.value,
                    "configuration": {
                         "objects": [{"object": objectNames.person.value},{"object":objectNames.vehicle.value}],
                         "detection":alertsType.appearance.value,
                         "detectionAdditionalAttributes": None,
                         #"filters":{
                                    #"genderType": ["male"],
                                    #"ageType": [],
                                    #"upperbodyColor": ["grey"],
                                    #"lowerbodyColor": ["blue"],
                         #}
                    },
                    'selectedCamera': {'locationId': '633994ad5c2e03efa5dce6ff', 'cameraId': '63468c0885203eb5c7a133d6', 'edgeId': '633998434b80396a8cccc9a7', 'timezone': 'Asia/Jerusalem'},
                    #"settings": {"schedule":{"monday":{"from":"01:00 AM","to":"04:30 AM"},"wednesday":{"from":"01:00 PM","to":"04:30 PM"},"tuesday":{"from":"01:00 AM","to":"04:30 AM"},"thursday":{"from":"09:00 AM","to":"10:30 AM"},"friday":{"from":"01:00 AM","to":"04:30 AM"},"saturday":{"from":"01:00 AM","to":"04:30 AM"},"sunday":{"from":"01:00 AM","to":"04:30 AM"}}}
                    'actions': {'gpioActions': [], 'msgActions': [{'cloudHttp': '', 'localHttp': '', 'direct': {'protocol': 1, 'address': '192.168.86.60', 'port': '2222', 'msg': 'Lumix Message: Dog in the yard', 'body': ''}}]}

}

motion_alert = {"_id":"643f8d1b589eec72287dfade","name":"temp2","alertType":0,"configuration":{"object":0,"offender":None,"detection":7,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},"filters":{},"tresholdTime":0},"settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":None,"enableNotificationSound":False,"alertThumbnail":True,"alertZoomThumbnail":True},"notifications":{"orgUsers":[],"manualUsers":[],"notificationMethods":{}},"timezone":"Asia/Beirut","actions":{"gpioActions":[],"msgActions":[],"directActions":[]},"zones":{},"definedZones":False,"markedIdx":[],"measureCrossZones":False,"selectedCamera":{"locationId":"643e3760589eec72287dfab4","cameraId":"643e4947589eec72287dfabe","edgeId":"643e34b2589eec72287dfab3","timezone":"Asia/Beirut"},"synced":True,"action":0,"orgId":"63bd6ed3d67d82346855d43a","enabled":True,"groupId":"5ed80326-30fb-4d47-826e-f18cfc580d5b","version":"1.0.0"}

detector_alert = {"_id":"63a404538ff1045894cf6584","name":"Detector","alertType":0,
                    "selectedCamera":{"locationId":"63627fbc28f6e6f856aa4884","cameraId":"637a3d3c2def796c9672e6e5","edgeId":"6374e39fcf376cea1a0b16a4","timezone":"Asia/Jerusalem"},
                    "configuration":{"object":1,"detection":8,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},"filters":{},"tresholdTime":0},
                    "settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},"frequency":0,"additionalOptions":False,"pushAlert":False,"blockNotificationPeriod":None,"enableNotificationSound":False,"alertThumbnail":True,"alertZoomThumbnail":True},
                    "notifications":{"orgUsers":[],"manualUsers":[],"notificationMethods":{}},
                    "timezone":"Asia/Jerusalem",
                    "actions":{"gpioActions":[],"msgActions":[],"directActions":[{"address":"192.168.200.19","port":"5555","protocol":1,"trafficController":True,"msg":"","body":""}]},
                    "zones":{"{\"x\":0.4468937875751503,\"y\":0.2801418439716312}":{"name":"21","color":"green","selection":[{"x":0.4468937875751503,"y":0.2801418439716312},{"x":0.43687374749499,"y":0.3617021276595745},{"x":0.5120240480961924,"y":0.36879432624113473},{"x":0.5110220440881763,"y":0.2872340425531915}],"markedIdx":[302,303,334,335,366,367]}
                            #,"{\"x\":0.5100200400801603,\"y\":0.2765957446808511}":{"name":"4","color":"blue","selection":[{"x":0.5100200400801603,"y":0.2765957446808511},{"x":0.5140280561122245,"y":0.375886524822695},{"x":0.5871743486973948,"y":0.3617021276595745},{"x":0.56312625250501,"y":0.2801418439716312}],"markedIdx":[304,305,336,337,338,368,369,370]},
                            #"{\"x\":0.43887775551102204,\"y\":0.2801418439716312}":{"name":"2","color":"yellow","selection":[{"x":0.43887775551102204,"y":0.2801418439716312},{"x":0.43286573146292584,"y":0.35815602836879434},{"x":0.3657314629258517,"y":0.35815602836879434},{"x":0.39579158316633267,"y":0.274822695035461}],"markedIdx":[300,301,332,333]},
                            #"{\"x\":0.38276553106212424,\"y\":0.2765957446808511}":{"name":"3","color":"purple","selection":[{"x":0.38276553106212424,"y":0.2765957446808511},{"x":0.3667334669338677,"y":0.3546099290780142},{"x":0.218436873747495,"y":0.35106382978723405},{"x":0.2995991983967936,"y":0.2730496453900709}],"markedIdx":[297,298,299,328,329,330,331]}
                            },
                    "definedZones":True,
                    "markedIdx":[265,268,297,298,299,300,301,302,303,304,305,328,329,330,331,332,333,334,335,336,337,338,363,364,365,366,367,368,369,370],
                    "measureCrossZones":False,"synced":True,"action":0,"orgId":"636279a628f6e6f856aa4881","enabled":True,"version":"0.1.606"}


audio_alerts = {"_id":"63e112a95b87e963ff539306","name":"audio alarm","alertType":0,"configuration":{"object":0,"detection":3,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},"filters":{"ageType":[],"carryingType":[],"lowerbodyType":[],"upperbodyType":[],"accessoryType":[],"footwearType":[],"hairType":[],"genderType":[],"upperbodyColor":[],"lowerbodyColor":[],"hairColor":[],"footwearColor":[]},"tresholdTime":0},"settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":None,"enableNotificationSound":False,"alertThumbnail":True,"alertZoomThumbnail":True},"notifications":{"orgUsers":[],"manualUsers":[],"notificationMethods":{}},"timezone":"Asia/Jerusalem","actions":{"gpioActions":[],"msgActions":[{"cloudHttp":"","localHttp":"http://192.168.200.25/api/v1/pattern/play","method":0,"body":"{\"pattern_number\": 1, \"volume\": 0, \"playcount\": 1, \"interval\": 0}"}],"directActions":[]},"lineCrossing":{"p1":{"x":0.32,"y":0.76},"p2":{"x":0.57,"y":0.73},"d":0,"state":2},"selectedCamera":{"locationId":"63146417eb9c1420f9bf0c01","cameraId":"635a81f7ff97b58676cb86a1","edgeId":"62f4e3ddfd167d8aec2008c8","timezone":"Asia/Jerusalem"},"synced":True,"action":2,"orgId":"63146239eb9c1420f9bf0bfd","enabled":True,"groupId":"b96e8de2-879d-45ce-b3c8-48631268b0ca","version":"0.1.794"}
disappeare_alert = {"_id":"Disppearance_ID",
                    "action":   alertsActions.add.value,
                    "configuration": {
                         "object": objectNames.person.value,
                         "detection":alertsType.disappeare.value,
                         "detectionAdditionalAttributes": None,
                         "filters":{
                                    #"genderType": ["male"],
                                    "ageType": [],
                                    "lowerbodyColor": ["blue"],

                                    "upperbodyColor": ["grey"],
                                    #"lowerbodyColor": ["white"],
                         }
                    },
                    'selectedCamera': {'locationId': '633994ad5c2e03efa5dce6ff', 'cameraId': '63468c0885203eb5c7a133d6', 'edgeId': '633998434b80396a8cccc9a7', 'timezone': 'Asia/Jerusalem'},
                    'settings': {"schedule":{"monday":{"from":"01:00 AM","to":"03:59 PM"},"wednesday":{"from":"01:00 PM","to":"03:30 PM"},"tuesday":{"from":"01:00 AM","to":"04:30 AM"},"thursday":{"from":"09:00 AM","to":"10:30 AM"},"friday":{"from":"01:00 AM","to":"04:30 AM"},"saturday":{"from":"01:00 AM","to":"04:30 AM"},"sunday":{"from":"01:00 AM","to":"04:30 AM"}}},
                    'actions': {'gpioActions': [], 'msgActions': [{'cloudHttp': '', 'localHttp': '', 'direct': {'protocol': 1, 'address': '192.168.86.60', 'port': '2222', 'msg': 'Lumix Message: Dog in the yard', 'body': ''}}]}

}

loitering_alert = {"_id":"Loitering_ID",
                    "action":   alertsActions.add.value,
                    "configuration": {
                         "object": objectNames.person.value,
                         "detection":alertsType.loitering.value,
                         "detectionAdditionalAttributes": {
                             "count": 5,
                             "direction": "above"
                         },
                         "filters":{
                                    #"genderType": ["male"],
                                    "ageType": [],
                                    #"upperbodyColor": ["blue"],
                                    #"lowerbodyColor": ["white"],
                         }
                    },
                    'settings': {
                                'additionalOptions': False,
                                'blockNotificationPeriod': None,"alertThumbnail":True,"alertZoomThumbnail":True}
}

tailgate_alert = {"action": 0,
                            "_id": 'tailgatingAlertID',
                            "action":   alertsActions.add.value,
                            "alertType": 0,
                            "configuration":
                                {'objects': [{"object": objectNames.person.value},{"object": objectNames.pet.value}],
                                'detection': alertsType.tailgating.value,
                                "detectionAdditionalAttributes": {
                                    "direction": "below",
                                    "count":3
                                    }
                                },
                            'settings': {
                                'additionalOptions': False,
                                'blockNotificationPeriod': None},
                            'lineCrossing': {'p1': {'x': 0.73, 'y': 0.54}, 'p2': {'x': 0.21, 'y': 0.76}, 'd': 1},
                            "enabled": True
                            }





tailgateAlertMsg = {
        "edgeId"   : DeviceID,
        "cameraId" : CameraID,
        "msgAction" : 2,
        "alert": tailgate_alert
    }

lpr_alert = {"_id":"lprAppearance_ID",
                    "action":   alertsActions.add.value,
                    "configuration": {
                         "object": objectNames.vehicle.value,
                         "detection":alertsType.lpr.value,
                         "filters":{
                             "greenList": "4736833",
                             "unrecognized": True
                         }
                    },
                    "settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},"frequency":0,"additionalOptions":False,"blockNotificationPeriod":None,"enableNotificationSound":False,"alertThumbnail":True,"alertZoomThumbnail":True},
                    "enabled": True

}

owner = {"_id":"641e037bf76a57af5b2cb9c9","name":"Owner Arrive Alert with LPR","alertType":0,
"configuration":{"object":1,"offender":None,"detection":4,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},
"filters":{"greenList":" ","redList":"","unrecognized":True},"tresholdTime":0},"settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},
"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":None,"enableNotificationSound":False,"alertThumbnail":True,"alertZoomThumbnail":True},
"notifications":{"orgUsers":[],"manualUsers":[],"notificationMethods":{}},"timezone":"America/New_York","actions":{"gpioActions":[],"msgActions":[],"directActions":[]},
"zones":{"{\"x\":0.6152304609218436,\"y\":0.1276595744680851}":{"name":"","color":"yellow","selection":[{"x":0.6152304609218436,"y":0.1276595744680851},{"x":0.5991983967935872,"y":0.33156028368794327},{"x":0.9879759519038076,"y":0.34929078014184395},{"x":0.9288577154308617,"y":0.1524822695035461}],
"markedIdx":[148,149,150,151,152,180,181,182,183,184,185,186,187,188,189,211,212,213,214,215,216,217,218,219,220,221,243,244,245,246,247,248,249,250,251,252,253,254,275,276,277,278,279,280,281,282,283,284,285,286,307,308,309,310,311,312,313,314,315,316,317,318,339,340,341,342,343,344,345,346,347,348,349,350]},"{\"x\":0.5280561122244489,\"y\":0.13652482269503546}":{"name":"","color":"blue","selection":[{"x":0.5280561122244489,"y":0.13652482269503546},{"x":0.5490981963927856,"y":0.3280141843971631},{"x":0.012024048096192385,"y":0.324468085106383},{"x":0.006012024048096192,"y":0.10638297872340426}],"markedIdx":[96,97,98,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,329,330,331,332,333,334,335,336,337]}},"definedZones":True,
"markedIdx":[96,97,98,99,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,148,149,150,151,152,153,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,180,181,182,183,184,185,186,187,188,189,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,211,212,213,214,215,216,217,218,219,220,221,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,243,244,245,246,247,248,249,250,251,252,253,254,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,275,276,277,278,279,280,281,282,283,284,285,286,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,307,308,309,310,311,312,313,314,315,316,317,318,320,325,326,327,328,329,330,331,332,333,334,335,336,337,339,340,341,342,343,344,345,346,347,348,349,350],
"measureCrossZones":False,
"selectedCamera":{"locationId":"641878adf76a57af5b2cb973","cameraId":"641ccab6f76a57af5b2cb9b1","edgeId":"63e1ac2384ff2e2945fb53cd","timezone":"America/New_York"},
"synced":True,"action":2,"orgId":"64187838f76a57af5b2cb96f","enabled":True,"groupId":"b3c75632-4e55-437a-87e6-1ff4725c67ba","version":"1.0.0"}
home_traffic = {"_id":"640f4f8edec4960c639824d2","name":"traffic","alertType":0,"configuration":{"object":1,"offender":None,"detection":10,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},"filters":{},"tresholdTime":0},"settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":None,"enableNotificationSound":False,"alertThumbnail":True,"alertZoomThumbnail":True},"notifications":{"orgUsers":[],"manualUsers":[],"notificationMethods":{}},"timezone":"Asia/Jerusalem","actions":{"gpioActions":[{"id":17,"action":True}],"msgActions":[],"directActions":[]},"trafficControl":{"state":0,"lines":[{"p1":{"x":0.63,"y":0.32},"p2":{"x":0.89,"y":0.25},"d":2,"color":"blue"},{"p1":{"x":0.65,"y":0.64},"p2":{"x":0.96,"y":0.38},"d":2,"color":"green"}],"distanceUnits":"meter"},"selectedCamera":{"locationId":"63bd6f4fd67d82346855d43b","cameraId":"63e8f885f71a5794848f5e3d","edgeId":"63bd6a94d67d82346855d439","timezone":"Asia/Jerusalem"},"synced":True,"action":2,"orgId":"63bd6ed3d67d82346855d43a","enabled":True,"groupId":"ebd26a87-a49f-442a-9b64-058a94c68e4d","version":"0.1.91"}
esp_alert_ll = {"_id":"63e2bd5784ff2e2945fb53dd","name":"LPR West","alertType":0,"configuration":{"object":1,"detection":4,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},"filters":{"greenList":" ","redList":"","unrecognized":True},"tresholdTime":0},"settings":{"schedule":{"monday":{"from":"12:00 AM","to":"11:59 PM"},"tuesday":{"from":"12:00 AM","to":"11:59 PM"},"wednesday":{"from":"12:00 AM","to":"11:59 PM"},"thursday":{"from":"12:00 AM","to":"11:59 PM"},"friday":{"from":"12:00 AM","to":"11:59 PM"},"saturday":{"from":"12:00 AM","to":"11:59 PM"},"sunday":{"from":"12:00 AM","to":"11:59 PM"}},"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":None,"enableNotificationSound":False,"alertThumbnail":True,"alertZoomThumbnail":True},"notifications":{"orgUsers":[],"manualUsers":[],"notificationMethods":{}},"timezone":"America/Chicago","actions":{"gpioActions":[],"msgActions":[],"directActions":[]},"zones":{"{\"x\":0.11322645290581163,\"y\":0.22163120567375885}":{"name":"1","color":"green","selection":[{"x":0.11322645290581163,"y":0.22163120567375885},{"x":0.41382765531062127,"y":0.1702127659574468},{"x":0.46893787575150303,"y":0.21099290780141844},{"x":0.467935871743487,"y":0.4627659574468085},{"x":0.23246492985971945,"y":0.5230496453900709}],"markedIdx":[199,200,201,202,203,204,205,206,228,229,230,231,232,233,234,235,236,237,238,260,261,262,263,264,265,266,267,268,269,270,293,294,295,296,297,298,299,300,301,302,325,326,327,328,329,330,331,332,333,334,357,358,359,360,361,362,363,364,365,366,390,391,392,393,394,395,396,397,398,422,423,424,425,426,427,428,429,430,455,456,457,458,459,460,461,462,487,488,489,490,491,492,519,520]},"{\"x\":0.5971943887775552,\"y\":0.25}":{"name":"","color":"blue","selection":[{"x":0.5971943887775552,"y":0.25},{"x":0.6112224448897795,"y":0.3900709219858156},{"x":0.6583166332665331,"y":0.4326241134751773},{"x":0.5511022044088176,"y":0.5212765957446809},{"x":0.46693386773547096,"y":0.5549645390070922},{"x":0.35671342685370744,"y":0.5975177304964538},{"x":0.5831663326653307,"y":0.9680851063829787},{"x":0.9839679358717435,"y":0.9787234042553191},{"x":0.9889779559118237,"y":0.3067375886524823},{"x":0.6763527054108216,"y":0.18617021276595744}],"markedIdx":[213,214,244,245,246,247,248,249,275,276,277,278,279,280,281,282,283,307,308,309,310,311,312,313,314,315,316,317,318,339,340,341,342,343,344,345,346,347,348,349,350,351,371,372,373,374,375,376,377,378,379,380,381,382,383,404,405,406,407,408,409,410,411,412,413,414,415,437,438,439,440,441,442,443,444,445,446,447,468,469,470,471,472,473,474,475,476,477,478,479,499,500,501,502,503,504,505,506,507,508,509,510,511,530,531,532,533,534,535,536,537,538,539,540,541,542,543,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,863,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,946,947,948,949,950,951,952,953,954,955,956,957,958,959,978,979,980,981,982,983,984,985,986,987,988,989,990,991]}},"definedZones":True,"markedIdx":[172,198,199,200,201,202,203,204,205,206,213,214,228,229,230,231,232,233,234,235,236,237,238,244,245,246,247,248,249,260,261,262,263,264,265,266,267,268,269,270,275,276,277,278,279,280,281,282,283,293,294,295,296,297,298,299,300,301,302,307,308,309,310,311,312,313,314,315,316,317,318,325,326,327,328,329,330,331,332,333,334,339,340,341,342,343,344,345,346,347,348,349,350,351,357,358,359,360,361,362,363,364,365,366,371,372,373,374,375,376,377,378,379,380,381,382,383,390,391,392,393,394,395,396,397,398,403,404,405,406,407,408,409,410,411,412,413,414,415,422,423,424,425,426,427,428,429,430,437,438,439,440,441,442,443,444,445,446,447,454,455,456,457,458,459,460,461,462,468,469,470,471,472,473,474,475,476,477,478,479,487,488,489,490,491,492,499,500,501,502,503,504,505,506,507,508,509,510,511,519,520,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,863,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,946,947,948,949,950,951,952,953,954,955,956,957,958,959,978,979,980,981,982,983,984,985,986,987,988,989,990,991,1023],"measureCrossZones":False,"selectedCamera":{"locationId":"63b74bfcfaf2c60c23b8fdcf","cameraId":"63e2a53c84ff2e2945fb53d9","edgeId":"635730eef5dee0ccdd0c7a63","timezone":"America/Chicago"},"synced":True,"action":2,"orgId":"63b74b74faf2c60c23b8fdcc","enabled":True,"groupId":"4089af4a-6439-49a1-a2e8-41bba4b9375d","version":"0.1.51"}
digi_alert = {"_id":"63f95b12381f9de3224c429d","name":"Wife's Home","alertType":0,"configuration":{"object":1,"detection":4,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},"filters":{"greenList":"","redList":"8RYM541","unrecognized":True},"tresholdTime":0},"settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":None,"enableNotificationSound":False,"alertThumbnail":True,"alertZoomThumbnail":True},"notifications":{"orgUsers":[{"id":"637cdd0e1ea29690489ab30e","firstname":None,"lastname":None,"email":"alarmperson@gmail.com","phone":"","roles":["owner"],"status":0}],"manualUsers":[],"notificationMethods":{"637cdd0e1ea29690489ab30e":"email"}},"timezone":"America/Los_Angeles","actions":{"gpioActions":[],"msgActions":[],"directActions":[]},"zones":{"{\"x\":0.10120240480961924,\"y\":0.39893617021276595}":{"name":"","color":"green","selection":[{"x":0.10120240480961924,"y":0.39893617021276595},{"x":0.09819639278557114,"y":0.7801418439716312},{"x":0.9739478957915831,"y":0.8333333333333334},{"x":0.9208416833667334,"y":0.50177304964539},{"x":0.6292585170340681,"y":0.36879432624113473}],"markedIdx":[391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,859,860,861,862]}},"definedZones":True,"markedIdx":[367,368,369,370,371,372,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,854,855,856,857,858,859,860,861,862],"measureCrossZones":False,"selectedCamera":{"locationId":"63e554104de14c8b13922ffe","cameraId":"63f94e79381f9de3224c4295","edgeId":"63a0b83d86fd40b13706dfb8","timezone":"America/Los_Angeles"},"synced":True,"action":2,"orgId":"63db2eaa425fa69b6d2ea9f3","enabled":True,"groupId":"42697419-a2e0-479c-ae6f-6b4ce39d6475","version":"0.1.51"}

family = {"_id":"6441d081a26739c780ae177c","name":"Family Member @ Home ","alertType":0,"configuration":{"object":1,"offender":None,"detection":0,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},    "filters":{"make":["Land Rover","Mercedes-AMG","Mercedes-Benz"],"model":["Range Rover","Range Rover (L322)","Range Rover (L405)","Range Rover (P38A)","Range Rover Classic","Range Rover Sport","G 63 AMG","G"],"type":["suv"],"colors":["grey","blue"]},"tresholdTime":0},"settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":1,"enableNotificationSound":True,"alertThumbnail":True,"alertZoomThumbnail":True},"notifications":{"orgUsers":[{"id":"6441bc5aa26739c780ae1778","firstname":"Ariel","lastname":"Shlezinger","email":"ariel.shlezinger@fon-llc.com","phone":None,"roles":["admin"],"status":0}],"manualUsers":[],"notificationMethods":{"6441bc5aa26739c780ae1778":"email"}},"timezone":"America/New_York","actions":{"gpioActions":[],"msgActions":[],"directActions":[]},"zones":{"{\"x\":0.006012024048096192,\"y\":0.3546099290780142}":{"name":"","color":"green","selection":[{"x":0.006012024048096192,"y":0.3546099290780142},{"x":0.35771543086172347,"y":0.24113475177304963},{"x":0.5110220440881763,"y":0.12943262411347517},{"x":0.996993987975952,"y":0.41312056737588654},{"x":0.9809619238476954,"y":0.8085106382978723},{"x":0.6342685370741483,"y":0.8457446808510638},{"x":0.39478957915831664,"y":0.9929078014184397},{"x":0.01002004008016032,"y":0.9875886524822695}],"markedIdx":[144,174,175,176,177,178,205,206,207,208,209,210,211,212,236,237,238,239,240,241,242,243,244,245,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004]}},"definedZones":True,"markedIdx":[112,143,144,145,174,175,176,177,178,179,204,205,206,207,208,209,210,211,212,235,236,237,238,239,240,241,242,243,244,245,246,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,858,859,860,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006],"measureCrossZones":False,"selectedCamera":{"locationId":"64417234a26739c780ae1762","cameraId":"644192c2a26739c780ae1766","edgeId":"6438ad15599deba77930bb53","timezone":"America/New_York"},"synced":True,"action":2,"orgId":"644171a8a26739c780ae175e","enabled":True,"groupId":"82bf3731-0599-43a5-be7d-d05e574803ee","version":"1.0.0"}
family_lpr = {"_id":"6441d081a26739c780ae177c","name":"Family Member @ Home ","alertType":0,"configuration":{"object":1,"offender":None,"detection":0,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},"filters":{"make":["Land Rover","Mercedes-AMG","Mercedes-Benz"],"model":["G","GLC","glc-class","G 63 AMG","Range Rover","Range Rover (L322)","Range Rover (L405)","Range Rover (P38A)","Range Rover Classic","Range Rover Sport"],"colors":["grey","blue","black"]},"tresholdTime":0},"settings":{"schedule":{"monday":None,"tuesday":None,"wednesday":None,"thursday":None,"friday":None,"saturday":None,"sunday":None},"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":1,"enableNotificationSound":True,"alertThumbnail":True,"alertZoomThumbnail":True},"notifications":{"orgUsers":[{"id":"6441bc5aa26739c780ae1778","firstname":"Ariel","lastname":"Shlezinger","email":"ariel.shlezinger@fon-llc.com","phone":None,"roles":["admin"],"status":0}],"manualUsers":[],"notificationMethods":{"6441bc5aa26739c780ae1778":"email"}},"timezone":"America/New_York","actions":{"gpioActions":[],"msgActions":[],"directActions":[]},"zones":{},"definedZones":False,"markedIdx":[],"measureCrossZones":False,"selectedCamera":{"locationId":"64417234a26739c780ae1762","cameraId":"644192c2a26739c780ae1766","edgeId":"6438ad15599deba77930bb53","timezone":"America/New_York"},"synced":True,"action":2,"orgId":"644171a8a26739c780ae175e","enabled":True,"groupId":"82bf3731-0599-43a5-be7d-d05e574803ee","version":"1.0.0"}

def Header_generator(width, height, frame_number,  PictureSize):
    frame_type = 4
    format = 0
    key_frame = 1
    timestamp = int(round(time.time() * 1000))
    NumOfMotionVector = 0
    # Parse the core descriptor into its components...
    H1 = struct.pack("HHHBBI",frame_type, width, height, format, key_frame, frame_number)
    H2 = struct.pack("Q",timestamp)
    H3 = struct.pack("II",NumOfMotionVector, PictureSize)
    Header = H1 + H2 + H3
    return Header

def HeaderJson(width, height, timestamp, frame_number,  pointer = None):
    format = 0
    # Parse the core descriptor into its components...
    header = {}

    header["width"] = width
    header["height"] = height
    header["format"] = format
    header["frame_number"] = frame_number
    header["timestamp"] = timestamp
    header["edgeId"] = DeviceID
    header["cameraId"] = CameraID
    header["pointer"] = pointer

    return header

def YUV420Bytes(array):
        Y  = array[:,:,0]
        U = array[::2,::2,1]
        V = array[::2,::2,2]
        payload = Y.tobytes() + U.tobytes() + V.tobytes()
        return payload

def YUVto420Bytes(array):
        Y  = array[:,:,0]
        U = array[:,:,1]
        V = array[:,:,2]

        # U sub sampling
        U = U[::2]
        U = U.flatten()
        U = U[::2]

        # V sub sampling
        V = V[::2]
        V = V.flatten()
        V = V[::2]

        payload = Y.tobytes() + U.tobytes() + V.tobytes()
        return payload

#TODO: how to pass a pointer over json and move to CUDA???
def read_buffer(path):
    with open(path, 'rb') as f:
        return str(base64.b64encode(f.read()))



def buildLibrary(DeviceID, CameraID, location):
    print(f'Building libraries on {location}')
    command_dir = f'rm -rf {config["locations"][location]}'
    subprocess.call(command_dir,shell=True)
    command_dir = f'mkdir {config["locations"][location]}'
    subprocess.call(command_dir,shell=True)
    command_dir = f'mkdir {config["locations"][location]}/{DeviceID}'
    subprocess.call(command_dir,shell=True)
    command_dir = f'mkdir {config["locations"][location]}/{DeviceID}/{CameraID}/'
    subprocess.call(command_dir,shell=True)

def buildAssets(DeviceID, CameraID, location):
    print(f'Building assets on /usr/src/app/{config["locations"][location]}/')

    command_dir = f'mkdir /usr/src/app/{config["locations"][location]}/{DeviceID}'
    subprocess.call(command_dir,shell=True)
    command_dir = f'mkdir /usr/src/app/{config["locations"][location]}/{DeviceID}/{CameraID}/'
    subprocess.call(command_dir,shell=True)



def send_video_i(queue_name, rate, video_path):

    buildLibrary(DeviceID, CameraID, "analyticInImages")
    buildLibrary(DeviceID, CameraID, "alertThumbnails")
    buildLibrary(DeviceID, CameraID, "thumbnails")
    buildLibrary(DeviceID, CameraID, "trainingThumbnails")
    buildAssets(DeviceID, CameraID, "assets")

    # Analytic parameters
    cap = cv2.VideoCapture(video_path)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    skips = np.floor(cap_fps / rate)

    i = 0
    sent = 0
    print("started sending images")
    t = time.time()
    repeat = 0
    rescale = 1

    if repeat > 0:
        i_offset = round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) * repeat
        ts_offset = round(i_offset / cap_fps * 1000)
    else:
        i_offset = 0
        ts_offset = int(time.time()*1000)
    store_path = os.path.join(config["locations"]["analyticInImages"], DeviceID, CameraID)
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        if i % skips == 0:
            if rescale != 1:
                image = cv2.resize(image, (int(image.shape[1] * rescale), int(image.shape[0] * rescale)))
            yuv = yuv_to_420_bytes(cv2.cvtColor(image, cv2.COLOR_BGR2YUV))
            ts = round(cap.get(cv2.CAP_PROP_POS_MSEC)) + ts_offset
            local_path = os.path.join(store_path, f"img-{ts}.yuv")
            with open(local_path, "wb") as f:
                f.write(yuv)
            message = {
                "timestamp": ts,
                "frameNumber": i + i_offset,
                "width": 0,
                "height": 0,
            }
            channel.basic_publish(exchange='', routing_key=queue_name, body=json.dumps(message))
            time.sleep(1 / rate)
            sent += 1
        i += 1
    print("Completed sending images, sent:" + str(sent))


NumOfFrames = 0
SentImages = 0


def send_images_i(QueueName,Rate,image_path, video_path=None):
    global SentImages
    global NumOfFrames
    RateDelay = 1 / Rate

    if (video_path==None) : file_names = glob.glob(image_path+'*.jpg')
    else:
        if (jetson):
            if (gpu_pointer): file_names = glob.glob(image_path+'*.buffer')
            else            : file_names = glob.glob(image_path+'*.yuv')
        else:                 file_names = glob.glob(image_path+'*.bmp')

    file_names.sort()
    NumOfFrames = len(file_names)
    print('Found ' + str(NumOfFrames)+' files on '+ image_path)

    buildLibrary(DeviceID, CameraID, "analyticInImages")
    buildLibrary(DeviceID, CameraID, "alertThumbnails")
    buildLibrary(DeviceID, CameraID, "thumbnails")
    buildLibrary(DeviceID, CameraID, "trainingThumbnails")
    buildAssets(DeviceID,  CameraID, "assets")

    timestamp = int(round(time.time() * 1000))
    Headers   = []
    pointer   = None
    #reading resolution when non images are supplied
    if not(video_path==None) and (jetson):
        vcap = cv2.VideoCapture(video_path)
        width  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    #copy all files to RAM to free compute on analytics side
    for idx, name in enumerate(file_names):
        i_timestamp = timestamp + int(RateDelay*1000)*idx
        path = f'{config["locations"]["analyticInImages"]}/{DeviceID}/{CameraID}/img-{str(i_timestamp)}.yuv'

        if (not(video_path==None)) and (jetson) :
            if (gpu_pointer) :
                pointer = json.dumps(read_buffer(name))
            else:
                command_mv = f'mv {name} {path}'
                subprocess.call(command_mv, shell=True)

        #write jpg/bmp files as yuv to RAM
        else:
            img = cv2.imread(name)
            height, width = img.shape[:2]
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv_bytes = YUVto420Bytes(yuv)
            file = open(path, "wb")
            file.write(yuv_bytes)

        Header = HeaderJson(width,height,i_timestamp, idx, pointer)
        Headers.append(Header)

    #Running now to send messages to analytics
    #getting queue status

    for Header in Headers:
        channel.basic_publish(exchange='', routing_key=QueueName, body=json.dumps(Header))
        SentImages+=1
        rabbit_queue = channel.queue_declare(queue=QueueName, arguments={'x-message-ttl' : config["msgQueue"]["bmpMessageTTL"]}, passive=True)
        message_count = rabbit_queue.method.message_count
        if not(burst_test) or message_count >= (int(config["streamer"]["skipOneAnalyticFiles"] - 10)):
            time.sleep(RateDelay)


    print("Completed sending images, sent:" + str(SentImages))


def send_images(QueueName,Rate,image_path):
    send_images_i(QueueName,Rate,image_path)


def yuv_to_420_bytes(array):
    y_plane = array[:, :, 0]
    u_plane = array[:, :, 1]
    v_plane = array[:, :, 2]

    # U sub sampling
    u_plane = u_plane[::2]
    u_plane = u_plane.flatten()
    u_plane = u_plane[::2]

    # V sub sampling
    v_plane = v_plane[::2]
    v_plane = v_plane.flatten()
    v_plane = v_plane[::2]

    return y_plane.tobytes() + u_plane.tobytes() + v_plane.tobytes()

def clean_video(output):
    command_dir = f'rm -rf {output}/tmp'
    subprocess.call(command_dir,shell=True)

def extract_video(video,rate,output):
    #clean previous files
    clean_video(output)
    command_dir = f'mkdir {output}/tmp/'

    command_ffmpeg = f'ffmpeg -i {video} -r {rate} {output}/tmp/$filename%05d.bmp'
    if (gpu_pointer): command_gstreamer = f'gst-launch-1.0 filesrc location={video} ! decodebin ! videorate max-rate={rate} ! multifilesink location="{output}/tmp/$filename%05d.buffer"'
    else:             command_gstreamer = f'gst-launch-1.0 filesrc location={video} ! decodebin ! nvvidconv ! "video/x-raw,format=I420" ! videorate ! video/x-raw,framerate={rate}/1 ! multifilesink location="{output}/tmp/$filename%05d.yuv"'

    subprocess.call(command_dir,shell=True)

    if jetson:
        print(command_gstreamer)
        subprocess.call(command_gstreamer,shell=True)

    else:
        print(command_ffmpeg)
        subprocess.call(command_ffmpeg,shell=True)

def send_video(QueueName,Rate,video_path):
    file_names = glob.glob(video_path+'*.mp4')

    if len(file_names)>0:
        if False:
            extract_video(file_names[0],Rate,video_path)
            image_path = video_path+"/tmp/"
            send_images_i(QueueName,Rate,image_path,file_names[0])
        else:
            send_video_i(QueueName,Rate,file_names[0])
    else:
        print("Didnt find mp4 file")

    while (True):
        #getting queue status
        rabbit_queue = channel.queue_declare(queue=QueueName, arguments={'x-message-ttl' : config["msgQueue"]["bmpMessageTTL"]}, passive=True)
        message_count = rabbit_queue.method.message_count
        time.sleep(1)
        if (message_count == 0): break

    if opt.video:
        print("Starting to process outputs")
        frames_to_video(resultpath,opt.thum_rate,opt.mot_rate)
        print("Detected alerts")
        grep_alerts = f'grep alertM {resultpath}/Json_*'
        subprocess.call(grep_alerts,shell=True)
        print("Alerts ID")
        grep_alerts = f'grep eventId {resultpath}/Json_*'
        subprocess.call(grep_alerts,shell=True)
        print("Variables")
        print(variables)


        print("Finished running press Ctrl^C")

def get_image(body):
    #image_bytes = io.BytesIO(body)
    #image = numpy.load(image_bytes, allow_pickle=True)
    #
    image = cv2.imdecode(numpy.frombuffer(body, dtype=numpy.uint8),cv2.IMREAD_UNCHANGED)
    return image

def frames_to_video(path,t_fps,m_fps):

   outputpath = path+'/movie/'
   command_dir = f'mkdir /{outputpath}'
   subprocess.call(command_dir,shell=True)

   ffmpeg_cmd = f'ffmpeg -framerate {t_fps} -pattern_type  glob -i "{path}/thumbnail-*0.jpg" {outputpath}/thum_output.mp4'
   subprocess.call(ffmpeg_cmd,shell=True)
   print("Writing thumbnail video using " + ffmpeg_cmd)

   ffmpeg_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {outputpath}/thum_output.mp4'
   result = subprocess.run([ffmpeg_cmd],shell=True,stdout=subprocess.PIPE)
   res = result.stdout.decode('utf-8')
   res = res.split('\n')[0]
   print(res)

   ffmpeg_cmd = f'ffmpeg -framerate {m_fps} -pattern_type  glob -i "{path}/Motion_*.png" -vf "scale={res}" {outputpath}/motion_output.mp4'
   subprocess.call(ffmpeg_cmd,shell=True)
   print("Writing motion video using " + ffmpeg_cmd)


   ffmpeg_cmd = f'ffmpeg -i {outputpath}/thum_output.mp4 -i {outputpath}/motion_output.mp4 -filter_complex hstack {outputpath}/output.mp4'
   subprocess.call(ffmpeg_cmd,shell=True)

   print("Writing merged video using " + ffmpeg_cmd)
   print("Finished writing output movie")

def CountAlerts(results):
    count = 0
    for result in results:
        if "alerts" in result:
            alerts = result["alerts"]

            for alert in alerts:
                if "thumbnails" in alert:
                    count+=1

    print(f'found {count} alerts')

def BuildJsonList(results):
    for result in results:
        if "info" in result:    info_list.append({"info" : result["info"]})
        if "objects" in result: objects_list.append({"objects": result["objects"]})



class ThreadedAnalysis(threading.Thread):
    def __init__(self, ThumbQueueName, JsonQueueName, MotQueueName, TraQueueName, variablesQueue, resultpath, BatchSize):
        threading.Thread.__init__(self)

        parameters = pika.ConnectionParameters('rabbitMQ')
        connection = pika.BlockingConnection(parameters)
        self.channel = connection.channel()
        self.channel.queue_declare(queue=ThumbQueueName, arguments={'x-message-ttl' : config["msgQueue"]["analyticsResultsQueueTTL"]})
        self.channel.queue_declare(queue=JsonQueueName, arguments={'x-message-ttl' : config["msgQueue"]["analyticsResultsQueueTTL"]})
        self.channel.queue_declare(queue=MotQueueName, arguments={'x-message-ttl' : config["msgQueue"]["motionVectorsQueueTTL"]})
        self.channel.queue_declare(queue=TraQueueName, arguments={'x-message-ttl' : config["msgQueue"]["trainingThumbanilQueueTTL"]})
        self.channel.queue_declare(queue=variablesQueue, arguments={'x-message-ttl' : config["msgQueue"]["variablesQueueTTL"]})

        self.channel.basic_consume(ThumbQueueName, on_message_callback=self.callback)
        self.channel.basic_consume(JsonQueueName, on_message_callback=self.callback)
        self.channel.basic_consume(MotQueueName, on_message_callback=self.callback)
        self.channel.basic_consume(TraQueueName, on_message_callback=self.callback)
        self.channel.basic_consume(variablesQueue, on_message_callback=self.callback)


        self.resultpath = resultpath
        self.BatchSize = BatchSize
        threading.Thread(target=self.channel.basic_consume(ThumbQueueName, on_message_callback=self.callback))
        threading.Thread(target=self.channel.basic_consume(JsonQueueName, on_message_callback=self.callback ))
        threading.Thread(target=self.channel.basic_consume(MotQueueName, on_message_callback=self.callback ))
        threading.Thread(target=self.channel.basic_consume(TraQueueName, on_message_callback=self.callback ))
        threading.Thread(target=self.channel.basic_consume(variablesQueue, on_message_callback=self.callback ))




    def callback(self, channel, method_frame, header_frame, body):
        global RcvdImages
        global RcvdTrain
        global RcvdJson
        global RcvdMotion
        global resultpath
        QueueName = method_frame.routing_key
        bodyjson = json.loads(body)

        # Sending Ack once completed
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)

        print("got results from queue: " + QueueName)
        try:
            if MotName in QueueName:
                s_ts = bodyjson['startTimestamp']
                e_ts = bodyjson['endTimestamp']
                nf = bodyjson['numberOfFrames']
                img=numpy.zeros(32 * 32)
                motion_dict = bodyjson["vectorJson"]
                for bid in motion_dict:
                    img[int(bid[1:])] = motion_dict[bid]


                # Strech to 255 for better view
                img = img.reshape(32,32).astype(float)
                img *= 2.55  # max is 100
                img = img.astype(int)
                filename = self.resultpath + '/Motion_'+str(s_ts)+'_'+str(e_ts)+'_'+str(nf)+'.png'
                cv2.imwrite(filename, img)
                print('Saved motion image: ' + str(RcvdMotion))
                with open(Path(filename).with_suffix(".json"), "wt") as fp:
                    json.dump(bodyjson, fp, indent=4)

                RcvdMotion += 1

            if (JsonName in QueueName):
                timestamp = bodyjson['metadata']['timestamp']
                results = bodyjson['results']
                #CountAlerts(results)
                BuildJsonList(results)
                #print(body)
                name = f'{self.resultpath}/Json_{timestamp}'
                if "splitPart" in bodyjson['metadata']:
                    name += f'_{bodyjson["metadata"]["splitPart"]}'

                with open(name + '.txt', 'w') as outfile:
                    json.dump(bodyjson, outfile, sort_keys=True, indent=4, ensure_ascii=False)
                if opt.parse_json:
                    with open(self.resultpath + '/info.json', "w") as output:
                        output.write(str(info_list))
                    with open(self.resultpath + '/objects.json', "w") as output:
                        output.write(str(objects_list))
                RcvdJson += 1

                alerts = results[0].get("alerts", [])
                if alerts:
                    for alert in alerts:
                        print(f"Alert raised: {alert}")
                        for thumb in alert.get("thumbnails", []):
                            source = os.path.join(config["locations"]["alertThumbnails"], DeviceID, CameraID, thumb)
                            dest = os.path.join(self.resultpath, thumb)
                            shutil.move(source, dest)


            if (ThumbName in QueueName):
                #print(bodyjson)
                timestamp = bodyjson["normalizedTimestamp"]
                thumbnail = bodyjson["thumbnail"]
                with open(self.resultpath + f'/ThumInfo_{thumbnail}.txt', 'w') as outfile:
                    json.dump(bodyjson, outfile, sort_keys=True, indent=4, ensure_ascii=False)
                source = os.path.join(config["locations"]["thumbnails"], DeviceID, CameraID, thumbnail)
                dest = os.path.join(self.resultpath, thumbnail)
                shutil.move(source, dest)
                # events = bodyjson["events"]
                # for event in events:
                #     #move global thumbnail
                #     file_path = globalpath+event["mainThumbnail"]
                #     mv_cmd = f'mv {file_path} {self.resultpath}/'
                #     subprocess.call(mv_cmd, shell=True)
                #     #print(mv_cmd)
    #
                #     #move alerts
                #     for file_name in event["alerts"]:
                #         file_path = alertpath+file_name
                #         mv_cmd = f'mv {file_path} {self.resultpath}/'
                #         subprocess.call(mv_cmd,shell=True)
                #         #print(mv_cmd)

                #print(f'Saved thum:{timestamp} image number: {RcvdImages}')
                RcvdImages += 1


            if (TrainName in QueueName):
                globalpath = f'{config["locations"]["trainingThumbnails"]}/{DeviceID}/{CameraID}/'
                file_path  = globalpath+bodyjson["thumbnail"]
                timestamp  = bodyjson["timestamp"]
                mv_cmd = f'mv {file_path} {self.resultpath}/'
                subprocess.call(mv_cmd,shell=True)
                print(mv_cmd)
                print(f'Saved training:{timestamp} image number: {RcvdTrain}')
                RcvdTrain+=1
                with open(self.resultpath + '/TrainInfo_'+str(timestamp)+'.txt', 'w') as outfile:
                    json.dump(bodyjson, outfile, sort_keys = True, indent = 4,
                        ensure_ascii = False)

                #Sending Ack once completed
            if (VarName in QueueName):
                print(f'Variable update: {bodyjson}')
                if bodyjson:
                    for variable in bodyjson["variables"]:
                        if not(variable["variableId"] in variables):
                            variables[variable["variableId"]] = {"count": variable["count"]}
                        else:
                            variables[variable["variableId"]]["count"]+= variable["count"]
        except Exception as e:
            print(f"Error processing message: {e}")
            print(f"Error processing message: {bodyjson}")

    def run(self):
        print ('starting thread to consume from rabbit...')
        self.channel.start_consuming()



started_streaming = False
def start_streaming(channel, method_frame, header_frame, body):
    global started_streaming
    channel.basic_ack(delivery_tag=method_frame.delivery_tag)
    body_msg = json.loads(body)
    print(body_msg)
    if ("status" in body_msg and body_msg["status"] == e_ComponentStatus.Available.value[0] and not(started_streaming)) or not(config["analytics"]["threadCheckerEnable"]):
        started_streaming = True

        print(f"start streaming")
    elif "action" in body_msg and body_msg["action"] == "ModelVersions":
        print("Got Model version response, registered models:")
        print(body_msg["versions"])
        return
    else:
        print(f"Got analytic status")
        return



    esp_alert = {'msgAction': 2, "edgeId"   : DeviceID, "cameraId" : CameraID, 'alert': {"_id":"63e2bd5784ff2e2945fb53dd","name":"LPR West","alertType":0,"configuration":{"object":1,"detection":4,"detectionAdditionalAttributes":{"direction":"above","count":0,"sensitivity":0.5},"filters":{"greenList":" ","redList":"","unrecognized":True},"tresholdTime":0},"settings":{"schedule":{"monday":{"from":"12:00 AM","to":"11:59 PM"},"tuesday":{"from":"12:00 AM","to":"11:59 PM"},"wednesday":{"from":"12:00 AM","to":"11:59 PM"},"thursday":{"from":"12:00 AM","to":"11:59 PM"},"friday":{"from":"12:00 AM","to":"11:59 PM"},"saturday":{"from":"12:00 AM","to":"11:59 PM"},"sunday":{"from":"12:00 AM","to":"11:59 PM"}},"frequency":0,"additionalOptions":False,"pushAlert":True,"blockNotificationPeriod":None,"enableNotificationSound":False,"alertThumbnail":True,"alertZoomThumbnail":True},"notifications":{"orgUsers":[],"manualUsers":[],"notificationMethods":{}},"timezone":"America/Chicago","actions":{"gpioActions":[],"msgActions":[],"directActions":[]},"zones":{"{\"x\":0.5531062124248497,\"y\":0.9592198581560284}":{"name":"","color":"green","selection":[{"x":0.5531062124248497,"y":0.9592198581560284},{"x":0.5420841683366734,"y":0.49113475177304966},{"x":0.5791583166332666,"y":0.45567375886524825},{"x":0.5841683366733467,"y":0.2801418439716312},{"x":0.6432865731462926,"y":0.28191489361702127},{"x":0.9779559118236473,"y":0.40602836879432624},{"x":0.9809619238476954,"y":0.9716312056737588}],"markedIdx":[307,308,309,339,340,341,342,343,344,371,372,373,374,375,376,377,378,403,404,405,406,407,408,409,410,411,412,413,435,436,437,438,439,440,441,442,443,444,445,446,467,468,469,470,471,472,473,474,475,476,477,478,498,499,500,501,502,503,504,505,506,507,508,509,510,529,530,531,532,533,534,535,536,537,538,539,540,541,542,561,562,563,564,565,566,567,568,569,570,571,572,573,574,593,594,595,596,597,598,599,600,601,602,603,604,605,606,625,626,627,628,629,630,631,632,633,634,635,636,637,638,657,658,659,660,661,662,663,664,665,666,667,668,669,670,689,690,691,692,693,694,695,696,697,698,699,700,701,702,721,722,723,724,725,726,727,728,729,730,731,732,733,734,753,754,755,756,757,758,759,760,761,762,763,764,765,766,786,787,788,789,790,791,792,793,794,795,796,797,798,818,819,820,821,822,823,824,825,826,827,828,829,830,850,851,852,853,854,855,856,857,858,859,860,861,862,882,883,884,885,886,887,888,889,890,891,892,893,894,914,915,916,917,918,919,920,921,922,923,924,925,926,946,947,948,949,950,951,952,953,954,955,956,957,958,978,979,980,981,982,983,984,985,986,987,988,989,990]}},"definedZones":True,"markedIdx":[307,308,309,339,340,341,342,343,344,371,372,373,374,375,376,377,378,379,403,404,405,406,407,408,409,410,411,412,413,435,436,437,438,439,440,441,442,443,444,445,446,466,467,468,469,470,471,472,473,474,475,476,477,478,497,498,499,500,501,502,503,504,505,506,507,508,509,510,529,530,531,532,533,534,535,536,537,538,539,540,541,542,561,562,563,564,565,566,567,568,569,570,571,572,573,574,593,594,595,596,597,598,599,600,601,602,603,604,605,606,625,626,627,628,629,630,631,632,633,634,635,636,637,638,657,658,659,660,661,662,663,664,665,666,667,668,669,670,689,690,691,692,693,694,695,696,697,698,699,700,701,702,721,722,723,724,725,726,727,728,729,730,731,732,733,734,753,754,755,756,757,758,759,760,761,762,763,764,765,766,785,786,787,788,789,790,791,792,793,794,795,796,797,798,818,819,820,821,822,823,824,825,826,827,828,829,830,850,851,852,853,854,855,856,857,858,859,860,861,862,882,883,884,885,886,887,888,889,890,891,892,893,894,914,915,916,917,918,919,920,921,922,923,924,925,926,946,947,948,949,950,951,952,953,954,955,956,957,958,977,978,979,980,981,982,983,984,985,986,987,988,989,990],"measureCrossZones":False,"selectedCamera":{"locationId":"63b74bfcfaf2c60c23b8fdcf","cameraId":"63e2a53c84ff2e2945fb53d9","edgeId":"635730eef5dee0ccdd0c7a63","timezone":"America/Chicago"},"synced":True,"action":2,"orgId":"63b74b74faf2c60c23b8fdcc","enabled":True,"groupId":"4089af4a-6439-49a1-a2e8-41bba4b9375d","version":"0.1.50"}}

    #channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(esp_3))

    #channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(veichle_var))

    #channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(loitering_temp))
    #channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(tampering_alert))
    #channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(tailgateAlertMsg))
    #channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(searchEsp))
    #channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(searchMsgLegacy))

    #channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(greenLPR))
    #channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(redLPR))
    #channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(esp_alert))



    if (opt.video):
        print("Video is parsed")
        imgth = threading.Thread(target=send_video,args=(ImageQueueName,fps,opt.image_path,))
    else:
        print("Images are parsed")
        imgth = threading.Thread(target=send_images,args=(ImageQueueName,fps,opt.image_path,))

    print("Opening Thread for listening")
    ts = ThreadedAnalysis(ThumbQueueName, JsonQueueName,MotQueueName, TraQueueName, variablesQueue, resultpath, configAnalytic["pre_process"]["batchSize"])

    imgth.start()
    ts.start()
    imgth.join()
    ts.join()

    print("Done")


def main():
    global resultpath
    global opt
    global fps
    parser = argparse.ArgumentParser()
    parser.add_argument('--thum_rate', type=float, default=0.5, help='Rate in Hz for thumbnails')
    parser.add_argument('--mot_rate', type=float, default=0.25, help='Rate in Hz for motion')
    parser.add_argument('--image_path', type=str, default = '/usr/src/analytics_testing/data/' , help='Batch of images')
    parser.add_argument('--out_path', type=str, default = '/usr/src/analytics_testing/results/' , help='Batch of results')
    parser.add_argument('--video', type=bool, default = False , help='Run mp4 file')
    parser.add_argument('--parse_json', action='store_true', help='running with motion info')
    parser.add_argument('--no_parse-json', dest='motion', action='store_false', help='running without motion info')
    parser.set_defaults(parse_json=True)
    opt = parser.parse_args()
    config         = json.load(open('/usr/src/app/configs/config.json'))
    configAnalytic = json.load(open('/usr/src/app/configs/configAnalytic.json'))


    fps = int(1000/config["streamer"]["minMsBetweenFrames"])

    opt.thum_rate = min(fps,1000/configAnalytic["thumbnailPolicy"]["thumbnailsDuration"])
    opt.mot_rate = min(fps,1000/configAnalytic["motionPolicy"]["motionVectorsDuration"])


    if os.getenv('VIDEO_TEST'): opt.video = (os.getenv('VIDEO_TEST')=='True')
    if (opt.video):
        print("Running on images under " +  opt.image_path)
    else:
        print("Running on movies under " +  opt.image_path)

    resultpath = opt.out_path + str(round(time.time()))
    command_dir = f'mkdir /{opt.out_path}'
    subprocess.call(command_dir,shell=True)
    command_dir = f'mkdir /{resultpath}'
    subprocess.call(command_dir,shell=True)
    print("Creaed library on: " + command_dir)

    print("Removed old RAM data: " + command_dir)
    #Queue connection

    msgQueuse = config["msgQueue"]["analyticsCmdQueue"] + "_0"
    #output global
    channel.queue_declare(queue=msgQueuse,                               arguments={'x-message-ttl' : config["msgQueue"]["analyticsCmdQueueTTL"]})
    channel.queue_declare(queue=ImageQueueName,                          arguments={'x-message-ttl' : config["msgQueue"]["bmpMessageTTL"]})

    #input
    channel.queue_declare(queue=config["msgQueue"]["analyticsResponseQueue"], arguments={'x-message-ttl' : config["msgQueue"]["analyticsResponseQueueTTL"]})

    print("Starting Producing Data")
    print("Opening a Camera")

    if not(opt.video): configAnalytic["pre_process"]["batchSize"] = min(len(glob.glob(opt.image_path+'*.jpg')),config["analytics"]["analyticAppConfig"]["pre_process"]["batchSize"])


    #reading resolution when non images are supplied
    file_names = glob.glob(opt.image_path+'*.mp4')
    vcap = cv2.VideoCapture(file_names[0])
    width  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    custom_info = Path(file_names[0]).with_suffix(".info")
    if os.path.exists(custom_info):
        with open(custom_info, "rt") as f:
            test_info = json.load(f)
            alert_info = test_info["alerts"]
    else:
        alert_info = []

    MsgBody = {
        "msgAction" : 0,
        "edgeId" : DeviceID,
        "cameraId" : CameraID,

        "queue"                 : ImageQueueName,
        "analyticQueue"         : JsonQueueName,
        "thumbQueue"            : ThumbQueueName,
        "motionVectorsQueue"    : MotQueueName,
        "trainingQueue"         : TraQueueName,
        "variablesQueue"        : variablesQueue,


        "analyticAppParameters":{


            "alertPolicy":{
                #"alerts": [occupancy_alert, multi_appearance_alert, appearance_alert, disappeare_alert, loitering_alert, tailgate_alert, lpr_alert]
                #"alerts": [protective]
                "alerts": alert_info

            },

            "inputStream":{
                "width": width,
                "height" : height
            },

            #'searchPolicy': {'action': 2, '_id': '6458edb522fce9a9ebb40764', 'licensePlates': False, 'vehicleMMC': False, 'objectsToScan': [2, 1, 0], 'selectedCamera': {'cameraId': '64579d0f22fce9a9ebb40705', 'locationId': '643e3760589eec72287dfab4', 'edgeId': '642196b7f4a4160d1670fab5'}, 'markedIdx': [738, 739, 740, 770, 771, 772, 773, 774, 775, 776, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004], 'zones': {'{"x":0.06613226452905811,"y":0.7127659574468085}': {'name': '', 'color': 'green', 'selection': [{'x': 0.06613226452905811, 'y': 0.7127659574468085}, {'x': 0.050100200400801605, 'y': 0.9911347517730497}, {'x': 0.3897795591182365, 'y': 0.9893617021276596}, {'x': 0.39879759519038077, 'y': 0.7960992907801419}], 'markedIdx': [738, 739, 740, 770, 771, 772, 773, 774, 775, 776, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004], 'objectsToScan': [0, 1, 2]}}, 'privacy': {'zones': {}, 'definedZones': False, 'markedIdx': [], 'objectsToPixelate': [0],'privacyType': "blacken"}}
            'searchPolicy': {'action': 0, '_id': '64f9f6fb5ff88948b3cbe91d', 'licensePlates': False, 'vehicleMMC': False, 'objectsToScan': [2, 1, 0], 'protectiveGear': False, 'selectedCamera': {'cameraId': '64ef0fc05cc73624ae406802', 'locationId': '63146417eb9c1420f9bf0c01', 'edgeId': '64eefe8c062c9aaa00bd64ed'}}
          #'variablesPolicy': {'variables': [{'action': 0, '_id': '63da4b0f41584ce721f89589', 'variableType': 0, 'selectedCamera': {'edgeId': '62f4e3ddfd167d8aec2008c8', 'cameraId': '631732efc4b7fc310f14d011', 'locationId': '63146417eb9c1420f9bf0c01'}, 'configuration': {'object': 0, 'filters': {'accessoryType': [], 'ageType': ['18_to_60', 'over60'], 'carryingType': [], 'colors': [], 'footwearColor': [], 'footwearType': [], 'genderType': ['male'], 'greenList': '', 'hairColor': [], 'hairType': [], 'lowerbodyColor': [], 'lowerbodyType': [], 'make': [], 'model': '', 'redList': '', 'type': [], 'unrecognized': False, 'upperbodyColor': [], 'upperbodyType': []}}, 'periodInMinutes': 5, 'synced': False, 'enabled': True}, {'action': 0, '_id': '63d8fee4308566737725ec52', 'variableType': 0, 'selectedCamera': {'edgeId': '62f4e3ddfd167d8aec2008c8', 'cameraId': '631732efc4b7fc310f14d011', 'locationId': '63146417eb9c1420f9bf0c01'}, 'configuration': {'object': 0, 'filters': {'accessoryType': [], 'ageType': [], 'carryingType': [], 'colors': [], 'footwearColor': [], 'footwearType': [], 'genderType': [], 'greenList': '', 'hairColor': [], 'hairType': [], 'lowerbodyColor': [], 'lowerbodyType': [], 'make': [], 'model': '', 'redList': '', 'type': [], 'unrecognized': False, 'upperbodyColor': [], 'upperbodyType': []}}, 'periodInMinutes': 5, 'synced': False, 'enabled': True}]}
        }
    }

    model_version_req = {
        "msgAction": 12,  # ManagementMessage.GET_MODELS_VERSION.value,
        "edgeId": DeviceID,
        "cameraId": CameraID,
        "token": "112233"
    }

    channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(MsgBody))
    time.sleep(5)
    channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(model_version_req))
    time.sleep(3)
    channel.basic_consume(config["msgQueue"]["analyticsResponseQueue"], on_message_callback=start_streaming)

    channel.start_consuming()

    check_status_req = model_version_req

    check_status_req["msgAction"] = 8  # CHECK_ANALYTIC_STATUS
    channel.basic_publish(exchange='', routing_key=msgQueuse, body=json.dumps(check_status_req))



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
