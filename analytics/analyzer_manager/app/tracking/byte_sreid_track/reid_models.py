from typing import Dict, Optional

from general.analyzer_general import InferenceType
from level1.reid.BoT import BotConfig, BoT
from level1.vehicle.vehicle_parsing import BoTWithClassifier, BotWithClassifierConfig


class TrackerPersonReidConfig(BotConfig):
    weights: str = "resnet18_ibn_a_tracker_person_reid_0_1.pt"  # path for builtin weight file

    name: InferenceType = InferenceType.TRACKER_PERSON_REID
    feature_dim =  512
    model_name =  "resnet18_ibn_a"

class TrackerPersonReid(BoT):
    _config_type = TrackerPersonReidConfig



class TrackerVehicleReidConfig(BotWithClassifierConfig):
    weights: str = "resnet18_ibn_a_tracker_vehicle_reid_0_0.pt"  # path for builtin weight file
    name: InferenceType = InferenceType.TRACKER_VEHICLE_REID
    feature_dim =  512
    model_name =  "resnet18_ibn_a"
    attributes: str = "vehicle_attributes_1.1.csv"


class TrackerVehicleReid(BoTWithClassifier):
    _config_type = TrackerVehicleReidConfig