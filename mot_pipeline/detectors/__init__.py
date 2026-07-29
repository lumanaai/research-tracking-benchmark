from mot_pipeline.detectors.existing import ExistingDetsDetector
from mot_pipeline.detectors.gt import GTDetector
from mot_pipeline.detectors.yolo_ultralytics import YoloUltralyticsDetector
from mot_pipeline.detectors.yolox import YOLOXDetector

DETECTORS = {
    "gt": GTDetector,
    "yolov8": YoloUltralyticsDetector,
    "existing": ExistingDetsDetector,
    "yolox": YOLOXDetector,
}


def get_detector(name: str, **kwargs):
    if name not in DETECTORS:
        raise KeyError(f"Unknown detector '{name}'. Choose from: {sorted(DETECTORS)}")
    return DETECTORS[name](**kwargs)
