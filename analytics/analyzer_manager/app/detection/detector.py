import os.path
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Any, Dict, Optional

import albumentations as A
import cv2
import numpy as np

from general import proj
from general.analyzer_general import logger, InferenceType, DEFAULT_DETECTOR_IM_SIZE
from general.core import ClassHandler
from general.inference import InferenceWrapper, BaseInferenceConfig


class DetectorType(str, Enum):
    BASE = "base"
    YOLOV5 = "yolov5"
    YOLOX = "yolox"
    YOLOV8 = "yolov8"
    EXPERT = "expert-detect"
    WEAPON_EXPERT = "weapon-expert"
    CONTAINER_OCD = "container-ocd"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


@dataclass
class ImagePredictions:
    data: Any
    timestamp: Any


@dataclass
class DetectionResults:
    predictions: Any
    names: Any
    preds: Any


class BaseDetectorConfig(BaseInferenceConfig):
    classes_metadata: str = None  # if None, use num_classes to generate classes file
    imgsz: List[int] = DEFAULT_DETECTOR_IM_SIZE
    batch_size: int = 4
    conf_thres: float = 0.25  # confidence treshold
    iou_thres: float = 0.3  # maximum suprression tresh
    augment: bool = False  # augment in test and run a few times to get better results (worse time)
    line_thickness: int = 3
    max_det = 100  # max number of detections
    hide_labels = False  # Hide label for annotated image
    hide_conf = False  # Hide confidence for annotated image
    dnn = False  # use OpenCV DNN for ONNX inference
    agnostic_nms: bool = False  # Agnostic non max supression
    antialias: bool = True  # Anti aliasing when downsampling
    normalize: bool = True  # Normalize network input
    resize_factor: float = 0.25  # image resize factor before detector
    objects: List = None  # list of the objects to detect
    subClasses: List = None  # list of classes to detect
    classes: List = None  # List of all the classes - generated from objects and subclasses
    use_lock: bool = False  # weather to use lock or not
    obj_agnostic_nms: bool = True  # object agnostic non max supression
    name: InferenceType = InferenceType.DETECTION
    special_classes: List[str] = None
    num_classes: int = 27
    is_bypass: bool = False
    is_bgr: bool = False  # input images are in RGB format

    @property
    def im_size(self):
        return self.imgsz


class BaseDetector(InferenceWrapper):
    detector_type: DetectorType = DetectorType.BASE
    _config_type = BaseDetectorConfig
    args: BaseDetectorConfig
    model: Any
    _lock = threading.Lock()
    use_cuda_parser = True
    _classes_filter: List[int]

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super().__init__(msg_dict, is_local)

        self._load_classes()

        if self.args.is_bypass:
            self.run = self.run_bypassed

        if self.args.is_trt and self.is_local:
            self.forward_on_crop_list = self.forward_on_crop_list_static

    def build_triton_model(self):
        from general.triton_utils import DetectorTriton

        self.model = DetectorTriton(self.remote_info)
        self.args.imgsz = self.model.im_size

    def build_full_model(self):
        pass

    def _load_classes(self):
        if self.args.classes_metadata is None:
            n_classes = 23 if self.args.num_classes is None else self.args.num_classes
            self.args.classes_metadata = f"{n_classes}cls.csv"
        file_to_load = proj.info_path("detection", self.args.classes_metadata)
        if file_to_load is None:
            weight_type = Path(self.args.weights).stem.split("_")[0]
            weights_parts = weight_type.split("-")
            if len(weights_parts) > 1:
                file_to_load = weights_parts[-1] + ".csv"
            else:
                file_to_load = "coco_classes.csv"

            file_to_load = proj.info_path("detection", file_to_load)

        if not os.path.exists(file_to_load):
            file_to_load = proj.info_path("detection", "coco_classes.csv")
            logger.warning("reverting to default classes metadata file")

        class_handler = ClassHandler(file_to_load)
        self.n_classes = class_handler.num_classes
        self.classes_names = class_handler.classes_names
        self.object_names = class_handler.objects_names
        self._mapping = class_handler.class_to_object_mapping
        self._class_handler = class_handler

        subclasses_filter = [] if self.args.subClasses is None else self.args.subClasses
        objects_filter = [] if self.args.objects is None else self.args.objects

        if self.args.special_classes:

            subclasses_filter += self.args.special_classes
            additional_objs = [self.class_handler.get_object_from_class(c) for c in self.args.special_classes]
            objects_filter += additional_objs
            objects_filter = list(set(objects_filter))
        subclasses_filter = [
            self.class_handler.class_str_to_int(c) for c in subclasses_filter if c in self.class_handler.classes_names
        ]

        classes_filter = self.class_handler.filter_objects(subclasses_filter, objects_filter)

        if len(classes_filter) == 0:
            self.args.classes = np.argwhere(self._mapping >= 0).flatten()
            logger.info(f"Detector init without classes filter")
        else:
            self.args.classes = classes_filter
            logger.info(f"Detector init with classes: {self.args.classes}")
        self._classes_filter = classes_filter
        self._objects_filter = objects_filter

        self.class_mapping = self.class_handler.class_to_object_mapping if self.args.obj_agnostic_nms else None

    def set_special_classes(self, special_classes: List[int]):
        self.args.classes = list(set(self.args.classes + special_classes))
        self._classes_filter = self.args.classes

    @property
    def class_filter(self):
        return self._classes_filter

    @property
    def objects_filter(self):
        return self._objects_filter

    @property
    def input_size(self):
        return self.args.imgsz

    @property
    def batch_size(self):
        return self.args.batch_size

    def get_mem_details(self):
        return None, None

    def forward(self, images):
        import torch

        with torch.no_grad():
            detection_results = self.model(images)
        return detection_results

    def forward_on_crop_list_static(self, crop_list):
        crops = self.prepare_crops(crop_list)
        norm_factors = [
            np.tile(np.array(cr.shape[1::-1]) / np.array(crops[idx].shape[:0:-1]), 2)
            for idx, cr in enumerate(crop_list)
        ]
        n_batches = int(np.ceil(len(crops) / self.batch_size))
        predictions = []
        for b in range(n_batches):
            batch_crops = crops[b * self.batch_size : (b + 1) * self.batch_size]
            batch_norm_factors = norm_factors[b : (b + 1) * self.batch_size]
            batch_len = len(batch_crops)
            reminder = self.batch_size - batch_len
            if reminder > 0:
                batch_crops += [batch_crops[0]] * reminder
                batch_norm_factors += [batch_norm_factors[0]] * reminder
            batch_stack = (np.stack(batch_crops, axis=0) / 255.0).astype(self.data_type)
            if self.args.is_trt and self.is_local:
                self.model.input_buffer.set(batch_stack)
            raw = self.infer(batch_stack)
            preds = self.post_infer(raw, batch_stack)
            for i, pr in enumerate(preds):
                pr[:, :4] *= batch_norm_factors[i]
            predictions += preds[:batch_len]
        return predictions

    def forward_on_crop_list(self, crop_list):
        crops = self.prepare_crops(crop_list)
        norm_factors = [
            np.tile(np.array(cr.shape[1::-1]) / np.array(crops[idx].shape[:0:-1]), 2)
            for idx, cr in enumerate(crop_list)
        ]
        batch_stack = (np.stack(np.array(crops), axis=0) / 255.0).astype(self.data_type)
        feats = self.infer(batch_stack)
        preds = self.post_infer(feats, batch_stack)
        for i, pr in enumerate(preds):
            pr[:, :4] *= norm_factors[i]
        return preds

    def run_bypassed(self, processed_images, timestamps):
        predictions = []
        for i in range(self.batch_size):
            predictions.append(ImagePredictions(np.zeros((0, 6)), timestamps[i]))
        return DetectionResults(predictions, self._get_classes_names(), np.zeros((0, 6)))

    def run(self, processed_images, timestamps):

        raw = self.infer(processed_images)
        preds = self.post_infer(raw, processed_images)

        predictions = []
        for i in range(self.batch_size):
            pred_with_objects = preds[i]
            if len(pred_with_objects) > 0:
                pred_with_objects = np.hstack((pred_with_objects, pred_with_objects[:, 5][:, None]))
                pred_with_objects[:, 5] = self._mapping[pred_with_objects[:, 5].astype(int)]

            predictions.append(ImagePredictions(pred_with_objects, timestamps[i]))
        return DetectionResults(predictions, self._get_classes_names(), preds)

    @property
    def class_handler(self) -> ClassHandler:
        return self._class_handler

    def _build_transform(self):
        interp_mode = cv2.INTER_AREA if self.args.antialias else cv2.INTER_LINEAR
        self.transform = A.Compose(
            [
                A.Resize(self.args.im_size[0], self.args.im_size[1], interpolation=interp_mode),
            ]
        )

    def _get_classes_names(self) -> List[str]:
        return self.classes_names

    def prepare_images(self, images):
        import torchvision.transforms as transforms
        import torch

        transform = transforms.ToTensor()
        resizer = transforms.Resize(size=self.args.imgsz, antialias=self.args.antialias)
        tensor_images = transform(images)
        tensor_images = resizer(tensor_images)

        batch = torch.unsqueeze(tensor_images, 0)
        if self.args.device == torch.device("cuda"):
            batch = batch.to("cuda")
        if self.args.half:
            batch = batch.half()
        return batch


class DetectorFactory(object):
    _instance = None
    detectors: List[BaseDetector] = []
    use_static_detector: bool = False

    def __new__(cls, use_static_detector: bool = False):
        if cls._instance is None:
            cls._instance = super(DetectorFactory, cls).__new__(cls)
            cls._instance.use_static_detector = use_static_detector
        return cls._instance

    def create_config(self, detector_name: str = "yolov8", detector_args: dict = None) -> BaseDetectorConfig:
        if DetectorType.YOLOV5 in detector_name:
            from .yolov5.yolo_detector import YoloV5Args

            return YoloV5Args(detector_args)
        elif DetectorType.YOLOV8 in detector_name:
            from .yolov5.yolov8_detector import YoloV8Args

            return YoloV8Args(detector_args)
        elif DetectorType.EXPERT in detector_name:
            from .yolov5.yolov8_detector import YoloV8ExpertArgs

            return YoloV8ExpertArgs(detector_args)
        elif DetectorType.WEAPON_EXPERT in detector_name:
            from .yolov5.yolov8_detector import YoloV8WeaponExpertArgs

            return YoloV8WeaponExpertArgs(detector_args)
        elif DetectorType.CONTAINER_OCD in detector_name:
            from .yolov5.yolov8_detector import YoloV8ContainerOcdArgs

            return YoloV8ContainerOcdArgs(detector_args)
        return BaseDetectorConfig(detector_args)

    def create(self, detector_name: str = "yolov5", detector_args: dict = None) -> BaseDetector:
        logger.info(f"Building {detector_name} detector with the arguments : {detector_args}")
        if self.use_static_detector:
            logger.warning("Static mode (shared) Not supported - producing a new detector")
        if DetectorType.YOLOV8 in detector_name and proj.get_host() == "xavier":
            detector_name = DetectorType.YOLOV5
            detector_args["name"] = detector_name
            detector_args.pop("weights", None)
            detector_args.pop("unique_weights", None)
            logger.warning(
                "changing requested detector from yolo v8 to yolo v5 due to Xavier compatibility issue."
                " unique weights were lost"
            )

        if DetectorType.YOLOV5 in detector_name:
            raise ValueError("Yolo v5 is deprecated and can no longer be used")
        elif DetectorType.YOLOV8 in detector_name:
            from .yolov5.yolov8_detector import YoloV8Detector

            return YoloV8Detector(detector_args)
        elif DetectorType.EXPERT in detector_name:
            from .yolov5.yolov8_detector import YoloV8ExpertDetector

            return YoloV8ExpertDetector(detector_args)
        elif DetectorType.WEAPON_EXPERT in detector_name:
            from .yolov5.yolov8_detector import YoloV8WeaponExpertDetector

            return YoloV8WeaponExpertDetector(detector_args)
        elif DetectorType.CONTAINER_OCD in detector_name:
            from .yolov5.yolov8_detector import YoloV8ContainerOcdDetector

            return YoloV8ContainerOcdDetector(detector_args)
        else:
            raise ValueError(f"detector_name must be a supported detector:{DetectorType.list()}")
