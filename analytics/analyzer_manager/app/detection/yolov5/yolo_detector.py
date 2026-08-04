from typing import Dict, Optional

from detection.detector import BaseDetector, BaseDetectorConfig, DetectorType
from .models.common import DetectMultiBackend
from .utils.general import check_img_size, non_max_suppression


class YoloV5Args(BaseDetectorConfig):
    weights: str = "yolov5m-18cls.pt"
    iou_thres = 0.3
    conf_thres = 0.1
    normalize = True
    name = "yolov5"


class YoloV5Detector(BaseDetector):
    _config_type: type = YoloV5Args
    detector_type: DetectorType = DetectorType.YOLOV5

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super().__init__(msg_dict, is_local)

    def _yolo_init(self):
        import torch

        return DetectMultiBackend(
            self.args.weights,
            device=torch.device(self.args.device),
            dnn=self.args.dnn,
            fp16=self.args.half,
        )

    def build_full_model(self):
        import torch

        self.model = self._yolo_init()
        self.args.imgsz = check_img_size(self.args.imgsz, self.stride)  # check image size
        dtype = torch.float16 if self.args.half else torch.float32
        self._image_mem = torch.zeros((self.args.batch_size, 3, *self.args.imgsz), dtype=dtype, device="cuda")
        self.mem_pointer = int(self._image_mem.data_ptr())

    def build_trt_model(self):
        from .yolo_trt import DetectorTrt

        self._internal_build_trt(DetectorTrt)

    def _internal_build_trt(self, trt_engine: type):
        self.model = trt_engine(self.args.weights)
        self._image_mem = self.model.input_buffer
        self.mem_pointer = int(self._image_mem.data)

        self.args.imgsz = self._image_mem.shape[2:]
        self.args.batch_size = self._image_mem.shape[0]
        self.args.half = self.model.is_half

    def post_infer(self, outputs, inputs):
        # NMS
        predictions = self._non_max_suppression(outputs)
        # Process predictions
        for i, det in enumerate(predictions):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = (det[:, :4]).round()
            predictions[i] = predictions[i].to("cpu").numpy()
        return predictions

    def _non_max_suppression(self, non_filtered):
        return non_max_suppression(
            non_filtered,
            self.args.conf_thres,
            self.args.iou_thres,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            mapping=self.class_mapping,
        )

    def infer_full(self, images):
        predictions = self.model(images, augment=False, visualize=False)

        return predictions

    @property
    def memory_buffer(self):
        return self._image_mem

    @property
    def stride(self):
        return self.model.stride
