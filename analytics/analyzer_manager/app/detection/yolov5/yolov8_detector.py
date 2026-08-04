import time

import torch
import torchvision
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2xyxy

from .yolo_detector import YoloV5Detector, YoloV5Args
from ..detector import DetectorType


class YoloV8Args(YoloV5Args):
    weights: str = "yolov8s-32cls_1_1.pt"
    iou_thres: float = 0.45
    conf_thres: float = 0.1
    stride: int = 32
    name = "yolov8"
    num_classes: int = 32


def ult_non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    mapping=None,
):
    """taken from ultralytics.utils.ops.non_max_suppression, modifying the mapping argument"""

    # Checks
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6]
        if mapping is not None:
            c = torch.tensor(mapping[c.to("cpu").numpy().astype(int)], device=x.device)
        c *= 0 if agnostic else max_wh  # classes

        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            # logger.warning(f"NMS time limit {time_limit:.3f}s exceeded, indicates poor performance of the system")
            break  # time limit exceeded

    return output


class YoloV8Detector(YoloV5Detector):
    _config_type = YoloV8Args
    detector_type: DetectorType = DetectorType.YOLOV8
    args: YoloV8Args

    def _yolo_init(self):
        import torch

        yolo = YOLO(self.args.weights)

        # warm up
        images = torch.zeros((self.args.batch_size, 3, *self.args.imgsz), device=self.args.device)
        if self.args.half:
            images = images.half()
        yolo.predict(
            images,
            device=self.args.device,
            half=self.args.half,
            conf=self.args.conf_thres,
            iou=self.args.iou_thres,
            classes=self.args.classes,
            agnostic_nms=self.args.agnostic_nms,
            max_det=self.args.max_det,
            imgsz=self.args.imgsz,
            verbose=False,
        )
        return yolo

    def build_triton_model(self):
        from .triton_v8 import YoloV8Triton

        self.model = YoloV8Triton(self.remote_info)
        self.args.imgsz = self.model.im_size
        self.args.half = self.model.is_half

    def build_trt_model(self):
        from .yolo_trt import DetectorTrtV8

        self._internal_build_trt(DetectorTrtV8)

    def infer_full(self, images):
        if isinstance(images, torch.Tensor):
            results = self.model.model(images, augment=False, visualize=False)
        else:
            results = self.model.model(torch.from_numpy(images).to(self.args.device), augment=False, visualize=False)
        return results[0].to("cpu")

    def _non_max_suppression(self, non_filtered):
        return ult_non_max_suppression(
            non_filtered,
            self.args.conf_thres,
            self.args.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            mapping=self.class_mapping,
        )

    @property
    def stride(self):
        return self.args.stride


class YoloV8ExpertArgs(YoloV8Args):
    weights: str = "yolov8m-expert_eff-1_2.pt"
    name = "expert-detect"
    imgsz = [704, 1280]
    batch_size = 1
    conf_thres = 0.01
    max_dynamic_batch = 2


class YoloV8ExpertDetector(YoloV8Detector):
    _config_type = YoloV8ExpertArgs
    detector_type: DetectorType = DetectorType.EXPERT


class YoloV8WeaponExpertArgs(YoloV8Args):
    weights: str = "yolov8m-expert-guns-2_0.pt"
    name = "weapon-expert"
    imgsz = [704, 1280]
    batch_size = 1
    conf_thres = 0.01
    classes_metadata = "guns_expert.csv"
    max_dynamic_batch: int = 2  # max dynamic batch size


class YoloV8WeaponExpertDetector(YoloV8Detector):
    _config_type = YoloV8WeaponExpertArgs
    detector_type: DetectorType = DetectorType.WEAPON_EXPERT


class YoloV8ContainerOcdArgs(YoloV8Args):
    weights: str = "container_ocd_yolov8n_eff_0_0.pt"
    num_classes: int = 1
    name = "container-ocd"
    imgsz = [320, 320]
    conf_thres = 0.25
    max_dynamic_batch = 2


class YoloV8ContainerOcdDetector(YoloV8Detector):
    _config_type = YoloV8ContainerOcdArgs
    detector_type: DetectorType = DetectorType.CONTAINER_OCD

    def _load_classes(self):
        self.n_classes = 1
        self.classes_names = ["serial"]
        self.object_names = ["container_part"]
        self._mapping = {}
        self._class_handler = None
        self.class_mapping = None
        self.args.classes = [0]
