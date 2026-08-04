import cv2
import torch
from argparse import Namespace
from general.analyzer_general import CROP_GB
from general import proj

config = proj.load_config()

MIN_CLOTHS_SIZE = 0
MIN_CLOTHS_CONF = 0
MAX_CLOTHS_AGE = 70
MIN_CLOTHS_GB = 5


def clip_bboxes(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[0].clamp_(0, shape[1])  # x1
        boxes[1].clamp_(0, shape[0])  # y1
        boxes[2].clamp_(0, shape[1])  # x2
        boxes[3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[::2] = boxes[::2].clip(0, shape[1])  # x1, x2
        boxes[1::2] = boxes[1::2].clip(0, shape[0])  # y1, y2


def scale_bboxes(img1_shape, bboxes, img0_shape):
    # Rescale bboxes (xyxy) from img1_shape to img0_shape
    bboxes = bboxes.astype(float)
    gain = (img1_shape[1] / img0_shape[1], img1_shape[0] / img0_shape[0])  # gain  = old / new

    bboxes[::2] /= gain[0]  # x scaling
    bboxes[1::2] /= gain[1]  # y scaling
    clip_bboxes(bboxes, img0_shape)
    return bboxes


def CropIm(im0, bboxes, im, bgrmap=True, crop_h=CROP_GB, crop_w=CROP_GB):
    bboxes = scale_bboxes(im0.shape, bboxes, im.shape).round()
    y_gb = int((bboxes[3] - bboxes[1]) * crop_h)
    x_gb = int((bboxes[2] - bboxes[0]) * crop_w)

    y1 = int(max((bboxes[1] - y_gb), 0))
    y2 = int(min((bboxes[3] + y_gb), im.shape[0] - 1))
    x1 = int(max((bboxes[0] - x_gb), 0))
    x2 = int(min((bboxes[2] + x_gb), im.shape[1] - 1))
    if bgrmap:
        colorim = im[y1:y2, x1:x2]
    else:
        colorim = cv2.cvtColor(im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

    return colorim


def CropFromPerimeter(im, perimeter, bgrmap=True):
    y1 = int(max(perimeter["topLeft"]["y"] * im.shape[0], 0))
    y2 = int(min(perimeter["bottomRight"]["y"] * im.shape[0], im.shape[0] - 1))
    x1 = int(max(perimeter["topLeft"]["x"] * im.shape[1], 0))
    x2 = int(min(perimeter["bottomRight"]["x"] * im.shape[1], im.shape[1] - 1))
    if bgrmap:
        colorim = im[y1:y2, x1:x2]
    else:
        colorim = cv2.cvtColor(im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

    return colorim


def PoseFrameCheck(im0, bboxes, conf):
    # checking if the frame is good
    confidence = round(float(conf), 2)
    size = int((bboxes[3] - bboxes[1]) * (bboxes[2] - bboxes[0]))

    # boundary test. In case the bounding boxes are at the edge - its not a good candidate to run
    # bboxes[0] = x min
    # bboxes[1] = y min
    # bboxes[2] = x max
    # bboxes[3] = y max
    edge = (
        (bboxes[0] <= MIN_CLOTHS_GB)
        or (bboxes[1] <= MIN_CLOTHS_GB)
        or (bboxes[2] >= im0.shape[1] - MIN_CLOTHS_GB)
        or (bboxes[3] >= im0.shape[0] - MIN_CLOTHS_GB)
    )
    return (size > MIN_CLOTHS_SIZE) and (confidence > MIN_CLOTHS_CONF) and not (edge)


def CapturePerson(id, im0, bboxes, im, device):

    pim = CropIm(im0, bboxes, im, bgrmap=False)
    images_tesor = torch.from_numpy(pim).unsqueeze(0).permute(0, 3, 1, 2)
    images_tesor = images_tesor.float() / 255
    images_tesor = images_tesor.to(device)

    return Namespace(image=images_tesor, id=id)


def NewPersons(detnames, confs, trkoutputs, history, im0, image, device=None):
    new_persons = []

    for det_idx, (output, conf) in enumerate(zip(trkoutputs, confs)):

        id = output[4]
        cls = output[5]
        c = int(cls)  # integer class
        name = detnames[c]
        bboxes = output[0:4].astype(int)
        add_person = False

        # Conditions to run L1:
        # - person class
        # - high confidence
        # - enough pixels
        # - new object without previous info (TBD)

        if (name == "person") and not (id == -1):
            if not (id in history.keys()):
                if PoseFrameCheck(im0, bboxes, conf):
                    add_person = True
                    history[id] = {"iterations": 1, "info": {}, "age": 0, "process": True}
                else:
                    history[id] = {"iterations": 0, "info": {}, "age": 0, "process": False}

            # search on previous frame
            elif not (history[id]["info"]) and (history[id]["iterations"] <= config["analytics"]["cloths"]["trials"]):
                # We didnt completed the number of trials
                if PoseFrameCheck(im0, bboxes, conf):
                    add_person = True
                    history[id]["iterations"] += 1

            if add_person:
                new_persons.append(CapturePerson(id, im0, bboxes, image.frame, device))
    return new_persons


def GetClothsRes(poses_data, trkid):
    info = {}
    pose_data = poses_data[trkid]
    if len(pose_data) > 0:
        if not (pose_data[0].pants_colors == None):
            info["pants"] = {"colors": pose_data[0].pants_colors}  #   list of colors (list of strings- varying length)
        if not (pose_data[0].shirt_colors == None):
            info["shirt"] = {"colors": pose_data[0].shirt_colors}  #   list of colors (list of strings- varying length)
    return info


def ClothUpdate(detnames, trkoutputs, poses_data, history):

    for output in trkoutputs:
        id = output[4]
        cls = output[5]
        c = int(cls)  # integer class
        name = detnames[c]
        if name == "person":
            if id in poses_data:
                history[id]["info"] = GetClothsRes(poses_data, id)

    # updating history ages
    deleted_id = []
    for id in history:
        if history[id]["process"]:
            history[id]["age"] = 0
        else:
            history[id]["age"] += 1
        # reset process
        history[id]["process"] = False
        # delet history above threshold
        if history[id]["age"] > MAX_CLOTHS_AGE:
            deleted_id.append(id)

    for del_id in deleted_id:
        del history[del_id]
    return history


# putting all run model options
def cloths_run(network, image_batch, detRes, trkRes):
    model = network.model
    detPred = detRes.predictions
    detnames = detRes.names
    info_batch = []

    for fi, frame in enumerate(detPred):
        det = frame.data
        timestamp = frame.timestamp
        im0 = frame.im

        image = image_batch[fi]

        confs = det[:, 4]
        info = {}

        # Go over the detections
        if det is not None and len(det):
            trkoutputs = trkRes[fi].trkObjs
            new_persons = NewPersons(
                detnames, confs, trkoutputs, network.history, im0, image, device=model.pose_args.device
            )
            poses_data = model(new_persons)
            network.history = ClothUpdate(detnames, trkoutputs, poses_data, network.history)

            # reset ages

            for output in trkoutputs:
                id = int(output[4])
                if id in network.history:
                    #        info[id] = network.history[id]["info"]
                    network.history[id]["age"] = 0

            for id in poses_data.keys():
                # we have a new update
                if len(poses_data[id]):
                    info[id] = network.history[id]["info"]

        info_batch.append(info)
    return info_batch
