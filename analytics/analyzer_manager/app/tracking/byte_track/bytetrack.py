from typing import List

from .byte_tracker import BYTETracker
import numpy as np
from general.core import BaseConfig, AnalyticImage
from tracking.tracker import BaseTracker, TrackerType


class BytetrackConfig(BaseConfig):
    track_thresh: float = 0.25
    first_track_compensation: float = 0.1
    baseline_track_compensation: float = 0.05
    use_per_class_thresh: bool = True
    n_init: int = 3
    track_buffer_seconds: float = 3
    match_thresh: float = 0.96
    match_thresh_2: float = 0.84
    match_thresh_3: float = 0.6
    min_box_area = 10
    mot20 = False
    use_timestamp = False
    untrack_objects: List[int] = None

    def __init__(self, msg_dict: dict):
        super(BytetrackConfig, self).__init__(msg_dict)
        self.untrack_objects = [] if self.untrack_objects is None else self.untrack_objects


class Bytetrack(BaseTracker):
    name: str = TrackerType.BYTETRACK.value
    engine_type: type = BYTETracker
    _config_type: type = BytetrackConfig
    args: BytetrackConfig

    def __init__(self, msg_dict: dict, fps: int, context):
        print("Initialize Tracker")
        super(Bytetrack, self).__init__(msg_dict, fps, context)
        self.last_time_stamp = 0

        print("Parsing ByteTrack args")
        bt_args = self._config_type(msg_dict)
        self.unique_detector = msg_dict.get("detector_unique", False)
        self.args = bt_args
        self.track_buffer_sec = bt_args.track_buffer_seconds
        self.filtered_objects = set(bt_args.untrack_objects)
        self.engine = self.engine_type(
            bt_args, frame_rate=fps, filter_objects=self.filtered_objects, context=context
        )  # Load model
        self.on_night_mode_changed(False)  # set defaults for day mode
        self.use_timestamp = bt_args.use_timestamp
        self.context.on_night_mode_changed += self.on_night_mode_changed
        self.set_image_size(self.context.detector_resolution)

    def set_image_size(self, im_size):
        self.engine.set_image_size(im_size)

    def on_night_mode_changed(self, night_mode: bool):
        mode = "night" if night_mode else "day"
        conf_config = self.context.class_handler.confidence_thresholds[mode]
        self._update_confidence(conf_config)

    def _update_confidence(self, conf_config):
        default_value = self.args.track_thresh
        if not self.unique_detector:
            default_value += self.args.baseline_track_compensation
        if self.unique_detector and self.args.use_per_class_thresh:
            conf_thresholds = np.where(conf_config < 0, default_value, conf_config)
        else:
            conf_thresholds = np.zeros_like(conf_config) + default_value
        self.engine.set_confidence(conf_thresholds, default_value)

    def run(self, detector_res, images: List[AnalyticImage]):
        predictions = detector_res.predictions
        frames = []
        last_time_stamp = self.last_time_stamp
        for idx, prediction in enumerate(predictions):  # frames
            dt = (
                (images[idx].timestamp - self.last_time_stamp) / 1000.0 if self.use_timestamp else None
            )  # dt in seconds

            last_time_stamp = images[idx].timestamp
            pred = prediction.data
            im0 = images[idx].processed

            info_imgs = im0.shape[:2]
            detections = self.filter_detections(pred, info_imgs)
            if detections is not None and len(detections):
                img_size = images[idx].frame.shape[:2]
                dets_and_confs = detections[:, :5]  # not uses classes
                classes = detections[:, 5:7]
                det_ids = np.arange(detections.shape[0])
                trk_objs = self.engine.update(
                    dets_and_confs, info_imgs, classes, det_ids, img_size, images[idx].frame, dt
                )
            else:
                trk_objs = self.engine.update_empty(dt)

            dets_and_trks = self.merge_untracked(trk_objs, detections)
            frames.append(dets_and_trks)

        self.last_time_stamp = last_time_stamp
        self.engine.cleanup()

        return frames

    def filter_detections(self, detections, img_size):
        return detections

    def get_confidence_th(self) -> float:
        return self.engine.args.track_thresh

    def merge_ids(self, id_to_keep: int, id_to_merge: int):
        self.engine.replace_ids(id_to_keep, id_to_merge)

    def merge_untracked(self, trkObjs, AllObjs):
        det_ids = np.arange(AllObjs.shape[0])

        # untracked
        output_untrked = []

        for id, det in zip(det_ids, AllObjs):
            if det[5] in self.filtered_objects:
                output_untrked.append(np.array([det[0], det[1], det[2], det[3], -1, det[5], det[6], det[4], id, -1, 0]))
            elif det[4] > self.engine.per_class_confidence_first[int(det[6])]:  # confident
                if len(trkObjs) == 0 or id not in trkObjs[:, 8]:  # non have been tracked or untracked
                    output_untrked.append(
                        np.array([det[0], det[1], det[2], det[3], -1, det[5], det[6], det[4], id, -1, 0])
                    )  # [x,y,x,y,id, cls, subclass, conf, det id, age, conflict]

        # concat
        if len(output_untrked) > 0:
            output = np.stack(output_untrked, axis=0)
            if len(trkObjs) > 0:
                output = np.concatenate([trkObjs, output], axis=0)
            return output
        else:
            return trkObjs
