from typing import List, Dict, Set

from general.core import AnalyticImage
from .byte_sreid_tracker import BYTESReidTracker
import numpy as np
from tracking.tracker import TrackerType
from tracking.byte_track.bytetrack import Bytetrack, BytetrackConfig
from ..byte_track.utils import update_size_filter


class BytetrackSReidConfig(BytetrackConfig):

    n_init = 3
    track_buffer_seconds = 4  #   keep "tracking" with kalman for this amount of time after detection is lost [3]
    removed_buffer_seconds = 30  #   how much time to keep ids in removed list before removing them (for reid)

    prev_track_thresh = 0.25
    prev_first_track_thresh = 0.35
    prev_match_thresh = 0.96
    prev_match_thresh_2 = 0.84
    prev_match_thresh_3 = 0.6

    ciou_track_thresh = 0.18
    ciou_first_track_thresh = 0.29
    ciou_match_thresh = 1.05
    ciou_match_thresh_2 = 0.99
    ciou_match_thresh_3 = 0.9

    use_reid_conflict_solver = False  #   solve conflcits
    reid_max_batch_size = 8  #   max batch size for reid networks, the reid modl itself actually does this too
    use_ciou = False
    reid_freq_seconds = 2  #   how often to push crop to memory
    maintain_id_cost_disscount = 0.15  #   disscount in cost to maintain same id in conflict solver
    max_reid_cost = 1  # max reid cost matrix
    conflict_classes = [0, 1]  #   classes to solve conflicts for

    #   if an object has a ration of this to the max size object (90% max) the object is considered to be "small" when initiated,  and "old" object (hysteresis)
    min_max_size_ratio_start = 0.15
    min_max_size_ratio_continue = 0.2
    max_size_filtered_subclasses = [2, 3, 4, 6]  #   classes to filter by max size

    #   From now on '0' is for person, '1' is for vehicle
    conflict_iou_thresh_dict = {0: {"start": 0.2, "continue": 0.15}, 1: {"start": 0.075, "continue": 0.025}}

    max_ass_cost_thresh_dict = {  #   max cost to associate
        0: np.inf,
        # 1: 0.35
        1: 0.65,
    }

    # Person and vehicle reid arguments
    reid_args = {
        0: {"enable": True},
        1: {"enable": True},
    }

    #  memory banks arguments, maximal number of ids and crops per id to keep in memory
    reid_memory_banks_args = {
        0: {
            "num_max_ids": 15,
            "num_max_crops": 4,
        },
        1: {
            "num_max_ids": 15,
            "num_max_crops": 4,
        },
    }
    cls_max_size = None

    def __init__(self, msg_dict: dict):
        super().__init__(msg_dict)
        if self.cls_max_size is None:
            self.cls_max_size = {}
        self.max_size_filtered_subclasses = set(self.max_size_filtered_subclasses)


class ByteSReidTrack(Bytetrack):
    name: str = TrackerType.BYTESREID.value
    engine_type = BYTESReidTracker
    _config_type: type = BytetrackSReidConfig
    class_max_filter: Dict[int, np.array]
    filtered_subclasses: Set[int]
    args: BytetrackSReidConfig
    engine: BYTESReidTracker

    def __init__(self, msg_dict: dict, fps: int, context):
        super().__init__(msg_dict, fps, context)
        self.on_entities_stats_update({})

    def filter_detections(self, detections, img_size):

        if not self.filtered_subclasses or len(detections) == 0:
            return detections

        subclasses = detections[:, 6].astype(int)
        bboxes = np.column_stack(
            (
                (detections[:, 3] - detections[:, 1]) / img_size[0],  # Heights
                (detections[:, 2] - detections[:, 0]) / img_size[1],  # Widths
            )
        )
        masks = np.ones(len(detections), dtype=bool)
        idxs = [i for i in range(len(subclasses)) if subclasses[i] in self.filtered_subclasses]
        for i in idxs:
            if np.any(bboxes[i] < self.class_max_filter[subclasses[i]]):
                masks[i] = False
        return detections[masks]

    def on_entities_stats_update(self, stats: Dict):
        if "cls_max_size" in stats:
            self.args.cls_max_size = stats["cls_max_size"]
        self.engine.update_max_size_filter()
        self.class_max_filter, sub_cls = update_size_filter(
            self.args.cls_max_size, self.args.min_max_size_ratio_continue
        )
        self.filtered_subclasses = self.args.max_size_filtered_subclasses.intersection(sub_cls)


