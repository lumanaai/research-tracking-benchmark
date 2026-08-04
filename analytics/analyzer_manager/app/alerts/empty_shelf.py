from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree

from general.analyzer_general import logger
from general.core import BatchDataResolver, AnalyticImage, BDR, AlertInfo
from general.img_utils import union_bboxes  # , plot_frame
from general.offline_analytics import check_expert_availability, OfflineAnalyticsClient, OfflineAnalyticsClientFactory
from .base_alerts import BaseAlert, AlertCandidate, duration_unit_to_sec, duration_unit_to_str


@dataclass
class UserROI:
    id: int
    name: str
    num_vert: int = 0
    polygon: Optional[Polygon] = None
    Hmat: Optional[np.ndarray] = None
    rect_H: Optional[float] = None
    rect_W: Optional[float] = None
    occupancy: float = 1.0
    curr_poly_crop: Optional[np.ndarray] = None
    poly_crop_params: Optional[Dict[str, float]] = None
    is_occl: bool = False

    def __init__(self, _id: int, name: str, zone_val_sel: List[List[float]]):
        pts = [(p["x"], p["y"]) for p in zone_val_sel if "x" in p and "y" in p]
        self.id = _id
        self.name = name
        self.num_vert = len(pts)
        self.polygon = self._safe_polygon(pts, self.num_vert)
        self.occupancy_history = []  # list of (timestamp, occupancy) tuples chromonologically ordered from oldest to newest
        if self.polygon is not None:
            self.Hmat, self.rect_H, self.rect_W = self._determine_roi_perspective(self.polygon, self.num_vert)
        else:
            raise ValueError(f"Invalid polygon at index {_id}: {pts}")
        # self._prepare_crop_params(img_size)

    @staticmethod
    def _safe_polygon(pts: List[List[float]], p_len: int) -> Optional[Polygon]:
        """Create a valid shapely Polygon or return None if invalid/degenerate."""
        if p_len < 3:
            return None
        poly = Polygon(pts)
        if not poly.is_valid or poly.is_empty or poly.area == 0:
            # try to fix common self-intersections
            poly = poly.buffer(0)
            if not poly.is_valid or poly.is_empty or poly.area == 0:
                return None
        return poly

    @staticmethod
    def _quad_side_lengths(poly: Polygon):
        """
        Given a quadrilateral shapely Polygon, return the lengths of its 4 edges
        in clockwise order (based on polygon vertex order).
        """
        # assumes shelves are normal to the ground and have 4 corners
        coords = list(poly.exterior.coords)[:-1]  # drop the repeated last point
        sorted_coord = sorted(coords, key=lambda p: p[0])
        leftmost = sorted_coord[:2]
        rightmost = sorted_coord[2:]
        # quad_xy: four points in image plane, ordered (e.g., TL, TR, BR, BL)
        if leftmost[0][1] < leftmost[1][1]:
            tl, bl = leftmost[0], leftmost[1]
        else:
            bl, tl = leftmost[0], leftmost[1]
        if rightmost[0][1] < rightmost[1][1]:
            tr, br = rightmost[0], rightmost[1]
        else:
            br, tr = rightmost[0], rightmost[1]

        def d(a, b):
            return float(np.hypot(*(b - a)))

        return (
            d(np.array(tl), np.array(bl)),
            d(np.array(tl), np.array(tr)),
            d(np.array(tr), np.array(br)),
            d(np.array(bl), np.array(br)),
            np.array([tl, tr, br, bl], dtype=np.float32),
        )

    def _rectify_iou_with_homography(self, quad_p: Polygon):
        left_side, upper_side, right_side, lower_side, tl2bl_arr = self._quad_side_lengths(quad_p)
        h = max(left_side, right_side)
        w = max(upper_side, lower_side)
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)  # target fronto-parallel rect
        hmat = cv2.getPerspectiveTransform(tl2bl_arr, dst)
        if hmat is None:
            logger.warning("Homography matrix could not be computed.")
            h = w = None
        if h * w < 1e-6:
            logger.warning("Homography matrix is degenerate.")
            return None, None, None
        return hmat, h, w

    def _determine_roi_perspective(self, p, p_len):
        if p_len != 4:
            logger.info(
                "Polygon is not a quadrilateral, skipping perspective rectification and result will be less accurate."
            )
            return None, None, None
        else:
            return self._rectify_iou_with_homography(p)

    def calc_rectified_intersection(self, detection):
        raw_inter = self.polygon.intersection(box(*detection))
        if raw_inter.is_empty or raw_inter.area <= 0 or raw_inter.geom_type != "Polygon":
            return 0.0
        if self.Hmat is None:  # no perspective rectification
            return float(raw_inter.area / self.polygon.area) if raw_inter.area > 0 else 0.0
        else:
            src_pts = np.asarray(list(raw_inter.exterior.coords)[:-1], dtype=np.float32).reshape(-1, 1, 2)
            rectified_inter = cv2.perspectiveTransform(src_pts, self.Hmat.astype(np.float32))
            rectified_inter = Polygon(rectified_inter.reshape(-1, 2))
            return float(rectified_inter.area / (self.rect_H * self.rect_W)) if raw_inter.area > 0 else 0.0

    def _prepare_crop_params(self, img_size: List[int], normalized: bool = True):
        """
        Prepare parameters for smart cropping the polygon from an image of given size.
        Meant to mask out areas outside the polygon while preserving the rectangular crop.
        """
        w, h = img_size
        minx, miny, maxx, maxy = self.polygon.bounds
        if normalized:
            minx, miny, maxx, maxy = minx * w, miny * h, maxx * w, maxy * h
        x0 = max(0, int(np.floor(minx)))
        y0 = max(0, int(np.floor(miny)))
        x1 = min(w, int(np.ceil(maxx)))
        y1 = min(h, int(np.ceil(maxy)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Polygon shape yields an invalid crop coordinates")

        def to_img_coords(coords):
            if normalized:
                return np.array([[c[0] * w, c[1] * h] for c in coords], dtype=np.float32)
            return np.array(coords, dtype=np.float32)

        # pre-allocate mask
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        # exterior
        ext = to_img_coords(list(self.polygon.exterior.coords))
        ext_shifted = (ext - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [ext_shifted], 255)
        # holes (interiors)
        for ring in self.polygon.interiors:
            ints = to_img_coords(list(ring.coords))
            ints_shifted = (ints - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
            cv2.fillPoly(mask, [ints_shifted], 0)
        self.poly_crop_params = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "mask": mask}

    def crop_polygon(self, img: np.ndarray) -> np.ndarray:
        """
        On the fly crop the polygon from the given image using pre-computed parameters.
        """
        # crop the rectangle
        crop = img[
            self.poly_crop_params["y0"] : self.poly_crop_params["y1"],
            self.poly_crop_params["x0"] : self.poly_crop_params["x1"],
            :,
        ].copy()
        # apply mask
        return cv2.bitwise_and(crop, crop, mask=self.poly_crop_params["mask"])

    def export_dict(self) -> Dict:
        out_dict = {
            "shelf_id": self.id,
            "shelf_name": self.name,
            "occupancy": self.occupancy,
            "is_occl": self.is_occl,
            "rect_H": self.rect_H,
            "rect_W": self.rect_W,
        }
        return out_dict


class EmptyShelfAlert(BaseAlert):
    type_name: str = "empty_shelf"
    alert_message: str = "Empty Shelf Detected"
    occlusion_classes = ["person", "vehicle", "shopping_cart"]
    enable_requests: bool = True  # toggle expert requests
    emptyshelf_class: int = 27
    detections_conf: float = 0.1  # confidence threshold for detections
    occupancy_th: float = 0.5
    filled_shelf_th: float = 0.85  # threshold to reset alert when shelf is filled again
    min_intersect_area: float = 0.1  # minimum intersection area for considering a detection
    emptyshelf_per_ms = 30 * 1000  # check-up period: 30 seconds in milliseconds
    emptyshelf_last_per: int = -1
    expert_retries: int = 0
    last_expert_request: int = -1000000  # last expert check timestamp
    alerts_status: Dict[int, Tuple[bool, int]] = {}  # keeps track of alert status and last triggered timestamp
    offline_client: OfflineAnalyticsClient  # expert client
    flexibility: int = 5000  # flexibility parameter for caching expert request
    immediate: bool = False  # whether to request immediate expert response
    request_timeout: int = 10000  # timeout for expert request in milliseconds

    def __init__(self, alert_dict: Dict, context):
        super(EmptyShelfAlert, self).__init__(alert_dict, context)
        self.expert_requests_queue: deque = deque()  # timestamps of pending expert requests
        self.shelves_id = []
        self.set_flow_values()
        if not self.shelves_id:
            raise ValueError("No shelves were defined for empty shelf alert.")

        # check expert availability
        expert_url = check_expert_availability(
            (self.context.get_app_config().get("analytics", {}).get("offlineAnalyticsUri", {}))
        )
        self.offline_client = OfflineAnalyticsClientFactory.get_client(expert_url, self.context.get_camera_id())

        self.occlusion_objs = [
            self.context.get_class_handler().object_str_to_int(cls) for cls in self.occlusion_classes
        ]
        # parse user's ROIs
        self.poly_objs, self.r_tree = self._construct_polygons()
        self.alerts_status = {i: (False, 0) for i in range(len(self.poly_objs))}
        self.alert_ignore_last_per = [0] * len(self.poly_objs)  # last ignored timestamp for each shelf
        self.is_healthy = True
        self.orig_emptyshelf_per_ms = self.emptyshelf_per_ms
        self.oversampling_flag = False  # flag to indicate if oversampling is active
        self.occl_status = [False] * len(self.poly_objs)  # track occlusion status for each shelf
        self.max_timestamp = -1                 # max timestamp of processed expert responses
        self.force_immediate_sampling = False   # flag to force immediate expert request on next batch

    def set_flow_values(self):
        if self.formValue:
            self.occupancy_th = self.formValue.get("threshold", self.occupancy_th)
            if self.occupancy_th > 1.0:
                self.occupancy_th /= 100.0  # convert from percentage to fraction
            self.emptyshelf_per_ms = self.formValue.get("emptyshelf_per_ms", self.emptyshelf_per_ms)
            self.shelves_id = self.formValue.get("shelves")

    def _construct_polygons(self) -> Tuple[List[UserROI], STRtree]:
        # zones = self.selectedCamera.get("zones", {})
        # if not zones:
        #     raise ValueError("No user-defined ROIs were defined for the scene.")
        polygons_lst = []
        poly_geoms = []
        shelves_db = self.context.get_analytics_db().get("shelves", [])
        for i, shelf_id in enumerate(self.shelves_id):
            shelf_entry = next((s for s in shelves_db if s.id == shelf_id), None)
            zone_val = list(shelf_entry.location_dict.get("zones").values())[0]
            # for i, zone_val in enumerate(zones.values()):
            curr_polygon = UserROI(shelf_id, shelf_entry.name, zone_val.get("selection", []))
            polygons_lst.append(curr_polygon)
            poly_geoms.append(curr_polygon.polygon)
        if not poly_geoms:
            raise ValueError("No valid user-defined ROIs found.")
        tree = STRtree(poly_geoms)
        if tree is None:
            raise ValueError("Could not build spatial index for user-defined ROIs.")
        return polygons_lst, tree

    def _map_polygons_to_bboxes(
        self,
        bboxes_xyxy: List[Tuple[float, float, float, float]],
    ) -> Dict[int, List[int]]:
        poly2box_mapping = {}
        for bb_idx, (xmin, ymin, xmax, ymax) in enumerate(bboxes_xyxy):
            rect = box(xmin, ymin, xmax, ymax)
            # quick candidate retrieval via spatial index
            cand_poly_idxs = list(self.r_tree.query(rect))
            cand_poly_geoms = [self.poly_objs[i].polygon for i in cand_poly_idxs]
            if len(cand_poly_idxs) > 0:  # bbox doesn't intersect any polygon
                for poly, pidx in zip(cand_poly_geoms, cand_poly_idxs):
                    inter_area = poly.intersection(rect).area
                    if inter_area > self.min_intersect_area * poly.area:
                        if pidx in poly2box_mapping:
                            poly2box_mapping[pidx].append(bb_idx)
                        else:
                            poly2box_mapping[pidx] = [bb_idx]
        return poly2box_mapping

    def _roi_occupancy(self, bboxes_xyxyn, poly2box_mapping):
        shelves_occupancy = [1.0] * len(self.poly_objs)
        for i, poly in enumerate(self.poly_objs):
            detections_lst = poly2box_mapping.get(i, [])
            if len(detections_lst):
                empty_sum = 0.0
                if len(detections_lst) > 1:  # handle overlapping detections which weren't suppressed before
                    merged_bboxes_xyxyn = union_bboxes(bboxes_xyxyn, detections_lst)
                    for merged_bbox in merged_bboxes_xyxyn:
                        empty_sum += poly.calc_rectified_intersection(merged_bbox)
                else:
                    for det in detections_lst:
                        empty_sum += poly.calc_rectified_intersection(bboxes_xyxyn[det])
                shelves_occupancy[i] = 1 - empty_sum
                if shelves_occupancy[i] < 0:
                    logger.warning(
                        f"Negative occupancy value {shelves_occupancy[i]} for shelf {i}, may indicate overlapping detections. setting to 0."
                    )
                    shelves_occupancy[i] = 0.0
            # update last measured occupancy
            self.poly_objs[i].occupancy = round(shelves_occupancy[i], 1)

    def _determine_alert(self, curr_timestamp):
        """
        Determine if a new alert should be raised based on the current shelf occupancy.
        If alert already raised before, do not raise a new one.
        Reset alert status only when shelf is filled again.
        """
        alerts_out = [False] * len(self.poly_objs)
        for i, poly in enumerate(self.poly_objs):
            if self.alerts_status[i][0] and poly.occupancy > self.filled_shelf_th and not poly.is_occl:
                # reset alert status if shelf is filled again and not occluded
                self.alerts_status[i] = (False, curr_timestamp)
            alerts_out[i] = poly.occupancy < self.occupancy_th
            if alerts_out[i] and not self.alerts_status[i][0]:
                # new alert
                self.alerts_status[i] = (alerts_out[i], curr_timestamp)
            elif alerts_out[i] and self.alerts_status[i][0]:
                # already alerted, cancel repeated alert
                alerts_out[i] = False
        return alerts_out

    def crop_alerted_polygon(self, p_idx: int, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        minx, miny, maxx, maxy = self.poly_objs[p_idx].polygon.bounds  # float
        x0 = max(0, int(np.floor(minx * w)))
        y0 = max(0, int(np.floor(miny * h)))
        x1 = min(w, int(np.ceil(maxx * w)))
        y1 = min(h, int(np.ceil(maxy * h)))
        if x1 <= x0 or y1 <= y0:
            return img[0:0, 0:0, :]  # empty crop
        return img[y0:y1, x0:x1, :]  # RGB crop

    def _occlusion_severity(self, occlusion_detections) -> int:
        """
        Analyzes occlusion detections over all frames in the batch
        and marks polygons as occluded if they are significantly occluded in the optimal frame.
        It returns the index of the optimal frame with the least occluded polygons & minimal occluded area.
        """
        # occlusion_detections := [x,y,x,y,id,cls, subclass, conf, det_id, age, conflict, frame_id, timestamp, [location, center, center location,  area, index]]
        # create a dict with total occlusion for each polygon (values) in each frame (keys)
        num_polys = len(self.poly_objs)
        frame_ids = set(int(det[BDR.FRAME_ID]) for det in occlusion_detections)
        occl_by_frame = {fid: np.zeros(num_polys) for fid in frame_ids}
        poly_areas = [poly.polygon.area for poly in self.poly_objs]
        for det in occlusion_detections:
            frame_id = int(det[BDR.FRAME_ID])
            det_cls = int(det[BDR.CLASS])
            if det_cls in self.occlusion_objs:
                bbox = tuple(det[BDR.POS])
                rect = box(*bbox)
                for pidx, poly in enumerate(self.poly_objs):
                    inter_area = poly.polygon.intersection(rect).area
                    if inter_area > 0:
                        occl_by_frame[frame_id][pidx] += inter_area
        min_count = float("inf")
        min_area = float("inf")
        optimal_frame = None
        for frame_id, areas in occl_by_frame.items():
            count = np.sum(areas >= self.min_intersect_area * np.array(poly_areas))
            total_area = np.sum(areas)
            if count < min_count or (count == min_count and total_area < min_area):
                min_count = count
                min_area = total_area
                optimal_frame = frame_id
        # mark polygons as occluded in optimal frame
        for pidx, inter_area in enumerate(occl_by_frame[optimal_frame]):
            if inter_area >= self.min_intersect_area * poly_areas[pidx]:
                self.poly_objs[pidx].is_occl = True
        return optimal_frame

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        alerts = [False] * len(self.poly_objs)
        alert_candidates = []
        curr_per = images[-1].timestamp // self.emptyshelf_per_ms
        
        # remove outdated expert requests from queue and re-enable requests if all timed out
        while self.expert_requests_queue and images[-1].timestamp - self.expert_requests_queue[0] > self.request_timeout:
            self.expert_requests_queue.popleft()
        # if all pending requests timed out, re-enable new requests
        if not self.enable_requests and len(self.expert_requests_queue) == 0:
            self.enable_requests = True
        
        # when time period had passed, try new request only if requests are enabled
        if (curr_per > self.emptyshelf_last_per or self.force_immediate_sampling) and self.enable_requests:
            # check if there are objects detections that can occlude the shelves
            occlusion_detections = batch_data.query(BatchDataResolver.CLASS, self.occlusion_objs)
            images_with_occlusions = (
                set(occlusion_detections[:, BDR.FRAME_ID].astype(int).tolist()) if len(occlusion_detections) else set()
            )
            # select optimal frame for expert request
            if len(images_with_occlusions) == len(images):
                # all images has occlusion, find occlusion severity
                opt_frame_idx = self._occlusion_severity(occlusion_detections)
            else:
                for idx in range(len(images)):
                    if idx not in images_with_occlusions:
                        opt_frame_idx = idx
                        break
            # send expert request with optimal frame
            success = self.offline_client.send_expert_request(images[opt_frame_idx], flexibility=self.flexibility, immediate=self.immediate)
            self.enable_requests = False
            if success:
                self.force_immediate_sampling = False
                self.expert_retries = 0
                self.emptyshelf_last_per = curr_per
                self.last_expert_request = images[opt_frame_idx].timestamp
                self.expert_requests_queue.append(images[opt_frame_idx].timestamp)
                logger.debug(f"Expert request successfully sent for timestamp: {images[opt_frame_idx].timestamp}.")
                if not self.offline_client.is_healthy:
                    self.context.validate_alert(self.event_id, is_valid=True)
            else:
                self.force_immediate_sampling = True    # force immediate request on next batch
                self.expert_retries += 1
                if self.expert_retries > 3:
                    self.context.validate_alert(
                        self.event_id, is_valid=False, reason="Expert failed to respond in a timely manner."
                    )
                logger.error(f"No answer from expert at {images[opt_frame_idx].timestamp}.")
        
        # process any pending expert responses
        expert_requests_cpy = list(self.expert_requests_queue)
        prev_max_timestamp = self.max_timestamp
        for req_ts in expert_requests_cpy:
            expert_res = self.offline_client.get_expert_detections(req_ts, flexibility=self.flexibility)
            if expert_res is None and not self.offline_client.is_healthy:
                self.context.validate_alert(
                    self.event_id, is_valid=False, reason="Expert failed to respond in a timely manner."
                )
                self.is_healthy = False

            if expert_res is not None:
                valid_dets = [d for d in expert_res if d is not None]
                if not self.is_healthy:
                    self.context.validate_alert(self.event_id, is_valid=True)
                    self.is_healthy = True
                if len(valid_dets) > 0:
                    self.expert_requests_queue.remove(req_ts)
                    self.max_timestamp = max(self.max_timestamp, req_ts)
                    # expert returned results
                    self.enable_requests = True
                    for det in valid_dets:
                        curr_det = det["detections"]
                        if len(curr_det) > 0:
                            # filter detections for empty shelves
                            relevant_detections = [
                                tuple(d[:4])
                                for d in curr_det
                                if d[5] == self.emptyshelf_class and d[4] > self.detections_conf
                            ]
                        else:
                            relevant_detections = []
                            logger.debug(f"No general detections were received from expert at {det['timestamp']}")
                        # handle relevant_detections
                        # process empty shelf detections
                        poly2box_mapping = self._map_polygons_to_bboxes(relevant_detections)

                        # calc current shelves occupancy, could be noisy
                        self._roi_occupancy(relevant_detections, poly2box_mapping)
                        alerts = self._determine_alert(det["timestamp"])

                        # create output alerts
                        for p_idx, alrt in enumerate(alerts):
                            if alrt:
                                poly_crop = self.crop_alerted_polygon(p_idx, images[-1].frame)
                                candidate = self.build_alert_candidate(
                                    det["timestamp"], extra=self.poly_objs[p_idx].export_dict(), crops=[poly_crop]
                                )
                                alert_candidates.append(candidate)
                        # for debug only
                        if type(self) != EmptyShelfCounterAlert:
                            # print(f"----- {type(self)} -----")
                            # print(f"Finished handling expert results with timestamp {req_ts} at timestamp:", {images[-1].timestamp}, f"({round((images[-1].timestamp / 1000), 1)} seconds)")
                            # print("##### Shelf Occupancy #####")
                            # for i, poly in enumerate(self.poly_objs):
                            #     print(f"Shelf {i}: occupancy {poly.occupancy}, alert status {self.alerts_status[i]}")
                            # print(f"##### Output Alerts: {alerts} ######")
                            logger.debug(f"----- {type(self)} -----")
                            logger.debug(
                                f"Finished handling expert results at timestamp:",
                                {images[-1].timestamp},
                                f"({round((images[-1].timestamp / 1000), 1)} seconds)",
                            )
                            logger.debug("##### Shelf Occupancy #####")
                            for i, poly in enumerate(self.poly_objs):
                                logger.debug(f"Shelf {i}: occupancy {poly.occupancy}, alert status {self.alerts_status[i]}")
                            logger.debug(f"##### Output Alerts: {alerts} ######")
        # clean up old requests from queue in case newer requests were processed
        if self.max_timestamp > prev_max_timestamp:
            while self.expert_requests_queue and self.expert_requests_queue[0] < self.max_timestamp:
                self.expert_requests_queue.popleft()
        return alert_candidates

    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = self.build_alert_info_dc(candidate)
        alert_info.crops += candidate.crops
        alert_info.validation_images += candidate.validation_images
        extra_fields = candidate.extra if candidate.extra else {}
        alert_info.extra.update(extra_fields)
        alert_info.alertData = candidate.extra["occupancy"]
        if self.special_filter is not None:
            alert_info.specialFilter = self.special_filter
        alert_info.alertMessage = self.alert_message
        return alert_info


class EmptyShelfCounterAlert(EmptyShelfAlert):
    alert_message: str = "Shelf Occupancy Status Report"
    occupancy_th = 1.1  # always trigger
    report_frequency_ms: int = 30 * 1000  # report every 20 seconds shelf occupancy status if below threshold
    alert_ignore_last_per: List[int]  # last ignored timestamp for each shelf

    def set_flow_values(self):
        self.shelves_id = self.formValue.get("shelves")
        self.report_frequency_ms = self.formValue.get("report_frequency_ms", self.report_frequency_ms)
        self.emptyshelf_per_ms = min(self.emptyshelf_per_ms, self.report_frequency_ms)
        self.alert_ignore_last_per = [0] * len(self.shelves_id)

    def _determine_alert(self, curr_timestamp):
        """
        Determine if a new alert should be raised based on the current shelf occupancy.
        If alert already raised before, do not raise a new one unless ignoring period has passed.
        In the EmptyShelfCounterAlert class alerts are raised to report occupancy.
        """
        alerts_out = [False] * len(self.poly_objs)
        for i, poly in enumerate(self.poly_objs):
            alerts_out[i] = poly.occupancy < self.occupancy_th
            curr_per = (curr_timestamp - self.alerts_status[i][1]) // self.report_frequency_ms
            if alerts_out[i] and not self.alerts_status[i][0]:
                # new alert
                self.alerts_status[i] = (True, curr_timestamp)
            elif curr_per > self.alert_ignore_last_per[i]:
                self.alert_ignore_last_per[i] = curr_per
                self.alerts_status[i] = (alerts_out[i], curr_timestamp)
            elif alerts_out[i] and self.alerts_status[i][0]:
                # cancel repeated alert if ignoring period has not passed
                alerts_out[i] = False
        return alerts_out


class EmptyShelfDropAlert(EmptyShelfAlert):
    emptyshelf_per_ms: int = 4 * 1000  # check-up period: 4 sec in milliseconds
    duration: int = 10 * 1000
    occupancy_th: float = 0.25  # occupancy drop threshold
    flexibility: int = 2000
    immediate: bool = True  # request immediate expert response

    def set_flow_values(self):
        super().set_flow_values()
        duration = self.formValue.get("duration", 0)
        units = self.apply_from_dict("durationUnit", self.formValue, 0)
        self.occupancy_th = self.apply_from_dict("threshold", self.formValue, 0)
        self.occupancy_th /= 100.0  # convert from percentage to fraction
        self.duration = int(duration * duration_unit_to_sec[units] * 1000)  # convert to milliseconds
        self.emptyshelf_per_ms = min(self.emptyshelf_per_ms, self.duration // 2)
        self.flexibility = min(self.flexibility, self.emptyshelf_per_ms // 2)
        self.alert_message = f"shelf occupancy had dropped by {int(self.occupancy_th * 100)}% in less than {duration} {duration_unit_to_str[units]}"

    def _determine_alert(self, curr_timestamp):
        """
        Determine if a new alert should be raised based on the current shelf occupancy.
        If alert already raised before, do not raise a new one.
        Raise alerts only if occupancy drops by a certain threshold within the specified duration.
        Notice: A sample isn't taken into account if the polygon is effectively occluded.
        """
        alerts_out = [False] * len(self.poly_objs)
        for i, poly in enumerate(self.poly_objs):
            # prune old history entries
            while poly.occupancy_history and curr_timestamp - poly.occupancy_history[0][0] > self.duration:
                poly.occupancy_history.pop(0)
            ## in case of occlusion, increase sampling frequency
            self.occl_status[i] = poly.is_occl
            ## check for occupancy drop
            is_noisy = poly.is_occl  # ignore sample if polygon is occluded
            if len(poly.occupancy_history) > 0:
                # determine validity of current sample
                if is_noisy and poly.occupancy <= poly.occupancy_history[-1][1]:
                    is_noisy = False  # do not consider sample as noisy if occupancy decrease
                # check delta occupancy
                if not is_noisy:
                    delta_occupancy = round(max(occ for _, occ in poly.occupancy_history) - poly.occupancy, 3)
                    if delta_occupancy >= self.occupancy_th:
                        if not self.alerts_status[i][0]:  # new alert triggered only once
                            self.alerts_status[i] = (True, curr_timestamp)
                            alerts_out[i] = True
                            while len(poly.occupancy_history) > 1:  # prevent multiple alerts for a continuous drop
                                poly.occupancy_history.pop(0)
                            logger.info(
                                f"Shelf {i} occupancy dropped by {delta_occupancy:.2f} within {self.duration} ms, triggering alert."
                            )
                            # print(        # for debug only
                            #     f"Shelf {i} occupancy dropped by {delta_occupancy:.2f} within {self.duration} ms, triggering alert."
                            # )
                    elif self.alerts_status[i][0]:
                        # reset alert status if occupancy gap returns to normal
                        self.alerts_status[i] = (False, curr_timestamp)
            # add new sample to history
            if not is_noisy:
                poly.occupancy_history.append((curr_timestamp, poly.occupancy))
        ## adjust sampling frequency based on occlusion status
        if not self.oversampling_flag and any(self.occl_status):
            self.emptyshelf_per_ms //= 2  # increase check-up frequency if occluded
            self.oversampling_flag = True
        elif self.oversampling_flag and not any(self.occl_status):
            self.emptyshelf_per_ms = self.orig_emptyshelf_per_ms  # reset to original frequency
            self.oversampling_flag = False
        return alerts_out
