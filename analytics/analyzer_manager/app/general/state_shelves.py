from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree

from general.analyzer_general import logger
from general.core import AnalyticImage, BDR, MotionData, ObjectManager
from general.img_utils import union_bboxes  # , plot_frame
from general.offline_analytics import check_expert_availability, OfflineAnalyticsClient, OfflineAnalyticsClientFactory


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
    last_update_time: int = -1
    last_itr_occupancies: List[Tuple[int, float]] = field(
        default_factory=list
    )  # list of (timestamp, occupancy) tuples chronologically ordered from oldest to newest

    def __init__(self, _id: int, name: str, zone_val_sel: List[List[float]]):
        pts = [(p["x"], p["y"]) for p in zone_val_sel if "x" in p and "y" in p]
        self.id = _id
        self.name = name
        self.num_vert = len(pts)
        self.polygon = self._safe_polygon(pts, self.num_vert)
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
            "last_update_time": self.last_update_time,
        }
        return out_dict


class StaticShelvesManager(ObjectManager):
    occlusion_classes = ["person", "vehicle", "shopping_cart"]
    enable_requests: bool = True  # toggle expert requests
    emptyshelf_class: int = 27
    detections_conf: float = 0.1  # confidence threshold for detections
    min_intersect_area: float = 0.1  # minimum intersection area for considering a detection
    emptyshelf_per_ms = 30 * 1000  # check-up period: 30 seconds in milliseconds
    emptyshelf_last_per: int = -1
    expert_retries: int = 0
    offline_client: OfflineAnalyticsClient  # expert client
    flexibility: int = 5000  # flexibility parameter for caching expert request
    immediate: bool = False  # whether to request immediate expert response
    request_timeout: int = 10000  # timeout for expert request in milliseconds

    def __init__(self, context):
        self.context = context
        self.num_shelves = 0
        self.shelves_id = []
        self.occl_status = (
            {}
        )  # occlusion status for each shelf                                                  # list of shelf IDs (populated by on_db_update)
        self.expert_requests_queue: deque = deque(maxlen=20)  # timestamps of pending expert requests
        # check expert availability
        expert_url = check_expert_availability((context.app_config.get("analytics", {}).get("offlineAnalyticsUri", {})))
        self.offline_client = OfflineAnalyticsClientFactory.get_client(expert_url, context.camera_id)
        self.occlusion_objs = [context.class_handler.object_str_to_int(cls) for cls in self.occlusion_classes]
        # parse user's ROIs
        self.orig_emptyshelf_per_ms = self.emptyshelf_per_ms
        self.oversampling_flag = False  # flag to indicate if oversampling is active
        self.max_timestamp = -1  # max timestamp of processed expert responses
        self.force_immediate_sampling = False  # flag to force immediate expert request on next batch
        # self.new_results_available = False                                    # flag to indicate new expert results are available
        self.last_report = {}

    def on_db_update(self) -> Tuple[List[UserROI], STRtree]:
        # Get shelves from DB directly
        try:
            shelves_db = self.context.analytic_db.get("shelves", [])
            # Reset state
            self.shelves_id = []
            self.occl_status = {}
            self.num_shelves = 0

            for s in shelves_db:
                self.shelves_id.append(s.id)
                self.occl_status[s.id] = False
                self.num_shelves += 1
            if not self.shelves_id:
                logger.error("No shelves were defined in database - shelves manager will be disabled.")
                return ([], None)
            self.last_report = {
                sid: {"value": None} for sid in self.shelves_id
            }  # last report for dashboard aggregation
            self.poly_objs_dict = {}
            self.poly_objs = []
            poly_geoms = []
            for shelf_id in self.shelves_id:
                shelf_entry = next((s for s in shelves_db if s.id == shelf_id), None)
                zone_val = list(shelf_entry.location_dict.get("zones").values())[0]
                curr_polygon = UserROI(shelf_id, shelf_entry.name, zone_val.get("selection", []))
                self.poly_objs.append(curr_polygon)
                self.poly_objs_dict[shelf_id] = curr_polygon
                poly_geoms.append(curr_polygon.polygon)
            if not poly_geoms:
                logger.error("No valid user-defined ROIs found - shelves manager will be disabled.")
                return ([], None)
            self.r_tree = STRtree(poly_geoms)
            if self.r_tree is None:
                logger.error("Could not build spatial index for user-defined ROIs - shelves manager will be disabled.")
                return ([], None)
        except Exception as e:
            logger.error(f"Error processing shelves from DB: {e}")
            raise Exception(f"Error processing shelves from DB: {e}")

    def _map_polygons_to_bboxes(
        self,
        bboxes_xyxy: List[Tuple[float, float, float, float]],
    ) -> Dict[int, List[int]]:
        poly2box_mapping = {}
        for bb_idx, (xmin, ymin, xmax, ymax) in enumerate(bboxes_xyxy):
            rect = box(xmin, ymin, xmax, ymax)
            # quick candidate retrieval via spatial index
            cand_poly_idxs = list(self.r_tree.query(rect))
            cand_shelf_ids = [self.shelves_id[i] for i in cand_poly_idxs]
            cand_poly_geoms = [self.poly_objs_dict[sid].polygon for sid in cand_shelf_ids]
            if len(cand_poly_idxs) > 0:  # bbox doesn't intersect any polygon
                for poly, shelf_id in zip(cand_poly_geoms, cand_shelf_ids):
                    inter_area = poly.intersection(rect).area
                    if inter_area > self.min_intersect_area * poly.area:
                        if shelf_id in poly2box_mapping:
                            poly2box_mapping[shelf_id].append(bb_idx)
                        else:
                            poly2box_mapping[shelf_id] = [bb_idx]
        return poly2box_mapping

    def _roi_occupancy(self, bboxes_xyxyn, poly2box_mapping, update_ts):
        shelves_occupancy = {sid: 1.0 for sid in self.shelves_id}
        for shelf_id in self.shelves_id:
            poly = self.poly_objs_dict[shelf_id]
            detections_lst = poly2box_mapping.get(shelf_id, [])
            if len(detections_lst):
                empty_sum = 0.0
                if len(detections_lst) > 1:
                    merged_bboxes_xyxyn = union_bboxes(bboxes_xyxyn, detections_lst)
                    for merged_bbox in merged_bboxes_xyxyn:
                        empty_sum += poly.calc_rectified_intersection(merged_bbox)
                else:
                    for det in detections_lst:
                        empty_sum += poly.calc_rectified_intersection(bboxes_xyxyn[det])
                shelves_occupancy[shelf_id] = 1 - empty_sum
                if shelves_occupancy[shelf_id] < 0:
                    logger.warning(
                        f"Negative occupancy value {shelves_occupancy[shelf_id]} for shelf {shelf_id}, may indicate overlapping detections. setting to 0."
                    )
                    shelves_occupancy[shelf_id] = 0.0
            # update last measured occupancy
            self.poly_objs_dict[shelf_id].occupancy = round(shelves_occupancy[shelf_id], 1)
            self.poly_objs_dict[shelf_id].last_update_time = update_ts

    def _occlusion_severity(self, occlusion_detections) -> int:
        """
        Analyzes occlusion detections over all frames in the batch
        and marks polygons as occluded if they are significantly occluded in the optimal frame.
        It returns the index of the optimal frame with the least occluded polygons & minimal occluded area.
        """
        # occlusion_detections := [x,y,x,y,id,cls, subclass, conf, det_id, age, conflict, frame_id, timestamp, [location, center, center location,  area, index]]
        # create a dict with total occlusion for each polygon (values) in each frame (keys)
        frame_ids = set(int(det[BDR.FRAME_ID]) for det in occlusion_detections)
        occl_by_frame = {fid: {sid: 0.0 for sid in self.shelves_id} for fid in frame_ids}
        poly_areas = {sid: self.poly_objs_dict[sid].polygon.area for sid in self.shelves_id}
        for det in occlusion_detections:
            frame_id = int(det[BDR.FRAME_ID])
            det_cls = int(det[BDR.CLASS])
            if det_cls in self.occlusion_objs:
                bbox = tuple(det[BDR.POS])
                rect = box(*bbox)
                for shelf_id in self.shelves_id:
                    poly = self.poly_objs_dict[shelf_id]
                    inter_area = poly.polygon.intersection(rect).area
                    if inter_area > 0:
                        occl_by_frame[frame_id][shelf_id] += inter_area
        min_count = float("inf")
        min_area = float("inf")
        optimal_frame = None
        for frame_id, areas_dict in occl_by_frame.items():
            areas = np.array([areas_dict[sid] for sid in self.shelves_id])
            poly_areas_arr = np.array([poly_areas[sid] for sid in self.shelves_id])
            count = np.sum(areas >= self.min_intersect_area * poly_areas_arr)
            total_area = np.sum(areas)
            if count < min_count or (count == min_count and total_area < min_area):
                min_count = count
                min_area = total_area
                optimal_frame = frame_id
        # mark polygons as occluded in optimal frame
        for shelf_id in self.shelves_id:
            inter_area = occl_by_frame[optimal_frame][shelf_id]
            if inter_area >= self.min_intersect_area * poly_areas[shelf_id]:
                self.poly_objs_dict[shelf_id].is_occl = True
        return optimal_frame

    def _cleanup_expired_requests(self, curr_ts: int):
        """Remove timed-out requests from queue and re-enable if all expired."""
        # remove outdated expert requests from queue and re-enable requests if all timed out
        while self.expert_requests_queue and curr_ts - self.expert_requests_queue[0] > self.request_timeout:
            self.expert_requests_queue.popleft()
        # if all pending requests timed out, re-enable new requests
        if not self.enable_requests and len(self.expert_requests_queue) == 0:
            self.enable_requests = True

    def _accumulate_occupancy_history(self, det_ts: int):
        """Update occupancy history for each shelf, handling occlusion noise."""
        for shelf_id in self.shelves_id:
            poly = self.poly_objs_dict[shelf_id]
            ## in case of occlusion, increase sampling frequency and ignore sample
            self.occl_status[shelf_id] = is_noisy = poly.is_occl
            if is_noisy and len(poly.last_itr_occupancies) > 0:
                # check if occupancy is increasing compared to last sample
                if poly.occupancy <= poly.last_itr_occupancies[-1][1]:
                    is_noisy = False  # do not consider sample as noisy if occupancy decreases
            if not is_noisy:
                self.poly_objs_dict[shelf_id].last_itr_occupancies.append((det_ts, poly.occupancy))

    def _adjust_sampling_frequency(self):
        """Adjust sampling frequency based on occlusion status of shelves."""
        any_occluded = any(self.occl_status.values())
        if not self.oversampling_flag and any_occluded:
            self.emptyshelf_per_ms //= 2  # increase check-up frequency if occluded
            self.oversampling_flag = True
        elif self.oversampling_flag and not any_occluded:
            self.emptyshelf_per_ms = self.orig_emptyshelf_per_ms  # reset to original frequency
            self.oversampling_flag = False

    def track(self, batch_data: BDR, image_batch: List[AnalyticImage], motion_data: MotionData):

        curr_per = image_batch[-1].timestamp // self.emptyshelf_per_ms
        self._cleanup_expired_requests(image_batch[-1].timestamp)
        # when time period had passed, try new request only if requests are enabled
        if (curr_per > self.emptyshelf_last_per or self.force_immediate_sampling) and self.enable_requests:
            # check if there are objects detections that can occlude the shelves
            occlusion_detections = batch_data.query(BDR.CLASS, self.occlusion_objs)
            images_with_occlusions = (
                set(occlusion_detections[:, BDR.FRAME_ID].astype(int).tolist()) if len(occlusion_detections) else set()
            )
            # select optimal frame for expert request
            if len(images_with_occlusions) == len(image_batch):
                # all images has occlusion, find occlusion severity
                opt_frame_idx = self._occlusion_severity(occlusion_detections)
            else:
                for idx in range(len(image_batch)):
                    if idx not in images_with_occlusions:
                        opt_frame_idx = idx
                        break
            # send expert request with optimal frame
            success = self.offline_client.send_expert_request(
                image_batch[opt_frame_idx], flexibility=self.flexibility, immediate=self.immediate
            )
            self.enable_requests = False
            if success:
                self.force_immediate_sampling = False
                self.expert_retries = 0
                self.emptyshelf_last_per = curr_per
                self.expert_requests_queue.append(image_batch[opt_frame_idx].timestamp)
                logger.debug(f"Expert request successfully sent for timestamp: {image_batch[opt_frame_idx].timestamp}.")
            else:
                self.force_immediate_sampling = True  # force immediate request on next batch
                self.expert_retries += 1
                logger.error(
                    f"No answer from expert at {image_batch[opt_frame_idx].timestamp}. Retry count: {self.expert_retries}."
                )

        # clean previous iteration occupancies
        for shelf_id in self.shelves_id:
            self.poly_objs_dict[shelf_id].last_itr_occupancies = []

        # process any pending expert responses
        expert_requests_cpy = list(self.expert_requests_queue)
        prev_max_timestamp = self.max_timestamp
        for req_ts in expert_requests_cpy:
            expert_res = self.offline_client.get_expert_detections(req_ts, flexibility=self.flexibility)
            if expert_res is not None:
                valid_dets = [d for d in expert_res if d is not None]
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
                        self._roi_occupancy(relevant_detections, poly2box_mapping, det["timestamp"])
                        # accumulate occupancy history
                        self._accumulate_occupancy_history(det["timestamp"])

        # clean up old requests from queue in case newer requests were processed
        if self.max_timestamp > prev_max_timestamp:
            while self.expert_requests_queue and self.expert_requests_queue[0] < self.max_timestamp:
                self.expert_requests_queue.popleft()

        ## adjust sampling frequency based on occlusion status
        self._adjust_sampling_frequency()

    def sync_and_cleanup(self, base: int) -> Dict:
        out_dict = {}
        for shelf_id in self.shelves_id:
            upd = {
                "value": int(self.poly_objs_dict[shelf_id].occupancy * 100),  # convert to percentage
                "in": 0,
                "out": 0,
            }
            if self.last_report[shelf_id]["value"]:
                delta = upd["value"] - self.last_report[shelf_id]["value"]
                if delta > 0:
                    upd["in"] = delta
                else:
                    upd["out"] = abs(delta)
            out_dict[shelf_id] = upd

        self.last_report = out_dict
        shelves_info = {
            "timestamp": base,
            "shelves": [{"shelfId": str(sid), "data": {**sdata}} for sid, sdata in out_dict.items()],
        }
        return {"shelves": shelves_info}

    @property
    def enabled(self) -> bool:
        return len(self.shelves_id) > 0
