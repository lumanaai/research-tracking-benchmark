from collections import OrderedDict
from enum import Enum
from io import BytesIO
from typing import List, Any, Dict, Optional, Tuple

import numpy as np
import requests

from detection.yolov5.yolov8_detector import YoloV8ContainerOcdDetector
from general.analyzer_general import logger, GLOBAL_TYPE_KEY, GLOBAL_SUCCESS, ROI_SHAPE
from general.core import (
    AdvanceAnalyzerType,
    AdvanceAnalyticResults,
    WorkItem,
    AttrConfidence,
    AttrProperty,
    BatchDataResolver,
    BDR,
    AnalyticImage,
    MotionData,
)
from general.img_utils import crop_image, concatenate_images
from level1.advanced import BaseAdvanceAnalyzerConfig, BaseAdvanceAnalyzer, ExtApiConfig, ExtApiAnalyzer, AttrData
from level1.shipping_container.bic_classifier import (
    BicField,
    BicMatchType,
    EntityFieldAccumulator,
    FieldObservation,
    classify_text,
    canonicalize_text,
    validate_check_digit,
)
from level1.shipping_container.ocr_models import ContainerOcr
from alerts.alerts_utils import WLSimilarity


def api_to_bbox(api_bbox: Dict) -> List:
    if not api_bbox or "xmin" not in api_bbox:
        return None
    return [api_bbox["xmin"], api_bbox["ymin"], api_bbox["xmax"], api_bbox["ymax"]]


# ═══════════════════════════════ Config Hierarchy ═════════════════════════════


class ContainerBaseConfig(BaseAdvanceAnalyzerConfig):
    """Shared configuration for both external and internal container analyzers."""
    guard_band_x: float = 0.015
    guard_band_y: float = 0.015
    min_crop_size: List[int] = [240, 200]
    revert_to_trucks: bool = False
    min_sec_between_calls: int = 10
    max_ocd_size = 50 * 100  # pixels
    min_ocd_conf: float = 0.35
    motion_sensitivity: float = 10
    max_candidates_same_place: int = 5


class ExternalContainerConfig(ContainerBaseConfig, ExtApiConfig):
    """Configuration for external API container analyzer."""
    api_token: str = "3dec628a0ed8dd1bf83aaa046a8c7487060373f6"
    api_url: str = "https://container.lumana.ai/api/v1/predict/"
    min_attribute_conf: float = 0.6


class InternalContainerConfig(ContainerBaseConfig):
    """Configuration for internal OCR container analyzer."""
    min_sec_between_calls: int = 0  # no rate limit for internal analyzer
    # OCR acceptance
    internal_min_ocr_conf: float = 0.6
    # Spatial clustering
    internal_consensus_similarity_th: float = 0.95      # similarity threshold for deduplicating anchors and blocking serials


class ContainerAttributes(str, Enum):
    SIZE_CODE = "sizeCode"
    OWNER_CODE = "ownerCode"
    SERIAL_NUMBER = "serialNumber"


# ═══════════════════════════════ Mixin (Shared Logic) ═════════════════════════


class ContainerAnalyzerMixin:
    """
    Mixin providing shared container analyzer functionality for both backends.
    Must be used with BaseAdvanceAnalyzer (or subclass) as the other parent.

    Provides: __init__ setup, track(), validate_candidate(), calc_pre_score(),
    to_properties(), merge_with_hist(), calculate_post_score(), _calc_area_score().
    """

    crop_on_success_only = True
    name = "containerAnalyzer"
    _type = AdvanceAnalyzerType.CONTAINER
    use_low_prio_set = False
    max_retries = 15
    max_required_retries = 15
    low_prio_max_time = 5
    analyzer_batch_limit = 1
    min_require_results: AdvanceAnalyticResults = AdvanceAnalyticResults.SUCCESS
    post_score_failure_th: float = 0.0
    occlusion_threshold: float = 1

    def _init_container(self, msg: dict, context, instance_id: str = None):
        """Initialize shared container state. Call from __init__ after super().__init__."""
        if self.class_handler.container_value < 0 and self.args.revert_to_trucks:
            self.supported_classes = [self.class_handler.vehicle_value]
            self.supported_subclasses = [self.class_handler.class_str_to_int("truck")]
        elif self.class_handler.container_value >= 0:
            self.supported_classes = [self.class_handler.container_value]
            self.supported_subclasses = self.class_handler.get_object_classes(self.class_handler.container_value)
        else:
            self.enabled = False
            self.supported_classes = []
            self.supported_subclasses = []
            logger.error("shipping container analyzer is disabled due to lack of support in container class")

        self.container_label_map = {
            "Size and Type Codes": ContainerAttributes.SIZE_CODE.value,
            "Owner Code and Category Identifier": ContainerAttributes.OWNER_CODE.value,
            "Serial Number": ContainerAttributes.SERIAL_NUMBER.value,
        }
        self.guard_band = np.array(
            [self.args.guard_band_x, self.args.guard_band_y, 1 - self.args.guard_band_x, 1 - self.args.guard_band_y]
        )
        self.min_crop_size = np.array(self.args.min_crop_size)
        self.norm_crop_size = self.min_crop_size * 2
        self.pre_detector = YoloV8ContainerOcdDetector({})

    def merge_with_hist(self, attr: Dict, obj_id: int) -> (Dict, float):
        """Return latest post_score (not max) so current multi-cluster assessment dominates."""
        return attr, self.processed[obj_id][-1].post_score

    def calculate_post_score(self, obj_id: int, attr_data: AttrData) -> float:
        """Score = min(cluster_scores). Entity SUCCESS only when ALL clusters validated.

        Uses _cluster_scores embedded in result by _build_result.
        Falls back to max field score for external analyzer compatibility.
        """
        cluster_scores = attr_data.attributes.pop("_cluster_scores", None)
        if cluster_scores:
            return min(cluster_scores)
        # Fallback for external analyzer or empty results
        max_scores = {"default": 0}
        for key in self.container_label_map.values():
            if key in attr_data.attributes:
                max_scores[key] = max([res["score"] for res in attr_data.attributes[key]])
        return max(max_scores.values())

    def track(self, image_batch, batch_data: BatchDataResolver, motion_data: MotionData):
        relevant_batch_ids = batch_data.unique(batch_data.ID, batch_data.SUBCLASS, self.supported_subclasses).astype(
            int
        )
        if self.enabled and len(relevant_batch_ids) > 0:
            score_table = self.calculate_pre_score(batch_data, image_batch)
            for obj_id in relevant_batch_ids:
                obj_id = int(obj_id)
                if obj_id < 0:
                    continue
                elif obj_id not in self.processed and obj_id not in self.candidates:
                    self.update_candidates(obj_id, score_table, batch_data, image_batch)
                else:
                    obj_matches = [
                        i for i in score_table.keys() if batch_data.data[i, BDR.ID] == obj_id and score_table[i] > 0
                    ]
                    if obj_matches:
                        obj_data = batch_data.data[obj_matches, :]
                        coords = np.zeros(4)
                        coords[:2] = np.floor(np.min(obj_data[:, :2] * ROI_SHAPE, axis=0))
                        coords[2:] = np.ceil(np.max(obj_data[:, 2:4] * ROI_SHAPE, axis=0))

                        sliced = np.clip(coords, 0, ROI_SHAPE[0]).astype(int)
                        has_motion = np.any(
                            motion_data.unified[sliced[1] : sliced[3], sliced[0] : sliced[2]]
                            >= self.args.motion_sensitivity
                        )
                        if has_motion:
                            self.update_candidates(obj_id, score_table, batch_data, image_batch)

    def to_properties(self, attr_data: Dict) -> Dict:
        prop_data = {}
        for key in attr_data:
            if key in self.container_label_map.values():
                prop_data[key] = [
                    AttrProperty(res["value"], res["score"], AttrConfidence.HIGH, self.priority)
                    for res in attr_data[key]
                ]
        if len(prop_data) > 0:
            prop_data[GLOBAL_TYPE_KEY] = [AttrProperty(GLOBAL_SUCCESS, 1, AttrConfidence.HIGH, self.priority)]

        self._add_metrics_to_properties(attr_data, prop_data)
        return prop_data

    def _calc_area_score(self, crop_sz, sub_class: int = -1):
        norm_size = np.clip((np.array(crop_sz) - self.min_crop_size) / self.norm_crop_size, 0, 1)
        return norm_size[0] * norm_size[1]

    def validate_candidate(
        self, best_index: int, batch_data: BDR, image_batch: List[AnalyticImage]
    ) -> Tuple[bool, Optional[float]]:
        frame_ind = int(batch_data.data[best_index, BDR.FRAME_ID])
        crop = crop_image(
            image_batch[frame_ind].frame, batch_data.data[best_index, BDR.POS], margins=[0, 0], bgr_map=False
        )
        detections = self.pre_detector.forward_on_crop_list([crop])[0]
        score = self.calc_pre_score(detections, crop)
        return score > 0.1, score

    def calc_pre_score(self, detections, crop) -> float:
        if len(detections) == 0:
            return 0
        h, w = crop.shape[:2]
        gb_lims = self.guard_band * np.array([w, h, w, h])
        in_gb = np.all((detections[:, :2] >= gb_lims[:2]) & (detections[:, 2:4] <= gb_lims[2:]), axis=1)
        conf_valid = detections[:, 4] > self.args.min_ocd_conf
        scores = [
            self.calc_area(det) / self.args.max_ocd_size if (conf_valid[i] and in_gb[i]) else 0
            for i, det in enumerate(detections)
        ]
        return np.max(scores)


# ═══════════════════════════════ External Analyzer ════════════════════════════


class ExternalContainerAnalyzer(ContainerAnalyzerMixin, ExtApiAnalyzer):
    """Container analyzer using external OCR API (existing production flow)."""

    _config_type = ExternalContainerConfig
    args: ExternalContainerConfig

    def __init__(self, msg: dict, context, instance_id: str = None):
        super().__init__(msg, context, instance_id)
        self._init_container(msg, context, instance_id)
        self.data_template = {"detection_mode": "vehicle", "mode": "fast", "regions": [], "camera_id": self.instance_id}

    def _run_api(self, jpeg_bytes: BytesIO) -> Any:
        response = requests.post(
            self.args.api_url,
            files={"image": ("example.jpg", jpeg_bytes, "image/jpeg")},
            headers={"Authorization": "Token " + self.args.api_token},
            timeout=self.args.timeout_sec,
            data=self.data_template,
        )
        if response is None:
            logger.error("ALPR: No response from API")
            return None
        elif response.status_code == 429:
            logger.error("ALPR: Max calls per second reached")
            return None
        elif response.status_code < 200 or response.status_code > 300:
            logger.error(f"Container API call failed. response: {response.text}")
            return None
        return response.json(object_pairs_hook=OrderedDict)

    def _parse_api_results(self, api_res: Dict, work_item: WorkItem) -> Dict:
        results = api_res.get("results", [])
        containers = {cat: [] for cat in self.container_label_map.values()}
        zooms = []
        for res in results:
            try:
                txt = res["texts"][0]["value"]
                txt_score = res["texts"][0]["score"]
                cat = res["object"]["label"]
                cat_score = res["object"]["score"]
                cat_bbox = api_to_bbox(res["object"]["value"])
                score = cat_score * txt_score
                if cat in self.container_label_map and score > self.args.min_attribute_conf:
                    cat_parsed = self.container_label_map[cat]
                    containers[cat_parsed].append({"value": txt, "score": score, "bbox": cat_bbox})
                    if cat_parsed == ContainerAttributes.SERIAL_NUMBER.value:
                        zooms.append(crop_image(work_item.image, cat_bbox))
            except Exception as e:
                logger.warning(f"Container API parsing failed: {e}")

        containers = {k: v for k, v in containers.items() if len(v) > 0}

        crop_size = work_item.image.shape[:2]
        metrics = {"location": work_item.extra.location, "area": np.prod(crop_size)}
        if len(containers) > 0:
            containers["use_image"] = True
            if "processing_time" in api_res:
                metrics["latency"] = api_res["processing_time"]
        containers["metrics"] = metrics

        if len(zooms) == 1:
            containers["zoom_image"] = zooms[0]
        elif len(zooms) > 1:
            containers["zoom_image"] = concatenate_images(zooms, is_vertical=True)
        return containers


# ═══════════════════════════════ Internal Analyzer ════════════════════════════


class InternalContainerAnalyzer(ContainerAnalyzerMixin, BaseAdvanceAnalyzer):
    """Container analyzer using local OCR model with BIC fragment assembly.

    Design:
    - Full BIC (ISO-validated) → emit immediately, block serial from re-emission
    - Full BIC (not validated) → emit with low score, don't block
    - Partial segments → add to accumulator pool, cluster by radius, emit assembly
    - Entity SUCCESS only when ALL clusters are ISO-validated
    - Accumulator persists across calls (cleaned on entity expiration)
    """

    _config_type = InternalContainerConfig
    args: InternalContainerConfig

    def __init__(self, msg: dict, context, instance_id: str = None):
        super().__init__(msg, context, instance_id)
        self._init_container(msg, context, instance_id)

        self._field_accumulators: Dict[int, EntityFieldAccumulator] = {}
        self._similarity = WLSimilarity()
        self._ocr_recognition = ContainerOcr(msg)

    def cleanup(self, ids_to_delete: List[int]):
        """Clean up accumulators when entities leave the scene."""
        super().cleanup(ids_to_delete)
        for obj_id in ids_to_delete:
            self._field_accumulators.pop(obj_id, None)

    def _run_model(self, batch: List[WorkItem]) -> List:
        """Internal OCR pipeline: detect → OCR → BIC classify → accumulate → cluster → emit."""
        results = [None] * len(batch)
        for i, work_item in enumerate(batch):
            obj_id = work_item.id
            timestamp = work_item.extra.timestamp if hasattr(work_item.extra, "timestamp") else 0

            detections = self.pre_detector.forward_on_crop_list([work_item.image])[0]
            if len(detections) == 0:
                results[i] = self._build_empty_result(work_item)
                continue

            crops, bboxes = self._extract_detection_crops(work_item.image, detections)
            if not crops:
                results[i] = self._build_empty_result(work_item)
                continue

            texts, scores = self._ocr_recognition.forward_on_crop_list(crops)

            accumulator = self._get_or_create_accumulator(obj_id)
            accumulator.increment_frame()
            immediate_emissions = []

            for text, score, bbox in zip(texts, scores, bboxes):
                if score < self.args.internal_min_ocr_conf:
                    continue

                canonicalized_text = canonicalize_text(text)
                classification = classify_text(canonicalized_text)

                if classification.match_type == BicMatchType.GARBAGE:
                    continue

                effective_score = score * classification.confidence_multiplier

                # === FAST PATH: Full BIC detected in single crop ===
                if classification.match_type == BicMatchType.FULL_BIC:
                    serial_value = None
                    owner_value = None
                    for field_type, field_value in classification.fields:
                        if field_type == BicField.SERIAL_NUMBER:
                            serial_value = field_value
                        elif field_type == BicField.OWNER_CODE:
                            owner_value = field_value

                    # Skip if this serial is already blocked
                    if serial_value and accumulator.is_serial_blocked(
                        serial_value, self._similarity.similarity, self.args.internal_consensus_similarity_th
                    ):
                        continue

                    iso_valid = classification.check_digit_valid
                    if iso_valid:
                        # ISO validated → emit with high score, block serial
                        immediate_emissions.append({
                            "serial": serial_value,
                            "owner": owner_value,
                            "score": effective_score,
                            "bbox": bbox,
                            "iso_validated": True,
                        })
                        if serial_value:
                            accumulator.block_serial(serial_value, effective_score)
                    else:
                        # Not validated → emit with reduced score, don't block
                        immediate_emissions.append({
                            "serial": serial_value,
                            "owner": owner_value,
                            "score": effective_score * 0.7,
                            "bbox": bbox,
                            "iso_validated": False,
                        })
                    continue

                # === SLOW PATH: Partial segments → add to accumulator pool ===
                for field_type, field_value in classification.fields:
                    # Skip blocked serials
                    if field_type == BicField.SERIAL_NUMBER and accumulator.is_serial_blocked(
                        field_value, self._similarity.similarity, self.args.internal_consensus_similarity_th
                    ):
                        continue

                    obs = FieldObservation(
                        value=field_value,
                        score=effective_score,
                        bbox=bbox,
                        timestamp=timestamp,
                        field_type=field_type,
                    )
                    accumulator.add_observation(obs)

            # Build result from immediate emissions + clustered pool
            results[i] = self._build_result(obj_id, accumulator, work_item, immediate_emissions)

        return results

    # ─────────────────────── Crop Extraction ─────────────────────────────────

    def _extract_detection_crops(
        self, image: np.ndarray, detections: np.ndarray
    ) -> Tuple[List[np.ndarray], List[List[float]]]:
        """Extract crops from pre-detector bboxes, filtering by guard band and confidence."""
        h, w = image.shape[:2]
        gb_lims = self.guard_band * np.array([w, h, w, h])
        crops = []
        bboxes = []
        for det in detections:
            if det[4] < self.args.min_ocd_conf:
                continue
            bbox = det[:4]
            if not (np.all(bbox[:2] >= gb_lims[:2]) and np.all(bbox[2:4] <= gb_lims[2:])):
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            bboxes.append([float(x1), float(y1), float(x2), float(y2)])
        return crops, bboxes

    def _get_or_create_accumulator(self, obj_id: int) -> EntityFieldAccumulator:
        if obj_id not in self._field_accumulators:
            self._field_accumulators[obj_id] = EntityFieldAccumulator()
        return self._field_accumulators[obj_id]

    # ─────────────────────── Result Building ─────────────────────────────────

    def _build_result(
        self, obj_id: int, accumulator: EntityFieldAccumulator,
        work_item: WorkItem, immediate_emissions: List[Dict]
    ) -> Dict:
        """Build result combining immediate full-BIC emissions + clustered partial assemblies."""
        containers = {}
        zooms = []
        all_cluster_scores = []

        # 1) Add immediate emissions (full BIC fast path)
        size_pool = accumulator.get_segments_by_field(BicField.SIZE_CODE)
        used_sizes_fast = set()

        for emission in immediate_emissions:
            serial = emission["serial"]
            owner = emission["owner"]
            em_score = emission["score"]
            bbox = emission["bbox"]

            if serial:
                if ContainerAttributes.SERIAL_NUMBER.value not in containers:
                    containers[ContainerAttributes.SERIAL_NUMBER.value] = []
                containers[ContainerAttributes.SERIAL_NUMBER.value].append(
                    {"value": serial, "score": em_score, "bbox": bbox}
                )
                # Zoom crop for serial
                if bbox:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    h, w = work_item.image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        zooms.append(work_item.image[y1:y2, x1:x2])

            if owner:
                if ContainerAttributes.OWNER_CODE.value not in containers:
                    containers[ContainerAttributes.OWNER_CODE.value] = []
                containers[ContainerAttributes.OWNER_CODE.value].append(
                    {"value": owner, "score": em_score, "bbox": bbox}
                )

            # Attach closest size/type from accumulator pool
            if bbox and size_pool:
                anchor_obs = FieldObservation(value="", score=0, bbox=bbox)
                closest_size = accumulator._find_closest(anchor_obs.centroid, anchor_obs.radius, size_pool, used_sizes_fast)
                if closest_size is not None:
                    used_sizes_fast.add(id(closest_size))
                    if ContainerAttributes.SIZE_CODE.value not in containers:
                        containers[ContainerAttributes.SIZE_CODE.value] = []
                    containers[ContainerAttributes.SIZE_CODE.value].append(
                        {"value": closest_size.value, "score": closest_size.score, "bbox": closest_size.bbox}
                    )

            # ISO-validated full BIC → score 1.0 for this cluster
            if emission["iso_validated"]:
                all_cluster_scores.append(1.0)
            else:
                all_cluster_scores.append(em_score)

        # 2) Cluster partial segments from accumulator pool
        clusters = accumulator.cluster_by_radius(
            self._similarity.similarity, self.args.internal_consensus_similarity_th
        )

        field_to_attr = {
            BicField.OWNER_CODE: ContainerAttributes.OWNER_CODE.value,
            BicField.SERIAL_NUMBER: ContainerAttributes.SERIAL_NUMBER.value,
            BicField.SIZE_CODE: ContainerAttributes.SIZE_CODE.value,
        }

        for cluster in clusters:
            cluster_score = 0.0
            has_serial = BicField.SERIAL_NUMBER in cluster
            has_owner = BicField.OWNER_CODE in cluster

            # Attempt ISO validation if we have both owner (4 chars) and serial (7 digits)
            iso_valid = None
            if has_serial and has_owner:
                serial_obs = cluster[BicField.SERIAL_NUMBER]
                owner_obs = cluster[BicField.OWNER_CODE]
                # Try to validate: owner (4 chars) + serial (7 digits) = full BIC
                if len(owner_obs.value) == 4 and len(serial_obs.value) == 7:
                    full_code = owner_obs.value + serial_obs.value
                    iso_valid = validate_check_digit(full_code)

            for bic_field, obs in cluster.items():
                attr_key = field_to_attr.get(bic_field)
                if not attr_key:
                    continue

                # Only emit isolated size/type if it's part of a cluster with serial or owner
                if bic_field == BicField.SIZE_CODE and not has_serial and not has_owner:
                    continue

                if attr_key not in containers:
                    containers[attr_key] = []
                containers[attr_key].append({"value": obs.value, "score": obs.score, "bbox": obs.bbox})

                if bic_field == BicField.SERIAL_NUMBER and obs.bbox:
                    bbox = obs.bbox
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    h, w = work_item.image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        zooms.append(work_item.image[y1:y2, x1:x2])

                cluster_score = max(cluster_score, obs.score)

            # Determine cluster score based on validation state
            if iso_valid is True:
                # Assembled and validated — block the serial
                serial_obs = cluster[BicField.SERIAL_NUMBER]
                accumulator.block_serial(serial_obs.value, cluster_score)
                all_cluster_scores.append(1.0)
            elif iso_valid is False:
                # Assembled but failed validation
                all_cluster_scores.append(cluster_score * 0.5)
            elif has_serial or has_owner:
                # Partial (serial-only or owner-only) — low score to keep entity alive
                all_cluster_scores.append(cluster_score * 0.3)

        # Build metrics
        crop_size = work_item.image.shape[:2]
        n_blocked = len(accumulator.blocked_serials)
        n_clusters = len(clusters) + len(immediate_emissions)
        metrics = {
            "location": work_item.extra.location if hasattr(work_item.extra, "location") else -1,
            "area": int(np.prod(crop_size)),
            "backend_mode": "internal",
            "accumulation_frames": accumulator.frame_count,
            "container_count": n_clusters,
            "blocked_serials": n_blocked,
            "pool_size": len(accumulator.segment_pool),
        }

        if not containers:
            return self._build_empty_result(work_item)

        containers["use_image"] = True
        containers["metrics"] = metrics

        if len(zooms) == 1:
            containers["zoom_image"] = zooms[0]
        elif len(zooms) > 1:
            containers["zoom_image"] = concatenate_images(zooms, is_vertical=True)

        # Store cluster scores for calculate_post_score to use
        # min score across all clusters determines entity fate
        containers["_cluster_scores"] = all_cluster_scores

        return containers

    def _build_empty_result(self, work_item: WorkItem) -> Dict:
        crop_size = work_item.image.shape[:2]
        return {
            "metrics": {
                "location": work_item.extra.location if hasattr(work_item.extra, "location") else -1,
                "area": int(np.prod(crop_size)),
                "backend_mode": "internal",
            }
        }


# ═══════════════════════════════ Factory Helper ═══════════════════════════════


def create_container_analyzer(msg: dict, context, instance_id: str = None):
    """Factory function: pick ExternalContainerAnalyzer or InternalContainerAnalyzer.

    Uses `use_external_apis` from the analytic config (l1_models section).
    For containers, default (use_external_apis=False) means internal OCR.
    """
    use_external_apis = getattr(context, "l1_config", {}).get("use_external_apis", False) if context else False
    if use_external_apis:
        return ExternalContainerAnalyzer(msg, context, instance_id)
    return InternalContainerAnalyzer(msg, context, instance_id)


# Backward-compat alias — existing code that imports ContainerAnalyzer will get the external one.
ContainerAnalyzer = ExternalContainerAnalyzer
