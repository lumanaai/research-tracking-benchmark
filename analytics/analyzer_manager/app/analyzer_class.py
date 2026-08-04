import glob
import json
import os
import shutil
import threading
import time
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Dict, Optional

import cv2
import numpy as np
from shapely import Polygon

from alerts.alerts import AlertManager
from detection.detector import DetectorFactory, BaseDetector
from general import proj, apply_from_dict_safe, trim_str
from general.analyzer_general import (
    logger,
    PrivacyType,
    ROI_SHAPE,
    roi_gen,
    recursive_update,
    VEHICLE_FILTERS,
    InferenceType,
    PERSON_FILTERS,
    PPE_FILTERS,
    BiometricProperty,
    extract_num_classes,
    CameraType,
    camera_type_to_location_center,
    log_exception,
)
from general.clip_encoder import ClipDispatcher
from general.core import (
    AlertsAction,
    AnalyticMode,
    AnalyticImage,
    ClassHandler,
    EndpointFactory,
    AdvanceAnalyzerType,
    Event,
    AlertType,
    BatchDataFactory,
    SpecialClasses,
    ZoneRegion,
)
from general.cuda_utils import NvJpegEncoder
from general.db_handler import SQLiteReader, SQLITE_KNOW_TABLES, generate_unified_table
from general.entity_db import EntityDB
from general.error_handle import try_except_raise
from general.image_quality_monitor.iq_monitor import ImageQualityMonitor
from general.img_utils import is_image_monochrome
from general.offline_analytics import check_expert_availability, OfflineAnalyticsClientFactory
from general.perf_utils import time_sync
from general.proj import load_bool_from_env, load_config, load_configAnalytic, set_environment_var, get_host
from general.triton_utils import unload_triton_model
from level1.l1_manager import L1Manager
from load_balancer.load_balancer import LoadBalancer
from manager.common import ManagementMessage, AnalyticTrainingMode
from metadata_analyzer.logistics_md_manager import LogisticsMetadataManager
from metadata_analyzer.md_manager import MetadataManager
from preprocessing import PreprocessorFactory, PreprocessorBase
from tracking.tracker import TrackerFactory, TrackerType

set_environment_var()


def first_part(model_name: str) -> str:
    return model_name.split("_")[0]


class Analyzer(object):
    edge_id: str
    camera_id: str
    file_processing: bool

    tracker_en: bool

    slow_start_n_frames: int
    slow_start_skips: int
    warn_on_skip: bool = False

    batch_size: int = 4  # default

    last_batch_image_idx: int
    image_idx: int
    nvjpeg_encoder_address = None
    is_logistic: bool = False
    image_batch: List[AnalyticImage]
    load_balancer: LoadBalancer
    iq_monitor: ImageQualityMonitor
    alert_manager: AlertManager
    class_handler: ClassHandler
    entity_db: EntityDB
    md_manager: MetadataManager
    detector: BaseDetector
    analytic_db: Dict[str, List]

    is_night_mode = False
    night_mode_last_per = 0
    night_mode_period_ms = 5 * 60 * 1000
    on_night_mode_changed: Event

    image_quality: bool = False
    last_iq_scores: Dict[str, Any] = {}
    on_low_grade_changed: Event

    resolution: List[int]
    preprocessor: PreprocessorBase = None

    # analyzer status
    total_processed = 0
    alert_status = False
    synced_status = True
    offline_status = True

    def __init__(self, init_msg, app_config=None):
        self.edge_id = init_msg["edgeId"]
        self.camera_id = init_msg["cameraId"]
        self.analyzer_id = f"{self.edge_id}_{self.camera_id}"
        self.file_processing = init_msg["footageAnalysis"] if "footageAnalysis" in init_msg else False
        self.camera_document = init_msg["analyticAppParameters"]
        self.zone_document = self.camera_document.get("zoneRegions", [])
        input_stream_msg = self.camera_document["inputStream"]
        self.camera_type = input_stream_msg.get("cameraType", CameraType.STATIC.value)
        self.is_location_center = camera_type_to_location_center.get(self.camera_type, False)
        self.analytic_db = {"persons": [], "custom_objects": [], "shelves": []}
        self.on_db_update = Event()
        self.zone_regions = {}

        try:
            self.update_resolution(input_stream_msg)
        except ValueError:
            logger.error(f"Illegal resolution. {self.resolution[0]}x{self.resolution[1]}")

        # Config stage
        if app_config is None:
            self.app_config = load_config()
        else:
            self.app_config = app_config

        self.config = self.load_analytic_config()

        inference_server_settings = self.app_config.get("analytics", {}).get("inferenceServerUri", {})
        endpoint_factory = EndpointFactory(inference_server_settings)

        # statistic setup
        self.log_input_ts = self.config.get("debug", {}).get("logInputTimestamp", False)
        self._reset_statistic()

        self.on_night_mode_changed = Event()

        # weights Lookup in assets if exists
        special_weights = self.weights_lookup()
        mock_endpoint = endpoint_factory.create("")
        self.local_inference = True
        force_triton = proj.load_bool_from_env("TRITON_FORCED", False)
        if proj.load_bool_from_env("TRITON_ENABLED", True) or force_triton:
            from general.triton_utils import is_triton_live

            attempts = 3 if force_triton else 1
            for aidx in range(attempts):
                if is_triton_live(mock_endpoint):
                    self._load_weights_to_triton(special_weights)
                    self.local_inference = False
                    break
                elif force_triton:
                    logger.warning(f"{aidx + 1}/{attempts} Connection to Triton failed")
                    time.sleep(5)

        self.last_offline_check = 0
        self.offline_check_interval = 60 * 1000  # every minute
        self.offline_client = None
        self.offline_url = None
        self.offline_enabled = self.config.get("offline_analytics", {}).get("enable", True)
        if self.offline_enabled:
            offline_analytics_settings = self.app_config.get("analytics", {}).get("offlineAnalyticsUri", {})
            self.offline_url = check_expert_availability(offline_analytics_settings)
            self.init_offline_client()

        if self.local_inference:
            if force_triton:
                raise ConnectionError("Triton not available")
            self._load_weights_locally(special_weights)

        self.registered_models = {k: Path(v).stem for k, v in special_weights.items()}
        logger.info(f"Registered models: {self.registered_models}")

        logger.info(f"Parsed Init: {init_msg}")
        logistic_config = self.camera_document.get("logistic", None) or self.config.get("logistic", None)
        detector_bypass = self.camera_document.get("analyticMode", None) == AnalyticMode.thumbnails_only.value
        if logistic_config is None:
            # Detector
            t0 = time_sync()
            detector_config = self.config["l0_detect"]
            detector_config["is_bypass"] = detector_bypass or detector_config.get("is_bypass", False)
            detector_config.setdefault("name", "yolov8")
            self.detector = DetectorFactory().create(detector_config["name"], detector_config)
            self.class_handler = self.detector.class_handler
            self.batch_size = self.detector.batch_size
            self.detector_resolution = self.detector.input_size
            self.batch_data_factory = BatchDataFactory(self)
            self._init_preprocessor()
            self.read_from_db()

            # load balancer
            self.load_balancer = LoadBalancer(self.config["load_balance"], self.detector_resolution)
            self.clip_dispatcher = ClipDispatcher(self.config.get("clip", {}), self)

            # image quality monitor
            self.iq_monitor = ImageQualityMonitor(self.config.get("image-quality", {}), self)

            t1 = time_sync()
            track_config = self.config["l0_track"]
            # Tracker
            if track_config is not None and not load_bool_from_env("AI_BYPASS_TRACKER", False):
                fps = int(self.app_config["streamer"]["analyticFramerate"])
                tracker = TrackerFactory().create(track_config["name"], track_config, fps, self)
                track_en = True
            else:
                tracker = None
                track_en = False
            self.tracker = tracker
            self.tracker_en = track_en

            if get_host() == "xavier":  # is_jetson_platform():  #
                self.nvjpeg_encoder_address = NvJpegEncoder().create_nvjpeg_encoder()

            json_sync_period = self.config["jsonPolicy"]["min_json_gap"]
            self.entity_db = EntityDB(context=self, inactivity_period_sec=6, sync_period_ms=json_sync_period)

            t2 = time_sync()
            self.l1_manager = L1Manager(self)
            self.entity_db.on_entities_stats_update += self.tracker.on_entities_stats_update

            self.init_search_and_alerts()
            self.update_l1_config()

            t3 = time_sync()
            t_init = int((t3 - t0) * 1000)
            t_det = int((t1 - t0) * 1000)
            t_track = int((t2 - t1) * 1000)
            t_l1 = int((t3 - t2) * 1000)
            logger.info(
                f"Init app in {t_init} [ms]: Detector: {t_det} [ms] Tracker: {t_track} [ms] L1 models: {t_l1} [ms]"
            )
        else:
            from level1.text.text_analyzer import TextAnalyzer

            self.is_logistic = True
            self.text_analyzer = TextAnalyzer(self, logistic_config)
            self.log_md_manager = LogisticsMetadataManager(self)
            self.batch_size = self.text_analyzer.batch_size
            self.detector = self.text_analyzer.detector
            self._init_preprocessor()

        init_mode = "testing" if load_bool_from_env("AI_TESTMODE", False) else "functional"
        self.slow_start_n_frames = self.app_config["analytics"]["initMode"][init_mode]["initNumberOfFrames"]
        if self.file_processing:
            self.slow_start_n_frames = 0
        self.slow_start_skips = self.app_config["analytics"]["initMode"][init_mode]["initSkipRate"] - 1
        logger.info(f"Skipping {self.slow_start_n_frames} at {self.slow_start_skips + 1} rate on {init_mode} mode")

        self.image_idx = self.slow_start_skips  # to force first image to be processed
        self.image_batch = []
        self.last_batch_image_idx = -1
        self._batch_lock = threading.Lock()
        self._last_batch_data = None
        self._skipped_frames = []

    def init_offline_client(self):
        try:
            self.offline_client = OfflineAnalyticsClientFactory.get_client(self.offline_url, self.camera_id)
        except Exception as e:
            log_exception(logger, f"Failed to initialize offline analytics client", e)
            self.offline_client = None

    def __del__(self):
        if self.nvjpeg_encoder_address is not None:
            NvJpegEncoder.destroy(self.nvjpeg_encoder_address)

    def _reset_statistic(self):
        # statistic stage
        self.last_stat_timestamp = 0
        self.last_stat_report = 0
        self.analytic_update = 0
        self.training_count = 0

        # putting two elements just for first iteration statistic
        self.input_frames_timestamp = [0, 0]
        self.processed_frames_timestamp = [0, 0]
        self.input_rate = [0, 0]
        self.output_rate = [0, 0]

    def update_statistic(self, timestamp, result):

        # calculating rate
        if (timestamp - self.last_stat_timestamp) > self.app_config["analytics"]["statAnalyzePeriod"]:

            input_gaps = [b - a for a, b in zip(self.input_frames_timestamp, self.input_frames_timestamp[1:])]
            output_gaps = [b - a for a, b in zip(self.processed_frames_timestamp, self.processed_frames_timestamp[1:])]

            if self.log_input_ts:
                logger.info(f"Input timestamps: {self.input_frames_timestamp}")
                logger.info(f"Input gaps: {input_gaps}")

            self.input_rate.append(round(1000 / np.mean(input_gaps), 2))
            self.output_rate.append(round(1000 / np.mean(output_gaps), 2))
            self.input_frames_timestamp.clear()
            self.processed_frames_timestamp.clear()
            self.last_stat_timestamp = timestamp

        if result.metadata is not None:
            # Sending training images
            self.training_count += sum(1 for s in result.metadata.training if "training" in s)
            # Sending jsons
            if result.metadata.info is not None:
                self.analytic_update += 1
                # checking if we need to send statistic
                report_period = timestamp - self.last_stat_report
                if report_period > self.app_config["analytics"]["statReportPeriod"]:
                    # building status
                    result.metadata.info["metadata"]["statistic"] = {
                        "timestamp": timestamp,
                        "inputRate": {
                            "max": max(self.input_rate),
                            "min": min(self.input_rate),
                            "avg": round(np.mean(self.input_rate), 2),
                            "std": round(np.std(self.input_rate), 2),
                        },
                        "outputRate": {
                            "max": max(self.output_rate),
                            "min": min(self.output_rate),
                            "avg": round(np.mean(self.output_rate), 2),
                            "std": round(np.std(self.output_rate), 2),
                        },
                        "analyticUpdateRate": round(self.analytic_update * 1000 / report_period, 3),
                        "trainingUpdateRate": round(self.training_count * 1000 / report_period, 3),
                        "l1Stats": result.l1Stats,
                        "imageQualityScores": self.last_iq_scores if self.last_iq_scores else None,
                    }

                    self.input_rate.clear()
                    self.output_rate.clear()
                    self.analytic_update = 0
                    self.training_count = 0
                    self.last_stat_report = timestamp

        return result

    # extracting YUV420 to YUV444 (cv2 format)
    def cpu_image_extract(self, image_batch: List[AnalyticImage], s):
        w = self.resolution[0]
        h = self.resolution[1]
        for image in image_batch:
            byte_array = bytearray(image.frame)
            y = byte_array[0:s]
            y = np.reshape(y, (h, w))
            y = y.astype(np.uint8)

            u = byte_array[s : s + int(s / 4)]
            u = np.repeat(u, 2, 0)
            u = np.reshape(u, (int(h / 2), w))
            u = np.repeat(u, 2, 0)
            u = u.astype(np.uint8)

            v = byte_array[s + int(s / 4) :]
            v = np.repeat(v, 2, 0)
            v = np.reshape(v, (int(h / 2), w))
            v = np.repeat(v, 2, 0)
            v = v.astype(np.uint8)
            image.frame = [y, u, v]
        return image_batch

    @try_except_raise(logger=logger, raise_exception=True)
    def _process_batch(self):
        analyzer_res = Namespace(metadata=None, motion_info=None, l1Stats={}, counting_info=None, encodings=None)
        # Filling the batch and start run analytics
        performance = {}
        image_batch = self.image_batch[-self.batch_size :]
        batch_sz = len(image_batch)
        timestamps = [image.timestamp for image in self.image_batch]

        # Image extract
        t0 = time_sync()
        self.preprocessor.process_batch(image_batch)

        t1 = time_sync()
        performance["ImageExtract"] = f"{round((t1 - t0) * 1000 / batch_sz, 1)} [ms]"
        logger.debug(f"Completed Image Extract in {performance['ImageExtract']} per frame")

        if self.is_logistic:
            results = self.text_analyzer.run(image_batch)
            analyzer_res.metadata = self.log_md_manager.run(image_batch, results)
            return analyzer_res

        t2 = time_sync()
        performance["LoadBalancer"] = f"{round((t2 - t1) * 1000 / batch_sz, 1)} [ms]"
        logger.debug(f"Completed load balancing in {performance['LoadBalancer']} per frame")

        # Detection batch
        # detRes = detector.run(self.detector_id, image_batch, bgr_batch_gpu)
        # det_res = self.detector.run(image_batch, bgr_batch_gpu)
        det_res = self.detector.run(self.preprocessor.detector_inputs_mem, timestamps)

        t3 = time_sync()
        performance["Detector"] = f"{round((t3 - t2) * 1000 / len(det_res.predictions), 1)} [ms]"
        logger.debug(f"Completed detection batch in {performance['Detector']} per frame")

        # Tracker batch
        if self.tracker_en:
            trk_res = self.tracker.run(det_res, image_batch)
        else:
            trk_res = []
            for pred in det_res.predictions:
                pred_np = pred.data
                n_preds = pred_np.shape[0]
                unknown = np.full(n_preds, -1)
                t_res = np.c_[pred_np[:, 0:4], unknown, pred_np[:, 5:7], pred_np[:, 4], np.arange(n_preds), unknown]
                trk_res.append(t_res)

        t4 = time_sync()
        if len(trk_res) > 0:
            performance["Tracker"] = f"{round((t4 - t3) * 1000 / len(trk_res), 1)} [ms]"
            logger.debug(f"Completed tracking batch in {performance['Tracker']} per frame")

        self.update_night_mode(timestamps, self.preprocessor.detector_images_bgr)

        # Update low grade image status
        curr_out, curr_scores = self.iq_monitor.iq_monitor(self.preprocessor.detector_images_bgr, timestamps)
        if curr_scores is not None:
            self.image_quality = not curr_out
            self.last_iq_scores = curr_scores

        # Motion estimation
        motion_data = self.load_balancer.extract_motion_estimations(
            timestamps, self.preprocessor.detector_images_bgr, None
        )

        # Running l1 only on batches with tracked objects
        batch_data = self.batch_data_factory.create(trk_res, timestamps)
        self.entity_db.track(batch_data, image_batch, motion_data)

        offline_ok = self.offline_client is not None
        if offline_ok:
            offline_ok = self.offline_client.prepare_batch()

        if (
            not offline_ok
            and self.offline_enabled
            and timestamps[-1] - self.last_offline_check > self.offline_check_interval
        ):
            # first try to re-init the client if not initialized
            is_healed = True
            if self.offline_client is None:
                self.init_offline_client()
            if self.offline_client is not None:
                self.offline_client.sync()
                if not self.offline_client.is_healthy:
                    is_healed = self.offline_client.try_heal()
            self.offline_status = self.offline_client is not None and is_healed
            self.last_offline_check = timestamps[-1]

        alerted_ents = self.alert_manager.check_alerts(image_batch, batch_data, motion_data)

        l1_results = []
        if self.tracker_en:
            l1_results, analyzer_res.l1Stats = self.l1_manager.analyze(
                image_batch, batch_data, alerted_ents, motion_data
            )
            self.entity_db.update_l1_attributes(l1_results)

        t5 = time_sync()
        performance["L1_Model"] = f"{round((t5 - t4) * 1000 / len(trk_res), 1)} [ms]"
        logger.debug(f"Completed l1 batch in {performance['L1_Model']} per frame")

        # Aggregating all information under single json
        performance["FullPipe"] = f"{round((t5 - t0) * 1000 / len(det_res.predictions), 1)} [ms]"
        logger.debug(f"Completed analyzer batch in {performance['FullPipe']} per frame")
        logger.debug("running analysis")

        self.run = self.md_manager.run(
            image_batch,
            batch_data,
            self.preprocessor.detector_images_bgr,
            l1_results,
            performance,
            self.is_night_mode,
            motion_data,
            self._skipped_frames,
            set(alerted_ents.keys()),
        )
        analyzer_res.metadata, analyzer_res.motion_info, analyzer_res.counting_info, analyzer_res.encodings = self.run
        self._last_batch_data = batch_data
        self.total_processed += self.batch_size
        return analyzer_res

    def _load_weights_to_triton(self, special_weights):
        # 1. check and map the assets to local repository
        # 2. load the model in triton - will be done by the calling module automatically
        # 3. update configuration
        from general.triton_utils import load_triton_model

        model_repository = os.path.join(proj.assets_path(), "model_repository")
        os.makedirs(model_repository, exist_ok=True)

        for key in special_weights:
            config = self._custom_key_to_config(key)
            asset = Path(special_weights[key])

            if config is None:
                continue
            model_name = first_part(config.get("name", ""))
            if not model_name:
                logger.warning(f"cant determine the asset {asset} remote network")
                continue
            new_model_name = model_name + f"_{self.camera_id}"
            model_infer_path = os.path.join(model_repository, new_model_name)
            requires_load = True
            if os.path.exists(model_infer_path):
                model_file = glob.glob(os.path.join(model_infer_path, "1", "model.*"))
                if len(model_file) > 0:
                    requires_load = asset.stat().st_nlink == 1 or len(model_file) > 1
                    for file in model_file:
                        if requires_load:  # means that it has hardlink to it
                            os.remove(file)
            if requires_load:
                # special treatment for yolo v8 since the old export process contains invalid engine files
                if "yolov8" in model_name:

                    def trim_v8(weights, file_out):
                        with open(weights, "rb") as f, open(file_out, "wb") as t:
                            meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
                            json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
                            model = f.read()  # read engine
                            t.write(model)

                    try:
                        normalized = asset.with_suffix(".normed")
                        trim_v8(asset.resolve(), normalized.resolve())
                        tmp = deepcopy(asset)
                        tmp.replace(tmp.with_suffix(".orig"))
                        normalized.replace(asset)
                    except Exception:
                        pass
                os.makedirs(os.path.join(model_infer_path, "1"), exist_ok=True)

                # new config new with the correct endpoint name
                if model_name.startswith("cls-alert") or model_name.startswith("state-object"):
                    config_path = proj.info_path("triton", "trt_config.pbtxt")
                else:
                    config_path = proj.local_inference_repository_path(model_name, "config.pbtxt")
                new_config_path = os.path.join(model_infer_path, "config.pbtxt")
                with open(config_path, "rt") as source, open(new_config_path, "wt") as target:
                    first_line = ""
                    while len(first_line.strip()) == 0:
                        first_line = source.readline()  # name
                    target.write(f'name: "{new_model_name}"\n')
                    lines = source.read()
                    target.write(lines)

                # hard link the new weight
                if asset.suffix in [".engine", ".plan"]:
                    file_name = "model.plan"
                else:
                    file_name = "model" + asset.suffix
                os.link(asset, os.path.join(model_infer_path, "1", file_name))

            # now that the file has been copied for the triton inference folder, we can ask triton to load it
            # this is done in the block itself when probing if network exist, but also done here to ensure the model is
            # loaded correctly, otherwise the inference will roll back to its default
            endpoint_info = EndpointFactory().create(new_model_name)
            is_loaded = load_triton_model(endpoint_info)
            if is_loaded:
                config["name"] = new_model_name
                logger.info(f"{new_model_name} was uploaded correctly to inference server and will be used")

    def _custom_key_to_config(self, key) -> Optional[Dict]:
        if key == "l0_detect":
            return self.config["l0_detect"]
        elif key == "l0_track":
            return self.config["l0_track"]
        elif key == "pa":
            return self.config["l1_models"]["attributes"]["human_parsing"]["pa"]
        elif key == "reid":
            return self.config["l1_models"]["attributes"]["human_parsing"]["reid"]
        elif key == "ppe":
            return self.config["l1_models"]["attributes"]["human_parsing"]["ppe"]
        elif key == "doors":
            return self.config["l1_models"]["attributes"]["doors"]
        elif key == "wc":
            return self.config["l1_models"]["attributes"]["weapons_classification"]
        elif key == "lpc":
            return self.config["l1_models"]["attributes"]["lpc"]
        elif key.startswith("cls-alert"):
            return self.config["cls-alerts"][key]
        elif key.startswith("state-object"):
            return self.config["state-objects"][key]
        logger.warning(f"cant parse key {key} when trying to determine the asset target")
        return None

    def _load_weights_locally(self, special_weights):
        for key in special_weights:
            asset = special_weights[key]
            config = self._custom_key_to_config(key)
            if config is not None:
                config["unique_weights"] = asset
                logger.info(f"{key} uses {asset} for {self.camera_id}")

    def weights_lookup(self, assets_names: List = None) -> Dict[str, str]:
        custom_weights = {}
        cam_assets_path = os.path.join(proj.assets_path(), self.edge_id, self.camera_id)

        if assets_names is None:
            # check for assets
            suitable_extensions = ["engine", "pt", "pth", "onnx", "csv"]
            assets_names = []
            os.makedirs(cam_assets_path, exist_ok=True)
            for ext in suitable_extensions:
                assets_names.extend(glob.glob(os.path.join(cam_assets_path, f"*.{ext}")))
            assets_names = sorted(assets_names, key=os.path.getmtime)
            logger.info(f"Found {len(assets_names)} assets for {self.camera_id} under {cam_assets_path}")

        if "human_parsing" not in self.config["l1_models"]["attributes"]:
            self.config["l1_models"]["attributes"]["human_parsing"] = {"pa": {}, "reid": {}, "ppe": {}}

        no_pf = os.path.exists(os.path.join(cam_assets_path, "no.pf"))

        # look for detections
        for asset in assets_names:
            asset_name = Path(asset).name
            if "classes" in asset:
                if no_pf:
                    logger.warning(f"Skipping {asset} for {self.camera_id} due to no.pf file")
                    continue
                logger.info(f"Analyzer uses classes metadata from {asset} for {self.camera_id}")
                self.config["l0_detect"]["classes_metadata"] = asset
            elif first_part(self.config["l0_detect"]["name"]) in asset:
                if no_pf:
                    logger.warning(f"Skipping {asset} for {self.camera_id} due to no.pf file")
                    continue
                custom_weights["l0_detect"] = asset
                self.config["l0_track"]["detector_unique"] = True
                num_classes = extract_num_classes(asset_name)
                if num_classes is not None:
                    if "num_classes" not in self.config["l0_detect"]:
                        self.config["l0_detect"]["num_classes"] = num_classes
                    face_config = self.config["l1_models"]["attributes"].get("face", {})
                    face_config["use_faces_from_detector"] = num_classes > 22
                    self.config["l1_models"]["attributes"]["face"] = face_config

            elif first_part(self.config["l0_track"]["name"]) in asset:
                custom_weights["l0_track"] = asset
            elif asset_name.startswith("pa"):
                custom_weights["pa"] = asset
                self.config["l1_models"]["attributes"]["human_parsing"]["pa"].setdefault(
                    "name", InferenceType.PERSON_ATTR
                )
            elif asset_name.startswith("reid"):
                custom_weights["reid"] = asset
                self.config["l1_models"]["attributes"]["human_parsing"]["reid"].setdefault(
                    "name", InferenceType.PERSON_REID
                )
            elif asset_name.startswith("ppe"):
                custom_weights["ppe"] = asset
                self.config["l1_models"]["attributes"]["human_parsing"]["ppe"].setdefault(
                    "name", InferenceType.PPE_ATTR
                )
            elif asset_name.startswith("doors"):
                custom_weights["doors"] = asset
                self.config["l1_models"]["attributes"]["doors"].setdefault("name", InferenceType.DOORS_CLASSIFICATION)
            elif asset_name.startswith("wc"):
                custom_weights["wc"] = asset
                self.config["l1_models"]["attributes"]["weapons_classification"].setdefault(
                    "name", InferenceType.WEAPONS_CLASSIFICATION
                )
            elif asset_name.startswith("cls-alert") or asset_name.startswith("state-object"):
                prefix = "cls-alert" if asset_name.startswith("cls-alert") else "state-object"
                category_name = prefix + "s"
                custom_name = first_part(asset_name)
                custom_weights[custom_name] = asset
                if category_name not in self.config:
                    self.config[category_name] = {}
                self.config[category_name][custom_name] = {"name": custom_name, "weights": asset}

            elif not ("config" in asset):
                logger.error(f"{self.camera_id} holds asset {asset} that is not fitting any model")
        return custom_weights

    def load_analytic_config(self):
        analytic_config = load_configAnalytic()
        # check if in assets there is an update config
        assets_dir = os.path.join(proj.assets_path(), self.edge_id, self.camera_id)
        assets_names = sorted(glob.glob(os.path.join(assets_dir, "configAnalytic*.*")), key=os.path.getmtime)
        # detectorConfig, trackerConfig, etc.
        part_configs = sorted(glob.glob(os.path.join(assets_dir, "*Config.*")), key=os.path.getmtime)
        if len(assets_names) > 0 or len(part_configs) > 0:
            base_config = assets_names[-1:] if len(assets_names) else []
            for asset in base_config:
                logger.info(f"Found config: {asset} for {self.camera_id} ")
                assets_config = load_configAnalytic(asset)
                recursive_update(analytic_config, assets_config)
            for part in part_configs:
                logger.info(f"Found partial config: {part} for {self.camera_id} ")
                with open(part) as f:
                    part_dict = json.load(f)
                    detection_name = part_dict.get("l0_detect", {}).get("name", None)
                    # patch due to bad proper fitting
                    if detection_name is not None and detection_name not in ["yolov8", "yolov5"]:
                        logger.error(f"Unsupported detection model {detection_name} in {part}. assusming yolov8")
                        part_dict["l0_detect"]["name"] = "yolov8"
                    recursive_update(analytic_config, part_dict)
        return analytic_config

    def _gen_roi_filter_from_msg(self, msg, roi=None):
        special_objects = self.config.get("special_objects", {})
        is_roi_filter = roi is not None
        objects_to_scan = set()
        special_classes = set()
        filter_out = {}

        for obj in msg.get("objectsToScan", []):
            if self.class_handler.is_int_an_object(obj):
                objects_to_scan.add(obj)

        # search no longer activates LPR - only container ID or alerts
        plate_en = False  # bool(msg.get("licensePlates", False))
        mmc_en = False  # bool(msg.get("vehicleMMC", False))
        con_en = bool(msg.get("container", False)) or bool(special_objects.get("container", False))
        forklift_en = bool(msg.get("forklift", False)) or bool(special_objects.get("forklift", False))
        shoppingcart_en = bool(msg.get("shoppingCart", False)) or bool(special_objects.get("shoppingCart", False))
        lpr_en = plate_en or mmc_en or con_en

        if lpr_en or forklift_en:
            objects_to_scan.add(self.class_handler.vehicle_value)

        if shoppingcart_en:
            special_classes.add(SpecialClasses.SHOPPING_CART.value)

        if forklift_en:
            special_classes.add(SpecialClasses.FORKLIFT.value)

        # update roiFilter to be false on these objects
        for obj in objects_to_scan:
            filter_out[obj] = Namespace(roi=roi, roiFilter=is_roi_filter)

        if plate_en:
            filter_out["plate"] = Namespace(roi=roi, roiFilter=is_roi_filter)
        elif mmc_en:
            filter_out["mmc"] = Namespace(roi=roi, roiFilter=is_roi_filter)
        if con_en:
            filter_out["containerId"] = Namespace(roi=roi, roiFilter=is_roi_filter)

        if bool(msg.get("protectiveGear", False)):
            filter_out["protectiveGear"] = Namespace(roi=roi, roiFilter=is_roi_filter)

        return filter_out, objects_to_scan, special_classes

    def update_biometric(self, msg):
        gender_classification = msg.get(BiometricProperty.GENDER, None)
        if gender_classification is not None:
            # this will change the state of the attribute filter according to gender_classification
            # empty will remove existing filter
            attr_dict = {"attribute_filter": ["genderType"] if not gender_classification else []}
            self.l1_manager.update_l1_config(AdvanceAnalyzerType.HUMAN_PARSING, filter_in=attr_dict)

        face_en = msg.get(BiometricProperty.FACE, None)
        if face_en is not None:
            self.l1_manager.update_l1_config(AdvanceAnalyzerType.FACE, enable=face_en)

    def update_search(self, msg):
        roi_filters = []
        classes_filter = self.detector.class_filter
        objects_filter = self.detector.objects_filter
        privacy_filter = []
        privacy_regs = []
        privacy_type = PrivacyType.PIXELATE

        thumb_input = self.detector.input_size
        scale_factor = self.md_manager.thumb_width / thumb_input[1]
        height = np.floor(thumb_input[0] * scale_factor / 2) * 2
        thumb_size = (self.md_manager.thumb_width, int(height))

        if (
            msg is not None
            and "action" in msg
            and (msg["action"] == AlertsAction.add.value or msg["action"] == AlertsAction.update.value)
        ):
            objects_to_scan = set()
            special_classes = set()

            if "objectsToScan" in msg or "zones" in msg:

                # later on we will add subclasses
                global_search = ("objectsToScan" in msg) and msg["objectsToScan"] is not None
                zone_search = "zones" in msg and msg["zones"] is not None and len(msg["zones"]) > 0

                if global_search:
                    g_roi_filter, g_objects, g_special_classes = self._gen_roi_filter_from_msg(msg)
                    roi_filters.append(g_roi_filter)
                    objects_to_scan.update(g_objects)
                    special_classes.update(g_special_classes)

                if zone_search:
                    zones = msg["zones"]
                    for key in zones:
                        zone = zones[key]

                        if not ("markedIdx" in zone) or zone["markedIdx"] is None or not (len(zone["markedIdx"])):
                            logger.error(f"illegal search configuration")
                            continue
                        z_roi = roi_gen(zone["markedIdx"])

                        z_roi_filter, z_objects, z_special_classes = self._gen_roi_filter_from_msg(zone, z_roi)
                        roi_filters.append(z_roi_filter)
                        objects_to_scan.update(z_objects)
                        special_classes.update(z_special_classes)

                # set global class filter
                classes_filter = self.class_handler.filter_objects([], list(objects_to_scan))

            else:  # new format - just definition of several fields
                objects_to_scan = set(objects_filter)
                filter_out = {}

                training_mode = msg.get("trainingMode", AnalyticTrainingMode.CUSTOMIZED.value)
                if training_mode == AnalyticTrainingMode.STATIC.value:  # reset and block training
                    logger.info("Training mode is static, will upload training images or use pf")
                    self.reset_pf({"blockTraining": True})
                else:  # don't reset pf, just block training if necessary or open locking
                    is_locked = training_mode == AnalyticTrainingMode.LOCKED.value
                    self.handle_future_pf_and_training(is_locked, not is_locked)

                if msg.get("forklift", True):
                    objects_to_scan.add(self.class_handler.vehicle_value)
                    special_classes.add(SpecialClasses.FORKLIFT.value)

                if msg.get("shoppingCart", False):
                    special_classes.add(SpecialClasses.SHOPPING_CART.value)

                if msg.get("container", False) and self.class_handler.container_value >= 0:
                    objects_to_scan.add(self.class_handler.container_value)
                    filter_out["containerId"] = Namespace(roi=None, roiFilter=False)

                # update roiFilter to be false on these objects
                for obj in objects_to_scan:
                    filter_out[obj] = Namespace(roi=None, roiFilter=False)

                if msg.get("protectiveGear", False):
                    filter_out["protectiveGear"] = Namespace(roi=None, roiFilter=False)
                if filter_out:
                    roi_filters.append(filter_out)

                if msg.get("specialLicensePlate", False):
                    # objects_to_scan.add(self.class_handler.vehicle_value)
                    filter_out["lpc"] = Namespace(roi=None, roiFilter=False)

                self.alert_manager.set_advance_fall(msg.get("advancedFallDetection", False))
                if filter_out:
                    roi_filters.append(filter_out)

            if len(special_classes) > 0:
                supported = []
                for c in special_classes:
                    if c in self.class_handler.classes:
                        c_val = self.class_handler.class_str_to_int(c)
                        classes_filter.append(c_val)
                        supported.append(c_val)
                    else:
                        logger.error(f"Illegal special class {c} in search message - not supported by detector")
                self.detector.set_special_classes(supported)

            if "privacy" in msg:
                privacy_dict = apply_from_dict_safe(msg, "privacy", {})
                privacy_objects = apply_from_dict_safe(privacy_dict, "objectsToPixelate", [])
                sub_classes_to_pixelate = apply_from_dict_safe(privacy_dict, "subClassesToPixelate", [])
                marked_idxs = apply_from_dict_safe(privacy_dict, "markedIdx", [])
                zones = apply_from_dict_safe(privacy_dict, "zones", {})
                polygons = []

                if zones:
                    try:  # parsing polygons
                        for zone in zones.values():
                            poly = [[p["x"], p["y"]] for p in zone["selection"]]
                            polygons.append(poly)
                    except Exception as e:
                        logger.warning(f"could not parse privacy zones: {str(e)}")
                        polygons.clear()
                privacy_regs = gen_privacy_regions(marked_idxs, polygons, thumb_size)
                # pixelate_bboxes = get_pixelate_coordinates(marked_idxs)
                privacy_type = PrivacyType(apply_from_dict_safe(privacy_dict, "privacyType", PrivacyType.PIXELATE))
            else:
                # keep legacy logic....
                privacy_objects = apply_from_dict_safe(msg, "objectsToPixelate", [])
                sub_classes_to_pixelate = apply_from_dict_safe(msg, "subClassesToPixelate", [])

            privacy_filter = self.class_handler.filter_objects(sub_classes_to_pixelate, privacy_objects)
        else:
            g_roi_filter = {}
            # set roi filter
            for obj in objects_filter:
                g_roi_filter[obj] = Namespace(roi=None, roiFilter=False)
            roi_filters.append(g_roi_filter)
        pixelate = privacy_type == PrivacyType.PIXELATE

        logger.info(f"Search filter running with classes: {classes_filter}")
        logger.info(f"Roi filter : {roi_filters}")

        logger.info(f"Pixelate classes: {privacy_filter}")
        logger.info(f"Privacy type {privacy_type.value}")

        self.entity_db.set_filters(roi_filters, classes_filter)
        self.md_manager.set_pixelate_filters(privacy_regs, privacy_filter, pixelate, thumb_size)

    def init_search_and_alerts(self):
        policies_dict = self.camera_document.get("armPolicies", {})
        zone_dict = self.camera_document.get("controlZone", {})

        self.alert_manager = AlertManager(
            init_dict=self.camera_document["alertPolicy"],
            policies_dict=policies_dict,
            zone_dict=zone_dict,
            config=self.config,
            entity_db=self.entity_db,
            l1_manager=self.l1_manager,
            context=self,
        )
        self.alert_status = self.alert_manager.alert_status
        self._adjust_trackers_to_alerts()

        self.md_manager = MetadataManager(self)

        search_msg = self.camera_document.get("searchPolicy", {})
        self.update_search(search_msg)

        biometric_msg = apply_from_dict_safe(search_msg, "biometricPolicy", {})
        self.update_biometric(biometric_msg)

        self.update_zone_regions(self.zone_document)

    def update_l1_config(self):

        # set l1 attributes
        class_filters = self.entity_db.class_filter
        roi_filter = self.entity_db.roi_filter

        pa_attr = Namespace(enable=False, filter=Namespace(ppe=[]))
        lpr_attr = Namespace(enable=False, filter=Namespace(mmc=[], plate=[], union=[]))
        lpc_attr = Namespace(enable=False, filter=Namespace(plate=[], union=[], lpc=[]))
        cont_attr = Namespace(enable=False, filter=Namespace(containerId=[]))
        rec_config = self.config.get("alertConfig", {}).get("recognition", {})
        use_external_apis = self.config.get("l1_models", {}).get("use_external_apis", False)
        use_external_lpr = rec_config.get("useALPR", True) or use_external_apis
        # Person attributes on search
        if self.class_handler.person_value in class_filters:
            pa_attr.enable = True

        # LPC on search
        if self.class_handler.plate_subclass_value in class_filters:
            lpc_attr.enable = True

        lpr_filters = ["plate", "mmc"]
        for r_filter in roi_filter:

            # look for vehicle
            for lpr_f in lpr_filters:
                if lpr_f in r_filter:
                    getattr(lpr_attr.filter, lpr_f).append(r_filter[lpr_f])
                    lpr_attr.enable = True
            # look for protective Gear
            if "protectiveGear" in r_filter:
                pa_attr.enable = True
                pa_attr.filter.ppe.append(
                    Namespace(roi=r_filter["protectiveGear"].roi, roiFilter=r_filter["protectiveGear"].roiFilter)
                )
            if "containerId" in r_filter:
                cont_attr.enable = True
                cont_attr.filter.containerId.append(
                    Namespace(roi=r_filter["containerId"].roi, roiFilter=r_filter["containerId"].roiFilter)
                )
            if "lpc" in r_filter:
                lpc_attr.enable = True
                lpc_attr.filter.lpc.append(
                    Namespace(roi=r_filter["lpc"].roi, roiFilter=r_filter["lpc"].roiFilter)
                )  # TODO: check if needed

        special_classes = set()
        # look for lpr alert
        for alert in self.alert_manager._alerts.values():
            if alert.enabled:
                if alert.alert_type == AlertType.lpr:
                    lpr_attr.enable = use_external_lpr
                    lpr_attr.filter.plate.append(Namespace(roi=alert.roi, roiFilter=alert.roiFilter))
                elif alert.alert_type == AlertType.containerId:
                    cont_attr.enable = True
                    cont_attr.filter.containerId.append(Namespace(roi=alert.roi, roiFilter=alert.roiFilter))

                vehicle_obj = self.class_handler.vehicle_value
                vehicle_filters = set(alert.filters.get(vehicle_obj, []))
                if len(vehicle_filters.intersection(VEHICLE_FILTERS)) > 0:
                    lpr_attr.enable = use_external_lpr
                    lpr_attr.filter.mmc.append(Namespace(roi=alert.roi, roiFilter=alert.roiFilter))
                if "type" in vehicle_filters and SpecialClasses.FORKLIFT.value in alert.filters[vehicle_obj]["type"]:
                    special_classes.add(SpecialClasses.FORKLIFT.value)
                recog_filters = alert.recognition_filters.get(vehicle_obj, {})
                if (
                    recog_filters.get("enabled", False)
                    and recog_filters["l1_analyzer"] == AdvanceAnalyzerType.ALPR.value
                ):
                    lpr_attr.enable = True
                    lpr_attr.filter.plate.append(Namespace(roi=alert.roi, roiFilter=alert.roiFilter))

                if SpecialClasses.SHOPPING_CART.value in alert.objects:
                    special_classes.add(SpecialClasses.SHOPPING_CART.value)

                person_filters = set(alert.filters.get(self.class_handler.person_value, []))
                if len(person_filters.intersection(PERSON_FILTERS)) > 0:
                    pa_attr.enable = True

                ppe_filters = person_filters.intersection(PPE_FILTERS)
                if len(ppe_filters) > 0:
                    has_hard_hat = False
                    for filt in ppe_filters:
                        if any(["hard_hat" in att for att in alert.filters[self.class_handler.person_value][filt]]):
                            has_hard_hat = True
                            break
                    pa_attr.enable = True
                    if has_hard_hat:
                        pa_attr.filter.ppe.append(Namespace(roi=alert.roi, roiFilter=alert.roiFilter))

                if alert.alert_type == AlertType.lpc:
                    lpc_attr.enable = True
                    lpc_attr.filter.lpc.append(Namespace(roi=alert.roi, roiFilter=alert.roiFilter))
                    lpc_attr.filter.plate.append(Namespace(roi=alert.roi, roiFilter=alert.roiFilter))

        if special_classes:
            self.detector.set_special_classes([self.class_handler.class_str_to_int(c) for c in special_classes])

        filtered_items = (
            lpr_attr.filter.plate,
            lpr_attr.filter.mmc,
        )
        lpr_attr.filter.union = [item for sublist in filtered_items for item in sublist]

        def merge_roi_filters(attr):
            if not attr:
                return attr
            filter_flat = [False] * (ROI_SHAPE[0] * ROI_SHAPE[1])
            for filter in attr:
                if not filter.roiFilter:
                    filter_flat = [True] * (ROI_SHAPE[0] * ROI_SHAPE[1])
                    break
                else:
                    for i in range(len(filter.roi)):
                        for j in range(len(filter.roi[i])):
                            index = i * len(filter.roi[0]) + j
                            filter_flat[index] = filter_flat[index] or filter.roi[i][j]
            return filter_flat

        # change from 32x32 to arr of 1024, so ROI check will be fast with batch.LOCATION
        lpr_attr.filter.plate = merge_roi_filters(lpr_attr.filter.plate)
        lpr_attr.filter.mmc = merge_roi_filters(lpr_attr.filter.mmc)

        lpc_attr.filter.plate = merge_roi_filters(lpc_attr.filter.plate)
        lpc_attr.filter.lpc = merge_roi_filters(lpc_attr.filter.lpc)

        pa_attr.filter.ppe = merge_roi_filters(pa_attr.filter.ppe)
        cont_attr.filter.containerId = merge_roi_filters(cont_attr.filter.containerId)

        logger.info(f"LPR Attributes Enabled :{lpr_attr.enable}")
        self.l1_manager.update_l1_config(AdvanceAnalyzerType.ALPR, lpr_attr.enable, lpr_attr.filter)

        if use_external_apis:
            lpc_attr.enable = False
            logger.info("LPC Attributes DISABLED due to usage of external apis")
            self.l1_manager.update_l1_config(AdvanceAnalyzerType.LPREC, lpc_attr.enable, lpc_attr.filter)
        else:
            logger.info(f"LPC Attributes Enabled :{lpc_attr.enable}")
            self.l1_manager.update_l1_config(AdvanceAnalyzerType.LPREC, lpc_attr.enable, lpc_attr.filter)

        logger.info(f"PA Attributes Enabled :{pa_attr.enable}")
        self.l1_manager.update_l1_config(AdvanceAnalyzerType.HUMAN_PARSING, pa_attr.enable, pa_attr.filter)

        logger.info(f"Container Attributes Enabled :{cont_attr.enable}")
        self.l1_manager.update_l1_config(AdvanceAnalyzerType.CONTAINER, cont_attr.enable, cont_attr.filter)

        # weapons
        weapons_en = False
        if "weapon" in self.class_handler.objects:
            weapon_value = self.class_handler.object_str_to_int("weapon")
            for c in class_filters:
                if self.class_handler.get_object_from_class(c) == weapon_value:
                    weapons_en = True
                    break
            if not weapons_en:
                for alert in self.alert_manager._alerts.values():
                    if alert.object_based_alert and "weapon" in alert.objects:
                        weapons_en = True
                        break

        logger.info(f"Weapons Attributes Enabled :{weapons_en}")
        self.l1_manager.update_l1_config(AdvanceAnalyzerType.WEAPONS, weapons_en, None)

    def _adjust_trackers_to_alerts(self):
        is_required = self.alert_manager.is_required_advance_tracking()
        if is_required and self.tracker.name != TrackerType.BYTESREID.value:
            logger.info("Switching to BytesReid tracker")
            track_config = self.config["l0_track"]
            # Tracker
            fps = int(self.app_config["streamer"]["analyticFramerate"])
            self.tracker = TrackerFactory().create(TrackerType.BYTESREID.value, track_config, fps, self)

    def get_traffic_alert_controllers(self):
        return self.alert_manager.get_traffic_alert_controllers()

    def update(self, msg):
        with self._batch_lock:
            if msg["msgAction"] == ManagementMessage.UPDATE_ALERT.value:
                self.alert_manager.parse_alerts_msg(msg)
                self._adjust_trackers_to_alerts()
                # after updating the alert, make sure the status is up to date
                self.alert_status = self.alert_manager.alert_status
            elif msg["msgAction"] == ManagementMessage.EXTERNAL_EVENT_VALIDATION.value:
                logger.info(f"Received external event validation message: {msg}")
                self.alert_manager.validate_external_event(msg)

            elif msg["msgAction"] == ManagementMessage.ARM_POLICY.value:
                self.alert_manager.parse_policy_msg(msg)

            elif msg["msgAction"] == ManagementMessage.UPDATE_CONTROL_ZONE.value:
                self.alert_manager.parse_zone_msg(msg)

            elif msg["msgAction"] == ManagementMessage.UPDATE_SEARCH.value:
                self.update_search(msg["search"])
                self.camera_document["searchPolicy"] = msg["search"]

            elif msg["msgAction"] == ManagementMessage.UPDATE_CUSTOM_OBJECT.value:
                self.class_handler.update_custom_object(msg)

            elif msg["msgAction"] == ManagementMessage.UPDATE_ANALYTIC_ASSET.value:
                self.read_from_db()
            elif msg["msgAction"] == ManagementMessage.UPDATE_ZONE_REGIONS.value:
                self.update_zone_regions(msg["zoneRegions"])
            else:
                logger.error(f"Non supported update: {msg}")
            self.update_l1_config()

    def update_zone_regions(self, zones: List[Dict]):
        for zone in zones:
            if zone["data"].get("countingEnabled", False):
                if zone["data"]["cameraId"] == self.camera_id:
                    region = ZoneRegion(zone)
                    self.zone_regions[region.id] = region
                else:
                    logger.warning(
                        f"Zone {zone.get('_id','unknown id')} does not belong to camera {self.camera_id}, skipping."
                    )
        if self.zone_regions:
            self.alert_manager.handle_zone_regions(self.zone_regions)

    # @try_except_raise(logger=logger, raise_exception=False, return_on_exception=(None, None))
    def process_image(self, message_count, image: AnalyticImage):
        # fill batch and send to work if batch is full
        # update input timestamp
        self.input_frames_timestamp.append(image.timestamp)
        skips = 0
        if not self.file_processing:
            if self.image_idx <= self.slow_start_n_frames:
                skips = self.slow_start_skips
            elif message_count > self.app_config["streamer"]["skipTwoAnalyticFiles"]:
                skips = 2
            elif message_count > self.app_config["streamer"]["skipOneAnalyticFiles"]:
                skips = 1
        illegal_frame = False
        if self.config["pre_process"]["checkImageSize"]:
            size_in_bytes = len(image.frame)
            expected_size = self.resolution[0] * self.resolution[1] * 1.5
            if expected_size != size_in_bytes:
                logger.error(f"Got illegal frame size. Got:{size_in_bytes} Bytes vs {expected_size} expected")
                illegal_frame = True

        if (self.image_idx > self.last_batch_image_idx + skips) and not illegal_frame:
            self.image_batch.append(image)
            self.processed_frames_timestamp.append(image.timestamp)
            self.last_batch_image_idx = self.image_idx
        else:
            if self.warn_on_skip:
                logger.warning(
                    f"Skipped frame. edgeID:{self.edge_id} cameraId:{self.camera_id} Timestamp:{image.timestamp}"
                )
            self._skipped_frames.append(image.timestamp)
        self.image_idx += 1

        analytic_results = Namespace(metadata=None, motion_info=None, counting_info=None, l1Stats={})
        if len(self.image_batch) >= self.batch_size:
            try:
                with self._batch_lock:
                    analytic_results = deepcopy(self._process_batch())
                logger.debug("clearing buffers")
                analytic_results = self.update_statistic(self.image_batch[0].timestamp, analytic_results)
            finally:
                self.image_batch.clear()
                self._skipped_frames.clear()
        return analytic_results

    @property
    def last_batch_data(self):
        return self._last_batch_data

    def update_night_mode(self, timestamps: List[int], detector_images: List[np.array]):
        curr_per = timestamps[-1] // self.night_mode_period_ms
        if curr_per > self.night_mode_last_per:
            is_monochrome = is_image_monochrome(detector_images[-1])
            if is_monochrome != self.is_night_mode:
                logger.info(f"analyzer {'entered' if is_monochrome else 'exited'} night mode")
                self.is_night_mode = is_monochrome
                self.on_night_mode_changed(is_monochrome)
            self.night_mode_last_per = curr_per

    def update_resolution(self, resolution_msg: Dict[str, str]):
        resolution = [
            int(resolution_msg["width"]),
            int(resolution_msg["height"]),
        ]

        if 0 in resolution:
            raise ValueError("Illegal resolution")
        else:
            self.resolution = resolution
        if self.preprocessor:
            # regenerate new processor
            del self.preprocessor
            self._init_preprocessor()

    def _init_preprocessor(self):
        preprocessor_config = self.config.get("preprocess", {})
        preprocessor_name = preprocessor_config.get("name", "lumana")  # "cv2" "lumana"
        logger.info(f"Using preprocessor {preprocessor_name}")
        self.preprocessor = PreprocessorFactory().create(
            preprocessor_name, preprocessor_config, self.resolution, self.detector
        )

    def clear(self):
        self.image_batch.clear()

    def add_model(self, models: List[str] = None):
        if models is None:
            all_weights = self.weights_lookup()
            all_files = {k: Path(v).stem for k, v in all_weights.items()}
            custom_weights = {}
            for key in all_files:
                if key not in self.registered_models or self.registered_models[key] != all_files[key]:
                    custom_weights[key] = all_weights[key]
            if len(custom_weights) == 0:
                return
        else:
            custom_weights = self.weights_lookup(models)

        # TODO: handle config file if exists - not operational
        if self.local_inference:
            self._load_weights_locally(custom_weights)
        else:
            self._load_weights_to_triton(custom_weights)

        for key in custom_weights:
            if key == "l0_detect":
                # update detector and its dependencies
                self.replace_detector()
            elif key == "l0_track":
                pass  # currently not used
            elif key in ["pa", "reid", "ppe"]:
                self.l1_manager.update_model("human_parsing", self.config)
            elif key == "doors":
                self.alert_manager.update_model(key, self.config["l1_models"]["attributes"]["doors"])
            elif key == "wc":
                self.alert_manager.update_model(key, self.config["l1_models"]["attributes"]["weapons_classification"])
                self.l1_manager.update_model("weapons_classification", self.config)
            elif "state-object" in key:
                self.entity_db.update_model("state-object", key)
                pass
            else:
                logger.warning(f"model {key} is not supported")
        self.registered_models.update({k: Path(v).stem for k, v in custom_weights.items()})

    def replace_detector(self):
        preproc_changed = False
        detector_config = self.config["l0_detect"]
        self.detector = DetectorFactory().create(detector_config["name"], detector_config)
        self.class_handler = self.detector.class_handler
        if self.batch_size != self.detector.batch_size:
            self.batch_size = self.detector.batch_size
            preproc_changed = True
        detector_res = self.detector.input_size
        if detector_res != self.detector_resolution:
            self.detector_resolution = detector_res
            self.load_balancer.update_resolution(detector_res)
            self.tracker.set_image_size(detector_res)
            preproc_changed = True
        if preproc_changed or (self.detector.is_local and self.detector.args.is_trt):
            # regenerate new processor
            del self.preprocessor
            self._init_preprocessor()

    def reset_pf(self, msg: dict = None):
        if msg is None:
            msg = {}

        # unload model from triton and remove the files
        if not self.detector.is_local and self.camera_id in self.detector.args.name:
            unload_triton_model(self.detector.remote_info)
            model_path = os.path.join(proj.assets_path(), "model_repository", self.detector.remote_info.model_name)
            try:
                shutil.rmtree(model_path)
                logger.info(f"Removed model path {model_path} for {self.camera_id}")
            except Exception as e:
                logger.error(f"Failed to remove model path {model_path} for {self.camera_id}. Reason: {e}")

        # roll back detector to factory default
        default_config = proj.load_configAnalytic()
        self.config["l0_detect"] = default_config["l0_detect"]
        self.config["l0_track"]["detector_unique"] = False
        self.replace_detector()

        # self.registered_models.
        cam_assets_path = os.path.join(proj.assets_path(), self.edge_id, self.camera_id)

        if not os.path.exists(cam_assets_path):
            logger.warning(f"Camera assets path {cam_assets_path} does not exist, cannot reset PF")
        else:
            detector_options = ["yolo", "detector"]  # currently only yolo variants are used
            for opt in detector_options:
                files_to_remove = glob.glob(os.path.join(cam_assets_path, f"{opt}*"))
                for file in files_to_remove:
                    try:
                        os.unlink(file)
                        logger.info(f"Removed file {file} from camera assets path")
                    except Exception as e:
                        logger.error(f"Failed to delete {file}. Reason: {e}")
                        logger.info("Trying to rename the file instead")
                        new_name = file + ".bak"
                        try:
                            os.rename(file, new_name)
                            logger.info(f"Renamed {file} to {new_name}")
                        except Exception as e:
                            logger.error(f"Failed to rename {file} to {new_name}. Reason: {e}")

        to_block = msg.get("blockTraining", True)
        self.handle_future_pf_and_training(to_block, True)

    def handle_future_pf_and_training(self, to_block: bool, handle_nopf_flag: bool = True):
        cam_assets_path = os.path.join(proj.assets_path(), self.edge_id, self.camera_id)
        logger.info(f"Training selection is {'' if to_block else 'not '}blocked for {self.camera_id}")
        self.md_manager.training_selector.reset_training_selection(to_block)
        if handle_nopf_flag:
            no_pf_file = os.path.join(cam_assets_path, "no.pf")
            if to_block:
                with open(no_pf_file, "w"):
                    pass
            else:
                if os.path.exists(no_pf_file):
                    try:
                        os.remove(no_pf_file)
                        logger.info(f"Removed no.pf file for {self.camera_id}")
                    except Exception as e:
                        logger.error(f"Failed to remove no.pf file for {self.camera_id}. Reason: {e}")

    def read_from_db(self):
        db_files = sorted(glob.glob(os.path.join(proj.assets_path(), self.edge_id, "*.db")), key=os.path.getctime)
        # prioritize analytic.db to be loaded first. other DB will override its data
        db_files.sort(key=lambda f: 0 if os.path.basename(f) == "analytic.db" else 1)

        is_healthy = True
        analytic_db = {}

        if db_files:
            for db_file in db_files:
                try:
                    with SQLiteReader(db_file) as reader:
                        tables = reader.fetch_all("SELECT name FROM sqlite_master WHERE type='table'", model=dict)
                        tables = set([t["name"] for t in tables])
                        for table_name in tables:
                            if "sqlite" in table_name:
                                continue
                            if table_name not in SQLITE_KNOW_TABLES:
                                logger.warning(f"DB READER: Table {table_name} unrecognized, skipping")
                                continue
                            analytic_db[table_name] = reader.fetch_all(
                                f"SELECT * from {table_name}", SQLITE_KNOW_TABLES[table_name]
                            )
                            logger.info(f"Loaded table {table_name} from db file {db_file}")
                except Exception as e:
                    log_exception(logger, f"Error loading table from db file {db_file}", e)
                    is_healthy = False
        else:
            logger.warning(f"No .db files found in {os.path.join(proj.assets_path(), self.edge_id)}")
            is_healthy = False

        # report missing tables - its ok to have them as the org might not define everything
        tables_found = set(analytic_db.keys())
        tables_known = set(SQLITE_KNOW_TABLES.keys())
        not_found = tables_known - tables_found
        if not_found:
            logger.warning(f"DB READER:Some known tables were not found in the loaded DB files: {not_found}")

        # try to load as many tables as possible, but if unhealthy report it
        if analytic_db:
            try:
                self.analytic_db.update(
                    generate_unified_table(analytic_db, self.class_handler, self.camera_id, person_db_enabled=True)
                )
            except Exception as e:
                log_exception(logger, "Error generating unified table from analytic db", e)
                is_healthy = False
        self.synced_status = is_healthy
        self.on_db_update()

    @property
    def bad_alerts(self) -> List[Dict[str, str]]:
        return [{k: trim_str(v)} for k, v in self.alert_manager.bad_alerts.items()]


def gen_privacy_regions(indices, polygons: list = None, thumb_size=(384, 640)) -> List[Dict]:
    privacy_regs = []
    fallback = True
    if polygons:  # first, attempt to load the polygons themselves to create the mask
        for polygon in polygons:
            if Polygon(polygon).is_valid:  # a valid polygon is a polygon that doesn't intersect itself
                poly = np.array(polygon) * np.array(thumb_size)
                bbox = np.append(np.floor(np.min(poly, axis=0)), np.ceil(np.max(poly, axis=0))).astype(int)
                mask = np.full([bbox[3] - bbox[1], bbox[2] - bbox[0]], 1, np.uint8)

                cv2.fillPoly(mask, [(poly - bbox[0:2]).astype(int)], 0)
                privacy_regs.append({"bbox": bbox, "mask": mask[:, :, np.newaxis]})
            else:  # if it's not a valid polygon, there is no need fall back to marked indexes
                break
        fallback = False
    if fallback:  # means the polygons are not usable for privacy
        if indices:
            roi = np.zeros(ROI_SHAPE, dtype=np.uint8)
            rows, cols = np.unravel_index(indices, ROI_SHAPE)
            roi[rows, cols] = 1
            mask = (cv2.resize(roi, thumb_size, interpolation=cv2.INTER_NEAREST)).astype(np.uint8)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)  # Get the bounding box
                bbox = [x, y, x + w, y + h]
                privacy_regs.append({"bbox": bbox, "mask": 1 - mask[y : y + h, x : x + w, np.newaxis]})

    return privacy_regs
