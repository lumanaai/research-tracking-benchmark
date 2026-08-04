import json
import os
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

from general.proj import assets_path
from general.img_utils import sharpness_score_batch, brightness_score_batch
from general.analyzer_general import InferenceType
from general.inference import InferenceWrapper, BaseInferenceConfig
from collections import deque
from sklearn.mixture import GaussianMixture
from general.analyzer_general import logger, log_exception

REGULAR_STATE = 0
OVERSAMPLING_STATE = 1
TRANSITION_STATE = 2

@dataclass
class MonitorState:
    state: int = 0
    start_time: int = 0
    # special flag for oversampling during transition state
    transition_oversampling: bool = False
    oversampling_start_time: int = 0


class IQConfig(BaseInferenceConfig):
    name: InferenceType = InferenceType.IMAGEQUALITY  # name of the inference engine, should be InferenceType
    model_name: str = "resnet_18"  # model name, can be resnet_50 or resnet_18
    im_size: Tuple[int, int] = (192, 320)  # input image size
    # im_size: Tuple[int, int] = (384, 640)  # input image size
    weights: str = f"image_quality_2cls_emb_{str(im_size[0])}x{str(im_size[1])}.pt"
    mean: List[float] = [0.485, 0.456, 0.406]  # normalizing factors
    std: List[float] = [0.229, 0.224, 0.225]  # normalizing factors
    target_mean: Dict[str, float] = {"arniqa": 0, "ilniqe": 23}  # target mean for each class
    target_std: Dict[str, float] = {"arniqa": 1, "ilniqe": 6}  # target std for each class
    classes: List[str] = ["arniqa", "ilniqe"]  # class names
    num_classes: int = 2  # number of classes for classification
    half = True


class ImageQualityNetMetrics(InferenceWrapper):
    _config_type = IQConfig
    args: IQConfig
    output_names = ["output", "embedding"]  # setting 2 output bindings for trt engine

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super(ImageQualityNetMetrics, self).__init__(msg_dict, is_local=is_local)
        self.mean = np.array([self.args.target_mean[c] for c in self.args.classes])
        self.std = np.array([self.args.target_std[c] for c in self.args.classes])
        self.target_scale = np.array(
            [100, -1]
        )  # Invert ILNIQE scores so all metrics are in the same direction higher is better

    def build_trt_model(self):
        super().build_trt_model()
        self._fix_output_order()

    def build_triton_model(self):
        super().build_triton_model()
        self._fix_output_order()

    def build_full_model(self):
        import torch
        import torchvision.models as models

        class _IQExportWrapper(torch.nn.Module):
            """Wraps a ResNet so that forward() returns (scores, embedding).

            The embedding is the flattened avgpool output.  This makes the ONNX
            graph (and therefore the TRT engine) expose it as a second output,
            so we don't need a runtime hook.
            """

            def __init__(self, backbone: torch.nn.Module):
                super().__init__()
                self.backbone = backbone

            def forward(self, x):
                # replicate ResNet forward but capture avgpool
                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)
                emb = self.backbone.avgpool(x)          # (B, C, 1, 1)
                emb_flat = torch.flatten(emb, 1)        # (B, C)
                scores = self.backbone.fc(emb_flat)      # (B, num_classes)
                return scores, emb_flat

        if self.args.model_name == "resnet_18":
            backbone = models.resnet18()
        elif self.args.model_name == "resnet_50":
            backbone = models.resnet50()
        else:
            raise ValueError(f"Unknown model name: {self.args.model_name}")

        backbone.fc = torch.nn.Linear(backbone.fc.in_features, self.args.num_classes)

        backbone.to(self.args.device)
        checkpoint = torch.load(self.args.weights)
        checkpoint = {k.replace("model.", ""): v for k, v in checkpoint.items()}
        backbone.load_state_dict(checkpoint)
        if self.args.half:
            backbone = backbone.half()
        backbone.eval()

        self.model = _IQExportWrapper(backbone)
        self.model.eval()

    def infer_full(self, crops):
        import torch

        crop_stack = torch.tensor(np.stack(crops, axis=0)).to(self.args.device)
        with torch.no_grad():
            scores, emb = self.model(crop_stack)
        return scores.cpu().numpy(), emb.cpu().numpy()

    def post_infer(self, outputs, inputs):
        # outputs is a tuple: (scores_array, embedding_array)
        scores, emb = outputs
        if scores.ndim == 1:
            scores = scores[None, :]
        elif scores.ndim != 2:
            raise ValueError(f"Unexpected output shape: {scores.shape}")

        # unnormalize and scale
        scores = (scores * self.std + self.mean) * self.target_scale
        # squeeze embedding to 1-D for single-image inference (matches old hook behavior)
        if emb is not None:
            emb = np.squeeze(emb)
        return scores, emb


class ImageQualityMonitor:
    """
    The following class is responsible for monitoring image quality degradation.
    The monitor calculates 4 image quality metrics (brightness, sharpness, ILNIQE, ARNIQA)
    once every `image_quality_period_ms` and aggregates them in `_metrics_stack` until it
    accumulates measurements. It then checks if there are extreme deviations.
    In addition, the monitor tracks long-term deviations from optimal image quality.
    The monitor can normalize quality for both dual mode scene (day \ night) and mono mode scene.

    Inputs:
        images: List[np.ndarray]
        timestamps: List[int]

    Output:
        bool
    """

    accumulation_period: int = 288  # number of frames to accumulate before checking image quality
    q: float = 0.2  # quantile for threshold calculation
    decision_p_th: float = 0.5  # which degraded portion of the batch is considered 'True'
    th_bias: List[float] = [5, 5, 5, 10]  # brightness, sharpness, ilniqe, arniqa
    img_q_period_ms: int = 20 * 60 * 1000  # IQ monitoring period: 20 minutes in milliseconds (3 frames per hour)
    oversample_factor: int = 20  # factor to oversample monitoring when degradation is detected
    min_sampling_period_ms: int = 1 * 60 * 1000  # minimal sampling period: 1 minute in milliseconds
    img_q_last_per: int = 0
    scene_mode_min_count: int = 10  # determines the minimal counts needed to seperate between 2 lighting conditions
    distrib_separation_const: int = 5  # minimal distance between 2 lighting Gauss distributions
    long_term_dev_th: float = 0.4  # controls sensitivity to deviation from optimal image quality
    hist_save_per: int = 288  # persistent history save period (in frames)
    batch_size = 4
    debounce_mode: str = "majority"  # debouncing mode: "simple" or "majority"
    debounce_window_size: int = 5  # only for majority debounce mode: size of the sliding window
    min_consecutive_count = 3  # only for simple debounce mode: number of consecutive degraded / non-degraded results to change the anchor flag
    transition_phase: int = 6 * 60 * 60 * 1000  # examination period for turnning from short-term to long-term degradation (6 hours)
    embed_hist_period: int = 144  # number of embeddings to keep in history (72 hours with 3 samples/hour)

    def __init__(
        self, msg_dict: Dict, context, is_local: Optional[bool] = None, load_prev: bool = True
    ):
        self.enabled = msg_dict.get("enable", True)  # enable or disable the monitor
        if not self.enabled:
            return
        self.sampling_period = self.img_q_period_ms
        self.img_q_last_stack_per = 0                   # track stack period separately
        self.anchor_flag = False                        # last IQ degradation flag
        self.consecutive_count = 0                      # only for simple debounce mode: count of consecutive same results
        self.debounce_window = deque([False] * self.debounce_window_size, maxlen=self.debounce_window_size)  # only for median debounce mode: sliding window of last results
        self.monitor_state = MonitorState()             # track procedure case and timestamp. case 0: regular sampling, case 1: oversampling for short term degredation, case 2: regular sampling and long term degradation examination
        self.flag_type = 0                              # type of flag: 0 - no flag, 1 - short-term degradation, 2 - long-term degradation
        self.net_metrics = ImageQualityNetMetrics(msg_dict, is_local=is_local)
        self.context = context
        self.accumulation_period = msg_dict.get("accumulation_period", self.accumulation_period)
        self.next_accumulation_period = self.accumulation_period
        self.timestamps_history = deque(maxlen=self.accumulation_period)  # used for logging info
        self._metrics_stack = np.zeros((self.accumulation_period, 5), dtype=float)  # stack for metrics
        self.write_idx = 0
        self.current_len = 0
        self.mode_th_history = deque(
            maxlen=self.accumulation_period
        )  # deque of mode thresholds for day/night separation
        self.ths_daytime = np.full(len(self.th_bias), -np.inf)
        self.ths_nighttime = np.full(len(self.th_bias), -np.inf)
        self.top_ths_bright_mode: np.ndarray = np.full(4, -np.inf)  # best bright thresholds for reference
        self.top_ths_dark_mode: np.ndarray = np.full(4, -np.inf)  # best dark thresholds for reference
        self.history_file_path = os.path.join(
            assets_path(), self.context.edge_id, self.context.camera_id, "history_iqm.json"
        )
        if load_prev:
            self._load_history()

        # check parameters validity
        if self.transition_phase > self.accumulation_period * 20 * 60 * 1000:
            logger.exception("Transition phase must be shorter than accumulation period")

        # embedding history for tampering validation: 2 buckets (day/night), 72h each
        # 3 samples/hour × 72 hours = 216 embeddings per bucket
        self.emb_vec = None                                                             # latest embedding (numpy)
        self.prev_day_part = -1                                                         # previous day part
        self.embedding_history_day = deque(maxlen=self.embed_hist_period)               # non-degraded day (timestamp, embedding) tuples
        self.embedding_history_night = deque(maxlen=self.embed_hist_period)             # non-degraded night (timestamp, embedding) tuples
    
    def _save_history(self):
        hist_dict = {
            "metrics_stack": self._metrics_stack.tolist(),
            "write_idx": self.write_idx,
            "current_len": self.current_len,
            "timestamps_history": list(self.timestamps_history),
            "mode_th_history": list(self.mode_th_history),
            "ths_daytime": self.ths_daytime.tolist(),
            "ths_nighttime": self.ths_nighttime.tolist(),
            "top_ths_bright_mode": self.top_ths_bright_mode.tolist(),
            "top_ths_dark_mode": self.top_ths_dark_mode.tolist(),
            "embedding_history_day": [[ts, e.tolist()] for ts, e in self.embedding_history_day],
            "embedding_history_night": [[ts, e.tolist()] for ts, e in self.embedding_history_night]
        }
        try:
            with open(self.history_file_path + ".backup", "w") as f:
                json.dump(hist_dict, f)
            with open(self.history_file_path, "w") as f:
                json.dump(hist_dict, f)
        except Exception as e:
            log_exception(logger, "Error saving ImageQualityMonitor history", e)

    def _load_history(self):
        hist_files = [self.history_file_path, self.history_file_path + ".backup"]
        for file in hist_files:
            if os.path.exists(file):
                try:
                    with open(file, "r") as f:
                        hist_dict = json.load(f)
                    self._metrics_stack = np.array(hist_dict["metrics_stack"], dtype=float)
                    self.write_idx = hist_dict["write_idx"]
                    self.current_len = hist_dict["current_len"]
                    self.timestamps_history = deque(hist_dict["timestamps_history"], maxlen=self.accumulation_period)
                    self.mode_th_history = deque(hist_dict["mode_th_history"], maxlen=self.accumulation_period)
                    self.ths_daytime = np.array(hist_dict.get("ths_daytime", [-np.inf] * len(self.th_bias)))
                    self.ths_nighttime = np.array(hist_dict.get("ths_nighttime", [-np.inf] * len(self.th_bias)))
                    self.top_ths_bright_mode = np.array(hist_dict["top_ths_bright_mode"], dtype=float)
                    self.top_ths_dark_mode = np.array(hist_dict["top_ths_dark_mode"], dtype=float)
                    raw_day = hist_dict.get("embedding_history_day", [])
                    raw_night = hist_dict.get("embedding_history_night", [])
                    # support both legacy (bare array) and new [timestamp, array] formats
                    # new format: [ts, [v0, v1, ...]] — 2-element list where second is a list
                    # legacy format: [v0, v1, ...] — flat array of floats
                    def _parse_emb_entry(e):
                        if isinstance(e, list) and len(e) == 2 and isinstance(e[1], list):
                            v = np.array(e[1], dtype=float)
                        else:
                            v = np.array(e, dtype=float)
                        norm = np.linalg.norm(v)
                        if norm > 0:
                            v = v / norm
                        return (e[0] if isinstance(e, list) and len(e) == 2 else 0, v)

                    self.embedding_history_day = deque(
                        [_parse_emb_entry(e) for e in raw_day], maxlen=self.embed_hist_period
                    )
                    self.embedding_history_night = deque(
                        [_parse_emb_entry(e) for e in raw_night], maxlen=self.embed_hist_period
                    )
                    logger.info(f"Restored ImageQualityMonitor history from {file}")
                    break
                except Exception as e:
                    log_exception(logger, f"Error loading ImageQualityMonitor history file {file}", e)
                    # optionally: delete corrupt file

    def _compute_scores(self, images):
        net_scores = []
        for img in images:
            # self.emb_vec_prev = self.emb_vec
            scores, self.emb_vec = self.net_metrics.forward_on_crop_list([img])
            net_scores.append(scores)
        net_scores = np.squeeze(np.array(net_scores))
        if net_scores.ndim == 3:  # Shape (batch, 1, 2) -> (batch, 2)
            net_scores = np.squeeze(net_scores, axis=1)
        elif net_scores.ndim == 1:  # Shape (2,) -> (1, 2) for single image
            net_scores = net_scores.reshape(1, -1)
        bright_score = brightness_score_batch(images)
        sharp_score = sharpness_score_batch(images)
        return net_scores, bright_score, sharp_score

    def _calibrate_day_part(self):
        bright_score = self._metrics_stack[:, 2]  # bright_score is the 3rd column
        # Compute mode threshold (day-night \ bright-dark)
        brightness_values = bright_score.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(brightness_values)
        means = gmm.means_.flatten()
        labels = gmm.predict(brightness_values)  # just to check the how many elements in each component
        counts = np.bincount(labels)
        # Default: everything is day
        day_parts = np.ones_like(bright_score, dtype=int)
        # Check if separation is clear
        if abs(means[0] - means[1]) < self.distrib_separation_const or np.any(counts < self.scene_mode_min_count):
            # No clear seperation of 2 lighting mode: setting all to 'bright'
            self._metrics_stack[:, -1] = day_parts
            self.mode_th_history.append(None)  # calibration indicates mono lighting mode in the scene
            logger.info(
                f"No clear seperation of 2 lighting mode at {self.timestamps_history[-1]}. Setting all to 'bright'"
            )
        else:
            day_threshold = (means[0] + means[1]) / 2
            # Label as bright / day (1) or dark / night (0)
            day_parts = (bright_score > day_threshold).astype(int)
            # Save day_parts into the last history slot (assuming shape (N, M))
            self._metrics_stack[:, -1] = day_parts
            self.mode_th_history.append(day_threshold)

    def _determine_day_part(self, curr_batch):
        # check if last mode_th_history is None (mono mode)
        if self.mode_th_history and self.mode_th_history[-1] is not None:
            bright_score = curr_batch[:, 2]  # bright_score is the 3rd column
            # Use the last known threshold
            day_parts = (bright_score > self.mode_th_history[-1]).astype(int)
            curr_batch[:, -1] = day_parts
        else:
            curr_batch[:, -1] = 1  # default to day part if no history is available
        return curr_batch

    def _compute_thresholds(self, scores):
        ths = np.zeros(len(self.th_bias))
        for i, met_th in enumerate(self.th_bias):
            ths[i] = np.quantile(scores[:, i], self.q) - met_th
        return ths

    def _calibrate_ths(self):
        long_term_check_day = False
        long_term_check_night = False
        # divide the stack into day and night scores
        day_scores = self._metrics_stack[self._metrics_stack[:, -1] == 1]
        day_scores = day_scores[:, :-1]  # keep only scores, remove day/night label
        night_scores = self._metrics_stack[self._metrics_stack[:, -1] == 0]
        night_scores = night_scores[:, :-1]  # keep only scores, remove day/night label
        # calc thresholds for day and night, for each score
        if day_scores.shape[0] != 0:
            self.ths_daytime = self._compute_thresholds(day_scores)
            # save top ths
            self.top_ths_bright_mode = np.maximum(self.top_ths_bright_mode, self.ths_daytime)
            # check for long term degredation
            long_term_check_day = np.any(
                np.abs(self.top_ths_bright_mode - self.ths_daytime)
                > np.abs(self.top_ths_bright_mode) * self.long_term_dev_th
            )
            if long_term_check_day:
                logger.error(
                    f"Long term degradation detected at {self.timestamps_history[-1]}: "
                    f"current day thresholds: {self.ths_daytime}, top thresholds: {self.top_ths_bright_mode}"
                )
        if night_scores.shape[0] != 0:
            self.ths_nighttime = self._compute_thresholds(night_scores)
            # save top ths
            self.top_ths_dark_mode = np.maximum(self.top_ths_dark_mode, self.ths_nighttime)
            # check for long term degredation
            long_term_check_night = np.any(
                np.abs(self.top_ths_dark_mode - self.ths_nighttime)
                > np.abs(self.top_ths_dark_mode) * self.long_term_dev_th
            )
            if long_term_check_night:
                logger.error(
                    f"Long term degradation detected at {self.timestamps_history[-1]}: "
                    f"current night thresholds: {self.ths_nighttime}, top thresholds: {self.top_ths_dark_mode}"
                )
        # save history every hist_save_per frames
        if self.current_len % self.hist_save_per == 0:
            self._save_history()

    def _debounce_mec(self, result, timestamp):
        if self.debounce_mode == "majority":
            ## debouncing mechanism using majority vote
            self.debounce_window.append(result)
            votes = sum(self.debounce_window)
            majority_res = votes > (len(self.debounce_window) / 2)
            if majority_res != self.anchor_flag:
                logger.info(f"IQ degradation flag ({majority_res}) differs from anchor ({self.anchor_flag}) at {timestamp}")
                self.anchor_flag = majority_res
            return self.anchor_flag
        else:
            ## simple mechanism for consecutive results
            # be advised: makes flagging delayed by min_consecutive_count
            if result == self.anchor_flag:
                # no change
                self.consecutive_count = 0
            else:
                logger.info(f"IQ degradation flag ({result}) differs from anchor ({self.anchor_flag}) at {timestamp}")
                self.consecutive_count += 1
                if self.consecutive_count >= self.min_consecutive_count:
                    # enough consecutive results, change anchor
                    self.anchor_flag = result
                    self.consecutive_count = 0
            return self.anchor_flag

    def _handle_regular_state(self, timestamp):
        # enter short term degradation examination phase immediately when flag is raised (result == True)
        self.monitor_state = MonitorState(OVERSAMPLING_STATE, timestamp)
        self.sampling_period = max(self.min_sampling_period_ms, self.img_q_period_ms // self.oversample_factor)
        self.consecutive_count = 0  # reset for oversampling debouncing
        logger.info(
            f"Entering short-term IQ degradation phase at {timestamp}. Oversampling activated."
            f" Sampling period was changed to {self.sampling_period}ms"
            )

    def _handle_oversampling_state(self, raw_res, timestamp, debounced_res):
        if timestamp - self.monitor_state.start_time >= self.img_q_period_ms:
            # stop oversampling degraded images and enter long-term degradation examination phase
            self.monitor_state = MonitorState(TRANSITION_STATE, timestamp)
            self.sampling_period = self.img_q_period_ms
            self.flag_type = 1      # make sure we're on short-term degradation
            logger.info(
                f"Entering long-term IQ degradation examination phase at {timestamp}."
                f" Oversampling was deactivated to {self.sampling_period}ms")
        elif self.consecutive_count == 0:
            if debounced_res:
                if self.monitor_state.transition_oversampling:
                    # a single non-degraded result turned out to be noise during transition state
                    self.monitor_state = MonitorState(TRANSITION_STATE, self.monitor_state.oversampling_start_time)  # restore original start time of transition state
                    self.sampling_period = self.img_q_period_ms
                    self.monitor_state.transition_oversampling = False     # reset transition oversampling flag
                    self.monitor_state.oversampling_start_time = 0
                    logger.info(f"Exiting oversampling mid-transition state at {timestamp}."
                                f" Resuming long-term IQ degradation examination. Sampling period set to {self.sampling_period}ms")
                else:
                    # image is degraded after oversampled debounced check
                    self.flag_type = 1  # short-term degradation
                    logger.info(f"Raising short-term IQ degradation flag at {timestamp} (raw={raw_res}, debounced={debounced_res}).")
            else:
                # image is non-degraded after oversampled debounced check
                self.monitor_state = MonitorState(REGULAR_STATE, timestamp)
                self.sampling_period = self.img_q_period_ms
                self.flag_type = 0  # no flag
                logger.info(
                    f"Exiting short-term IQ degradation phase at {timestamp}."
                    f" Oversampling was deactivated to {self.sampling_period}ms"
                    )
                if self.monitor_state.transition_oversampling:
                    # reset transition oversampling flag
                    self.monitor_state.transition_oversampling = False
                    self.monitor_state.oversampling_start_time = 0
                    logger.info(f"Exiting also oversampling mid-transition state at {timestamp}.")

    def _handle_transition_state(self, raw_res, timestamp, debounced_res):
        if not raw_res and not self.monitor_state.transition_oversampling:
            # enter oversampling mid-transition state due to a single non-degraded result
            self.monitor_state.transition_oversampling = True
            self.monitor_state.oversampling_start_time = self.monitor_state.start_time  # save the original start time of transition state
            self.monitor_state = MonitorState(OVERSAMPLING_STATE, timestamp)
            self.sampling_period = max(self.min_sampling_period_ms, self.img_q_period_ms // self.oversample_factor)
            self.consecutive_count = 0  # reset for oversampling debouncing
            logger.info(f"Entering oversampling examination mid-transition state at {timestamp}."
                        f" Sampling period changed to {self.sampling_period}ms")
        elif timestamp - self.monitor_state.start_time >= self.transition_phase:
            if debounced_res:
                # camera's images are long term degraded, stop thresholds recalibrating process
                # TODO: later add VCC inspection mechanism here
                self.flag_type = 2  # long-term degradation
                logger.info(f"Raising long-term IQ degradation flag at {timestamp}. "
                            f" Halting recalibrating process.")
            else:
                self.monitor_state = MonitorState(REGULAR_STATE, timestamp)
                self.flag_type = 0  # no flag
                logger.info(f"Exiting long-term IQ degradation phase at {timestamp}. Resuming recalibrating process.")

    def _output_control_mec(self, raw_res, timestamp):
        debounced_res = self._debounce_mec(raw_res, timestamp)
        if raw_res and self.monitor_state.state == REGULAR_STATE:
            self._handle_regular_state(timestamp)
        elif self.monitor_state.state == OVERSAMPLING_STATE:
            self._handle_oversampling_state(raw_res, timestamp, debounced_res)
        elif self.monitor_state.state == TRANSITION_STATE:
            self._handle_transition_state(raw_res, timestamp, debounced_res)
        return debounced_res

    def _is_degraded(self, curr_batch, timestamp):
        # monitoring current batch based on previous calibration
        num_degraded_day = 0
        num_degraded_night = 0
        curr_day_scores = curr_batch[curr_batch[:, -1] == 1]
        curr_day_scores = curr_day_scores[:, :-1]  # keep only scores, remove day/night label
        curr_night_scores = curr_batch[curr_batch[:, -1] == 0]
        curr_night_scores = curr_night_scores[:, :-1]  # keep only scores, remove day/night label
        if curr_day_scores.shape[0] != 0:
            num_degraded_day = np.sum(curr_day_scores <= self.ths_daytime)
        if curr_night_scores.shape[0] != 0:
            num_degraded_night = np.sum(curr_night_scores <= self.ths_nighttime)
        res = (num_degraded_day + num_degraded_night) > (self.decision_p_th * self.batch_size)
        # flagging control mechanism
        res = self._output_control_mec(res, timestamp)
        # store embedding in the appropriate daypart bucket (only non-degraded frames)
        if self.emb_vec is not None and not res:
            last_day_part = curr_batch[-1, -1]
            emb_normed = self.emb_vec / np.linalg.norm(self.emb_vec)
            if last_day_part == 1:
                self.embedding_history_day.append((timestamp, emb_normed))
            else:
                self.embedding_history_night.append((timestamp, emb_normed))
        return res

    def iq_monitor(self, images: List[np.ndarray], timestamps: List[int]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        self.batch_size = len(images)
        curr_per = timestamps[-1] // self.sampling_period
        curr_stack_per = timestamps[-1] // self.img_q_period_ms  # always use original period for stack and calibration updates
        if self.enabled and curr_per > self.img_q_last_per:
        # if True:            # TODO: for debug only
            res = False
            self.img_q_last_per = curr_per
            net_scores, bright_score, sharp_score = self._compute_scores(images)
            new_rows = np.column_stack(
                [net_scores, bright_score, sharp_score, -1 * np.ones(self.batch_size)]  # (B,2)  # (B,)  # (B,)  # (B,)
            )
            # check if enough data has accumulated for determining day part and iq
            if self.current_len > self.accumulation_period:
                # determine daytime & check image quality for last batch
                new_rows = self._determine_day_part(new_rows)
                res = self._is_degraded(new_rows, timestamps[-1])
                # save last day part state
                self.prev_day_part = new_rows[-1, -1]
            # stack update only active on original 20-minute schedule, not during oversampling amd not when long-term degradation is detected
            if curr_stack_per > self.img_q_last_stack_per and self.flag_type != 2:
                self.timestamps_history.extend(timestamps)
                if self.current_len >= self.next_accumulation_period:
                    # calibrate day part statistically
                    self._calibrate_day_part()
                    # calibrate metrics thresholds
                    self._calibrate_ths()
                    self.next_accumulation_period += self.accumulation_period
                # update metrics history (cyclic buffer)
                end_idx = self.write_idx + self.batch_size
                if end_idx <= self._metrics_stack.shape[0]:
                    self._metrics_stack[self.write_idx : end_idx, :] = new_rows
                else:
                    split_idx = self._metrics_stack.shape[0] - self.write_idx  # split idx in stack
                    self._metrics_stack[self.write_idx :, :] = new_rows[:split_idx]
                    self._metrics_stack[: end_idx % self._metrics_stack.shape[0]] = new_rows[split_idx:]
                self.write_idx = end_idx % self._metrics_stack.shape[0]
                self.current_len = self.current_len + self.batch_size
                self.img_q_last_stack_per = curr_stack_per
            # prepare metrics scores as a dict
            metrics_scores_dict = {
                "arniqa": net_scores[-1, 0], 
                "ilniqe": net_scores[-1, 1],
                "bright_scores": bright_score[-1],
                "sharp_scores": sharp_score[-1],
                "day_parts": new_rows[-1, -1].astype(int),  # convert to list for JSON serialization
            }
            return res, metrics_scores_dict
        # TODO: add types of flags: short term / long term
        return False, None
