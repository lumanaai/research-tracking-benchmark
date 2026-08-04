import os
import numpy as np
import json
from dataclasses import dataclass
from general.analyzer_general import logger, log_exception
from general.proj import assets_path
from general.core import BatchDataResolver
from typing import Dict, Set, Tuple

HISTOGRAM_SIZE = 1000


@dataclass
class DetImageData:
    data: np.array

    @property
    def subclasses(self) -> np.ndarray:
        return self.data[:, BatchDataResolver.SUBCLASS]

    @property
    def track_ids(self) -> np.ndarray:
        return self.data[:, BatchDataResolver.ID]

    @property
    def pos(self) -> np.ndarray:
        return self.data[:, BatchDataResolver.POS]

    @property
    def hw(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.pos[:, 3] - self.pos[:, 1], self.pos[:, 2] - self.pos[:, 0]

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, BatchDataResolver.CONFIDENCE]

    def __len__(self):
        return self.data.shape[0]


class RobustStatsCollector:
    """Saver for the size of the detected objects"""

    subclass_sizes: Dict
    ids_list: Set

    def __init__(self, ent_db_context):

        context = ent_db_context.context
        self.history_file_path = os.path.join(assets_path(), context.edge_id, context.camera_id, "history_sizes.json")
        config = context.config.get("entityManagement", {}).get("stats_collection", {})
        tracking_config = context.config.get("l0_track", {})
        self.enable = config.get("enable", False)
        if not self.enable:
            self.add_id_data = lambda x: None
            self.collect = lambda: {}
            return

        self.min_samples_for_filtering = config.get("minSamplesForFiltering", 500)
        self.robust_max_percentile = config.get("robustMaxPercentile", 0.9)
        self.min_ids_for_filtering = config.get("minIdsForFiltering", 50)

        self.n_classes = context.class_handler.n_classes
        to_track = context.class_handler.filter_out_objects(tracking_config.get("untrack_objects", []))
        self.supposed_classes = set(to_track)
        self.history_file_path = self.history_file_path
        self.reset_history()
        ent_db_context.on_entities_update += self.on_entity_update
        ent_db_context.on_entities_purged += self.on_entities_purged
        self.entity_db = ent_db_context

        hist_files = [self.history_file_path, self.history_file_path + ".backup"]
        for file in hist_files:
            if os.path.exists(file):
                try:
                    with open(file, "r") as f:
                        hist_dict = json.load(f)

                    for k in hist_dict.keys():
                        v = hist_dict.get(k, None)
                        if v is not None:
                            self.subclass_sizes[int(k)] = {
                                "h": FloatNormalizedHistogram.from_list(v["h"]),
                                "w": FloatNormalizedHistogram.from_list(v["w"]),
                                "samples": v.get("samples", 0),
                            }
                    break
                except Exception as e:
                    log_exception(logger, f"Error occurred loading history file {file}", e)
                    try:
                        os.remove(self.history_file_path)
                    except Exception as e:
                        log_exception(logger, "Error occurred removing history file", e)

    def reset_history(self):
        self.subclass_sizes = {}
        for i in self.supposed_classes:
            self.subclass_sizes[i] = {
                "h": FloatNormalizedHistogram(HISTOGRAM_SIZE),
                "w": FloatNormalizedHistogram(HISTOGRAM_SIZE),
                "samples": 0,
            }
        self.ids_list = set()  #   set of ids that were already processed- when size lower than min_ids_for_filtering

    def on_entity_update(self, ent_ids):
        for ent_id in ent_ids:
            if ent_id not in self.ids_list:
                ent_data = self.entity_db.get_entity(ent_id)
                self.ids_list.add(ent_id)
                if ent_data is not None and ent_data.class_id in self.supposed_classes:
                    self.subclass_sizes[ent_data.class_id]["samples"] += 1

    def on_entities_purged(self, ent_ids):
        for ent_id in ent_ids:
            self.ids_list.discard(ent_id)

    def _save_history(self):
        # save history to file, not saving tracking information
        hist_dict = {}
        for k, v in self.subclass_sizes.items():
            hist_dict[k] = {"h": v["h"].hist.tolist(), "w": v["w"].hist.tolist(), "samples": v["samples"]}
        try:
            with open(self.history_file_path + ".backup", "wt") as fp:
                json.dump(hist_dict, fp)
            with open(self.history_file_path, "wt") as fp:
                json.dump(hist_dict, fp)
        except Exception as e:
            log_exception(logger, "Error occurred saving history file", e)

    def _update_histograms(self, det_image_data: DetImageData):
        """Get Detections from the image data"""
        hs, ws = det_image_data.hw  # update the class sizes
        subclasses = det_image_data.subclasses
        unique_scls, inverse_indices = np.unique(subclasses.astype(int), return_inverse=True)
        indices_by_scls = {
            scls: np.where(inverse_indices == i)[0]
            for i, scls in enumerate(unique_scls)
            if scls in self.supposed_classes
        }
        for scls, idxs in indices_by_scls.items():
            self.subclass_sizes[scls]["h"].add_values(hs[idxs])
            self.subclass_sizes[scls]["w"].add_values(ws[idxs])

    def _get_max_size(self) -> Dict:
        """Get the robust max size from histograms and high percentile"""
        max_size_dict = {}
        for k, v in self.subclass_sizes.items():
            samples_count = v["h"].sample_count()
            if samples_count > self.min_samples_for_filtering and v["samples"] >= self.min_ids_for_filtering:
                h_robust_max = v["h"].get_percentile(self.robust_max_percentile)
                w_robust_max = v["w"].get_percentile(self.robust_max_percentile)
                max_size_dict[k] = {"h": h_robust_max, "w": w_robust_max}
        return max_size_dict

    def add_id_data(self, id_data: np.array):
        image_data = DetImageData(id_data)
        self._update_histograms(image_data)

    def collect(self) -> Dict:
        self._save_history()
        sz_dict = self._get_max_size()
        ret_dict = {}
        if sz_dict:
            ret_dict["cls_max_size"] = sz_dict
        return ret_dict


class FloatNormalizedHistogram:
    """Histogram for float values between 0 and 1, with a bin for each range of values."""

    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.hist = np.zeros(n_bins, dtype=int)

    def add_values(self, values: np.array):
        values = np.clip(values, 0, 0.999999)
        bin_indices = np.floor(values * self.n_bins).astype(int)
        np.add.at(self.hist, bin_indices, 1)

    def get_percentile(self, percentile):
        cumulative_sum = np.cumsum(self.hist)
        total = cumulative_sum[-1]
        if total == 0:
            return 0
        target = total * percentile
        bin_index = np.argmax(cumulative_sum >= target)
        return bin_index / self.n_bins

    @staticmethod
    def from_list(hist_list):
        hist = FloatNormalizedHistogram(len(hist_list))
        hist.hist = np.array(hist_list, dtype=int)
        return hist

    def sample_count(self):
        return np.sum(self.hist)

