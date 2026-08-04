from collections import defaultdict
from datetime import datetime
import os

import numpy as np
import json
from typing import List, Optional
from copy import deepcopy
from dataclasses import dataclass

from general.analyzer_general import ROI_SHAPE, logger, log_exception
from general.proj import assets_path
from general.core import BatchDataResolver, AnalyticImage
from general.img_utils import scale_image_by_width


@dataclass
class ImageData:
    data: np.array
    timestamp: float
    is_night_mode: bool
    is_blacklist: bool
    image: Optional[np.array] = None

    @property
    def classes(self) -> np.ndarray:
        return self.data[:, BatchDataResolver.SUBCLASS]

    @property
    def track_ids(self) -> np.ndarray:
        return self.data[:, BatchDataResolver.ID]

    @property
    def center_locations(self) -> np.ndarray:
        return self.data[:, BatchDataResolver.CENTER_LOCATION]

    @property
    def pos(self) -> np.ndarray:
        return self.data[:, BatchDataResolver.POS]

    @property
    def detections(self) -> np.array:
        return np.concatenate([self.classes[:, None], self.pos], 1)

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, BatchDataResolver.CONFIDENCE]

    @property
    def is_background(self) -> bool:
        return self.data.shape[0] == 0


class SaverSelector:
    """
    Selects the best images to be saved for proper fitting training.
    For each image calculate a score based on the following criteria:
    1. Balance between the number of images per class, time, locations, etc.
    2. Images with low confidence.

    The saver maintains a priority queue of the best images and their scores.
    When image is asked for sending to the server, the image with the highest score is extracted.
    """

    def __init__(self, context):
        self.history_file_path = os.path.join(assets_path(), context.edge_id, context.camera_id, "history.json")
        self.resolution = context.config["trainThumbPolicy"].get("resize", -1)
        self.background_selection_freq = context.config["trainThumbPolicy"].get("backgroundFreq", 30)
        self.min_score_th = context.config["trainThumbPolicy"].get("minScoreThr", 0)
        self.max_repeats_track_id = context.config["trainThumbPolicy"].get("maxRepeatdTrkId", 2)
        self.queue_max_length = context.config["trainThumbPolicy"].get("TrainThumbnailQueueLen", 15)
        self.ir_ratio = context.config["trainThumbPolicy"].get("irRatio", 0.2)
        self.track_id_max_size = context.config["trainThumbPolicy"].get("trackIdMaxSize", 100)
        self.n_classes = context.class_handler.n_classes
        self.counter2periods = context.config["trainThumbPolicy"].get(
            "TrainThumbnailDurationCounter", {"1000": 250, "3000": 400, "5000": 800, "after": 1600}
        )
        self.entity_db = context.entity_db
        self.queue = MaxScoredQueue(self.queue_max_length)
        self.is_blocked = False

        history_loaded = False
        hist_files = [self.history_file_path, self.history_file_path + ".backup"]
        for file in hist_files:
            if os.path.exists(file):
                try:
                    with open(file, "r") as f:
                        hist_dict = json.load(f)

                    self.class_hist = IntHistogram.from_list(hist_dict["class_hist"])
                    self.month_hist = IntHistogram.from_list(hist_dict["month_hist"])
                    self.hour_hist = IntHistogram.from_list(hist_dict["hour_hist"])
                    self.location_hist = IntHistogram.from_list(hist_dict["location_hist"])
                    self.ir_hist = BinaryHistogram.from_list(hist_dict["ir_hist"], required_ratio=self.ir_ratio)
                    self.last_background_day = datetime.fromtimestamp(hist_dict["last_background_day"])
                    self.last_background_night = datetime.fromtimestamp(hist_dict["last_background_night"])
                    # not saved in the DB
                    self.track_id_hist = HistoryList(max_size=self.track_id_max_size, max_repeats=self.max_repeats_track_id)
                    self.blacklist_ids = {-1}  # -1 is null id
                    history_loaded = True
                    self.is_blocked = hist_dict.get("is_blocked", False)
                    break
                except Exception as e:
                    log_exception(logger, f"Error occurred loading history file {file}", e)
                    try:
                        os.remove(self.history_file_path)
                    except Exception as e:
                        log_exception(logger, "Error occurred removing history file", e)
        if not history_loaded:
            self.reset_history()

    def reset_training_selection(self, is_blocked: bool = False):
        """Reset the training selection history and queue"""
        if is_blocked != self.is_blocked:
            logger.info(f"Resetting training selection, is_blocked: {is_blocked}")
            self.reset_history()
        self.queue = MaxScoredQueue(self.queue_max_length)
        hist_files = [self.history_file_path, self.history_file_path + ".backup"]
        for file in hist_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception as e:
                    log_exception(logger, f"Error occurred removing history file {file}", e)
                    try:
                        os.rename(file, file + ".error")
                    except Exception as e:
                        log_exception(logger, f"Error occurred renaming history file {file}", e)
        if is_blocked:
            self.is_blocked = True
            # generate history files with blocked indication
            self.save_to_files()

    def reset_history(self):
        self.class_hist = IntHistogram(self.n_classes)
        self.month_hist = IntHistogram(12)
        self.hour_hist = IntHistogram(24)
        n_location_bins = ROI_SHAPE[0] * ROI_SHAPE[1]
        self.location_hist = IntHistogram(n_location_bins)
        self.ir_hist = BinaryHistogram(required_ratio=self.ir_ratio)
        self.last_background_day = datetime.fromtimestamp(0)
        self.last_background_night = datetime.fromtimestamp(0)
        # not saved in the DB
        self.track_id_hist = HistoryList(max_size=self.track_id_max_size, max_repeats=self.max_repeats_track_id)
        self.blacklist_ids = {-1}  # -1 is null id
        self.is_blocked = False


    def get_save_period(self) -> float:
        """get the duration of the saver in seconds according to the history counter"""
        counter = self.ir_hist.hist.sum()
        for k, v in self.counter2periods.items():
            if k == "after":
                return v
            if counter < int(k):
                return v
        return self.counter2periods["after"]

    def add_images(
        self, image_batch: List[AnalyticImage], batch_data: BatchDataResolver, is_night_mode: bool
    ):
        if self.is_blocked:
            return
        """Calculate a score per image and if above the treshold, add to the queue"""
        frame_ids = batch_data.data[:, BatchDataResolver.FRAME_ID]

        # run it once per batch
        i = 0
        image = image_batch[i]
        idxs_of_frame = np.where(frame_ids == i)[0]
        data = batch_data.data[idxs_of_frame]
        timestamp = image.timestamp
        image_data = ImageData(data, timestamp, is_night_mode, self.is_blacklist(data[:, BatchDataResolver.ID]))
        score = self.calc_score(image_data)

        if score > self.min_score_th and self.queue.can_push(score):
            image = deepcopy(image.frame)
            if self.resolution != -1:
                image = scale_image_by_width(image, self.resolution)
            image_data.image = image
            self.queue.push(image_data, score)

    def calc_score(self, image_data: ImageData) -> float:
        """Calculate the score of an image"""
        n_detctions = image_data.data.shape[0]
        # score for background images, must take
        timestamp = datetime.fromtimestamp(image_data.timestamp / 1000)
        if n_detctions == 0:
            if image_data.is_night_mode:
                if (timestamp - self.last_background_night).days > self.background_selection_freq:
                    return 1000
            else:
                if (timestamp - self.last_background_day).days > self.background_selection_freq:
                    return 1000
            return 0

        scores = []
        # scores per detection in the image, also increase many objects in the same image
        new_trk_ids = False
        for i in range(n_detctions):
            score = 0
            score += self.class_hist.get_score(image_data.classes[i])
            score += self.location_hist.get_score(image_data.center_locations[i])
            track_score = self.track_id_hist.get_score(image_data.track_ids[i])
            score += track_score
            new_trk_ids |= track_score != 0
            if self.is_new_blacklist(image_data.track_ids[i]):
                score += 1
            scores.append(score)
        if not new_trk_ids:
            return 0
        score  = np.max(scores)
        # scores per image
        score += self.month_hist.get_score(timestamp.month - 1)  # month is 1-12, hist is 0-11
        score += self.hour_hist.get_score(timestamp.hour)
        score += self.ir_hist.get_score(image_data.is_night_mode)

        # TODO: Add score for low confidence?
        return score

    def is_new_blacklist(self, trk_id: int) -> bool:
        """Check if a track id is new in the blacklist"""
        orig_trk_id = self.entity_db.blacklist_id2orig_id.get(trk_id, -1)
        return orig_trk_id not in self.blacklist_ids

    def is_blacklist(self, trk_ids: np.array) -> bool:
        """Check if a any of the track ids is in the blacklist"""
        return any(trk_id in self.entity_db.blacklist_id2orig_id for trk_id in trk_ids)

    def update_history(self, image_data: ImageData):
        """Update the history of the saver with a new image"""
        for i in range(image_data.data.shape[0]):
            self.class_hist.add_value(image_data.classes[i])
            self.location_hist.add_value(image_data.center_locations[i])
            self.track_id_hist.add_value(image_data.track_ids[i])
            orig_trk_id = self.entity_db.blacklist_id2orig_id.get(image_data.track_ids[i], -1)
            self.blacklist_ids.add(orig_trk_id)
        timestamp = datetime.fromtimestamp(image_data.timestamp / 1000)
        self.month_hist.add_value(timestamp.month - 1)  # month is 1-12, hist is 0-11
        self.hour_hist.add_value(timestamp.hour)
        self.ir_hist.add_value(image_data.is_night_mode)

        is_background = image_data.data.shape[0] == 0
        if is_background:
            if image_data.is_night_mode:
                if (timestamp - self.last_background_night).days > self.background_selection_freq:
                    self.last_background_night = timestamp
            else:
                if (timestamp - self.last_background_day).days > self.background_selection_freq:
                    self.last_background_day = timestamp

        self.save_to_files()

    def save_to_files(self):
        # save history to file, not saving tracking information
        hist_dict = {
            "class_hist": self.class_hist.hist.tolist(),
            "month_hist": self.month_hist.hist.tolist(),
            "hour_hist": self.hour_hist.hist.tolist(),
            "location_hist": self.location_hist.hist.tolist(),
            "ir_hist": self.ir_hist.hist.tolist(),
            "last_background_day": self.last_background_day.timestamp(),
            "last_background_night": self.last_background_night.timestamp(),
            "is_blocked": self.is_blocked,
        }
        try:
            with open(self.history_file_path + ".backup", "wt") as f:
                json.dump(hist_dict, f)
            with open(self.history_file_path, "wt") as f:
                json.dump(hist_dict, f)
        except Exception as e:
            log_exception(logger, "Error occurred saving history file", e)

    def update_scores(self):
        """Update the scores of the images in the queue"""
        new_queue = MaxScoredQueue(self.queue.max_length)
        for _, item in self.queue.queue:
            new_score = self.calc_score(item)
            if new_score > self.min_score_th:
                new_queue.push(item, new_score)

        self.queue = new_queue

    def pop(self) -> Optional[ImageData]:
        """Get the image with the highest score"""
        if self.queue.is_empty():
            return None

        score, image_data = self.queue.pop()
        self.update_history(image_data)
        self.update_scores()  # Update the scores after the history update
        return image_data

    def __len__(self):
        return len(self.queue)


class MaxScoredQueue:
    """A priority queue that keeps the items with the highest score"""

    def __init__(self, max_length: int):
        self.queue = []
        self.max_length = max_length

    def push(self, item, score):
        # Insert the new item in a sorted position
        inserted = False
        for i in range(len(self.queue)):
            if score > self.queue[i][0]:
                self.queue.insert(i, (score, item))
                inserted = True
                break
        if not inserted:
            self.queue.append((score, item))

        # Ensure the length of the queue does not exceed max_length
        if len(self.queue) > self.max_length:
            self.queue.pop()  # Remove the item with the lowest score

    def can_push(self, score):
        return len(self.queue) < self.max_length or score > self.queue[-1][0]

    def pop(self):
        if not self.queue:
            raise IndexError("pop from an empty priority queue")
        return self.queue.pop(0)  # Return the item with the highest score

    def is_empty(self):
        return len(self.queue) == 0

    def max_score(self):
        if not self.queue:
            raise IndexError("pop from an empty priority queue")
        return self.queue[0][0]

    def min_score(self):
        if not self.queue:
            raise IndexError("pop from an empty priority queue")
        return self.queue[-1][0]

    def __len__(self):

        return len(self.queue)


class IntHistogram:
    """Histogram for integer values, from 0 to n_bins -1, with a bin for each value"""

    def __init__(self, n_bins):
        self.hist = np.zeros(n_bins, dtype=int)

    def add_value(self, value):
        value = int(value)
        if 0 <= value < len(self.hist):
            self.hist[value] += 1

    def get_hist_value(self, value):
        value = int(value)
        return self.hist[value] if 0 <= value < len(self.hist) else 0

    def get_score(self, value):
        """Calculate the score of a value based on the distance from the max,
        normalized to adress histograms with different hights and number of beans"""
        hist_max = np.max(self.hist)
        if hist_max == 0:  # If the histogram is empty, encourage all values
            return 1
        return max([(hist_max - self.get_hist_value(value)) / hist_max, 0])

    @staticmethod
    def from_list(hist_list):
        hist = IntHistogram(len(hist_list))
        hist.hist = np.array(hist_list)
        return hist


class HistoryList:
    """List containing the last max_size values."""

    def __init__(self, max_size: int, score: float = 1, max_repeats: int = 2):
        self.list = []
        self.count = defaultdict(int)
        self.max_size = max_size
        self.max_repeats = max_repeats
        self.score = score

    def add_value(self, value):
        if value in self.count:
            self.count[value] += 1
        else:
            self.list.append(value)
            self.count[value] = 1
            if len(self.list) > self.max_size:
                removed_id = self.list.pop(0)
                del self.count[removed_id]

    def get_score(self, value):
        """Score is 0 if the appears more than max_repeat times, else return the score."""
        if value not in self.list or self.count[value] < self.max_repeats:
            return self.score

        return 0


class BinaryHistogram:
    """Histogram for binary values,
    The score is calculated as the distnace of the True ratio from the required ratio"""

    def __init__(self, required_ratio: float = 0.5):
        self.hist = np.zeros(2, dtype=int)
        self.ratio = required_ratio

    def add_value(self, value):
        self.hist[int(value)] += 1

    def get_hist_value(self, value):
        return self.hist[value]

    def get_score(self, value):
        if (self.hist[0] + self.hist[1]) == 0:
            hist_ratio = 0.5
        else:
            hist_ratio = self.hist[1] / (self.hist[0] + self.hist[1])
        if value and hist_ratio < self.ratio:
            return self.ratio - hist_ratio
        elif not value and hist_ratio > self.ratio:
            return hist_ratio - self.ratio
        else:
            return 0

    def from_list(hist_list, required_ratio):
        hist = BinaryHistogram(required_ratio)
        hist.hist = np.array(hist_list)
        return hist
