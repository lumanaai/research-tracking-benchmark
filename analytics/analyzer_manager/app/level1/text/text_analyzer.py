from typing import Optional, List

import cv2
import numpy as np

from general.core import BaseConfig, AnalyticImage
from .text_detector import CraftTextDetector
from .text_recognizer import CrnnTextRecognition
from .utils import four_point_transform


class TextAnalyzerConfig(BaseConfig):
    im_size = [384, 640]
    batch_size = 8
    image_in_batch = 4
    slope_ths = 0.03
    add_margin = 0.1
    roi = None
    enabled = False
    min_sentence_conf = 0.5
    min_word_conf = 0.4
    min_word_sz = 5


class TextAnalyzer:
    def __init__(self, context, msg_dict: dict = None):
        if msg_dict is None:
            msg_dict = {}
        self.args = TextAnalyzerConfig(msg_dict)
        if self.args.roi is None:
            self.prepare_image = lambda x: x
            if context.resolution:
                img_w, img_h = context.resolution
                input_sz = [img_h, img_w]
            else:
                input_sz = None
        else:
            img_w, img_h = context.resolution
            x1, y1, x2, y2 = (np.array(self.args.roi) * np.array([img_w, img_h, img_w, img_h])).astype(int).tolist()
            self.prepare_image = lambda x: x[y1:y2, x1:x2]
            input_sz = [y2 - y1, x2 - x1]
        self.detector = CraftTextDetector(
            msg_dict.get("detector", {"input_sz": input_sz, "batch_size": self.args.batch_size})
        )
        self.recognizer = CrnnTextRecognition(msg_dict.get("recognizer", {}))

    def run(self, image_batch: List[AnalyticImage]):
        results = []
        images = [
            self.prepare_image(image.frame)
            for idx, image in enumerate(image_batch)
            if idx % self.args.image_in_batch == 0
        ]
        detections = self.detector.forward_on_crop_list(images)
        ordered = self.reorder_detections(detections)
        recognition_crops = []
        for i, det in enumerate(ordered):
            if len(det) == 0:
                continue
            crops = self.get_crops_from_detections(det, images[i])
            recognition_crops += crops

        if len(recognition_crops) > 0:
            text, confidence = self.recognizer.forward_on_crop_list(recognition_crops)
            t_idx = 0
            for i, det in enumerate(ordered):
                next_t = t_idx + len([d for d in det if d is not None])
                confidences = confidence[t_idx:next_t]
                if len(confidences) > 0 and np.max(np.array(confidences)) > self.args.min_sentence_conf:
                    results.append({"detections": det, "text": text[t_idx:next_t], "confidence": confidences})
                t_idx = next_t
        return results

    @staticmethod
    def reorder_detections(detections):
        ordered = []
        for frame_det in detections:
            if len(frame_det) < 2:
                ordered.append(frame_det)
                continue

            arr = np.array(frame_det)

            lines = []
            lines_y = []
            next_line = 0

            y_coords = arr[:, 1::2]
            words_h = np.max(y_coords, axis=1) - np.min(y_coords, axis=1)

            ltr = np.argsort(arr[:, 0])
            for idx in ltr:
                y = np.mean(y_coords[idx])
                if next_line == 0:
                    lines.append([idx])
                    lines_y.append(y)
                    next_line += 1
                else:
                    line_dists = np.abs(np.array(lines_y) - y)
                    closest = np.argmin(line_dists)
                    if line_dists[closest] < words_h[idx]:
                        lines_y[closest] = (lines_y[closest] + y) / 2
                        lines[closest].append(idx)
                        next_line += 1
                    else:
                        lines.append([idx])
                        lines_y.append(y)
                        next_line += 1
            new_img_ord = []
            sort_lines = np.argsort(lines_y)
            for line in sort_lines:
                new_img_ord += [frame_det[int(idx)] for idx in lines[int(line)]]
            ordered.append(new_img_ord)
        return ordered

    def _calc_warp(self, poly):
        height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
        width = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])

        margin = int(1.44 * self.args.add_margin * min(width, height))

        theta13 = abs(np.arctan((poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4]))))
        theta24 = abs(np.arctan((poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6]))))
        # do I need to clip minimum, maximum value here?
        x1 = poly[0] - np.cos(theta13) * margin
        y1 = poly[1] - np.sin(theta13) * margin
        x2 = poly[2] + np.cos(theta24) * margin
        y2 = poly[3] - np.sin(theta24) * margin
        x3 = poly[4] + np.cos(theta13) * margin
        y3 = poly[5] + np.sin(theta13) * margin
        x4 = poly[6] - np.cos(theta24) * margin
        y4 = poly[7] + np.sin(theta24) * margin

        return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=float)

    def get_crops_from_detections(self, detections, image):
        horizontal_list, free_list, combined_list, merged_list = [], [], [], []
        crops = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        de_normalized = np.array(detections)
        de_normalized[:, ::2] *= image.shape[1]
        de_normalized[:, 1::2] *= image.shape[0]
        for idx, poly in enumerate(de_normalized):
            # poly: top-left, top-right, low-right, low-left

            # check if text is horizontally aligned
            slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
            slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))
            if max(abs(slope_up), abs(slope_down)) < self.args.slope_ths:
                # aligned text - only need to crop
                x_max = int(max([poly[0], poly[2], poly[4], poly[6]]))
                x_min = int(min([poly[0], poly[2], poly[4], poly[6]]))
                y_max = int(max([poly[1], poly[3], poly[5], poly[7]]))
                y_min = int(min([poly[1], poly[3], poly[5], poly[7]]))
                crop = gray[y_min:y_max, x_min:x_max]
            else:
                # unaligned - need to warp the image
                rect = self._calc_warp(poly)
                crop = four_point_transform(gray, rect)
            if np.any(np.array(crop.shape) < self.args.min_word_sz):
                detections[idx] = None
            else:
                crops.append(crop)
        return crops

    @property
    def batch_size(self):
        return self.args.batch_size
