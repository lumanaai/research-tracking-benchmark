"""Identity Consistency (IDCons) metric.

For each GT identity, after per-frame IoU bipartite matching (threshold 0.5):

    IDCons_i = (# frames matched to the modal tracker ID) / (# frames with any match)

GT identities that are never matched are excluded. Sequence / COMBINED score is
the unweighted mean over scored GT identities (range [0, 1]; higher is better).
A score of 1.0 means every scored GT identity was always matched to the same
tracker ID.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._base_metric import _BaseMetric
from .. import _timing
from .. import utils


class IDCons(_BaseMetric):
    """Mean per-GT purity of matched tracker IDs."""

    @staticmethod
    def get_default_config():
        return {
            "THRESHOLD": 0.5,  # IoU (similarity) required for a match.
            "PRINT_CONFIG": True,
        }

    def __init__(self, config=None):
        super().__init__()
        self.integer_fields = ["IDCons_num_ids"]
        self.float_fields = ["IDCons"]
        # IDCons_sum is accumulated across sequences then converted to IDCons.
        self.summed_fields = ["IDCons_sum", "IDCons_num_ids"]
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = ["IDCons"]

        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config["THRESHOLD"])

    @_timing.time
    def eval_sequence(self, data):
        res = {
            "IDCons": 0.0,
            "IDCons_sum": 0.0,
            "IDCons_num_ids": 0,
        }
        if data["num_gt_ids"] == 0 or data["num_tracker_dets"] == 0:
            return res

        num_gt_ids = data["num_gt_ids"]
        # match_counts[gt_id, tracker_id] = frames where that pair was matched.
        match_counts = np.zeros((num_gt_ids, data["num_tracker_ids"]), dtype=np.float64)

        for gt_ids_t, tracker_ids_t, similarity in zip(
            data["gt_ids"], data["tracker_ids"], data["similarity_scores"]
        ):
            if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
                continue

            score_mat = similarity.copy()
            score_mat[similarity < self.threshold - np.finfo("float").eps] = 0.0
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            actually_matched = score_mat[match_rows, match_cols] > 0 + np.finfo("float").eps
            match_rows = match_rows[actually_matched]
            match_cols = match_cols[actually_matched]
            if len(match_rows) == 0:
                continue

            matched_gt = gt_ids_t[match_rows]
            matched_tracker = tracker_ids_t[match_cols]
            for gid, tid in zip(matched_gt, matched_tracker):
                match_counts[int(gid), int(tid)] += 1.0

        per_gt_matched = match_counts.sum(axis=1)
        scored = per_gt_matched > 0
        if not np.any(scored):
            return res

        modal = match_counts[scored].max(axis=1)
        per_gt = modal / per_gt_matched[scored]
        res["IDCons_sum"] = float(per_gt.sum())
        res["IDCons_num_ids"] = int(scored.sum())
        return self._compute_final_fields(res)

    def combine_sequences(self, all_res):
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        return self._compute_final_fields(res)

    def combine_classes_det_averaged(self, all_res):
        return self.combine_sequences(all_res)

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        if ignore_empty_classes:
            all_res = {
                k: v
                for k, v in all_res.items()
                if v.get("IDCons_num_ids", 0) > 0
            }
        if not all_res:
            return {"IDCons": 0.0, "IDCons_sum": 0.0, "IDCons_num_ids": 0}
        return self.combine_sequences(all_res)

    @staticmethod
    def _compute_final_fields(res):
        num_ids = float(res.get("IDCons_num_ids", 0))
        res["IDCons"] = float(res.get("IDCons_sum", 0.0)) / np.maximum(1.0, num_ids)
        return res
