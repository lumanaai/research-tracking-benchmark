from dataclasses import dataclass
from typing import List, Dict, Set

from general.analyzer_general import logger
from general.core import (
    AdvanceAnalyticPriority,
    AdvanceAnalyzerType,
    ClassHandler,
    AdvanceAnalyticResults,
    advance_analyzer_priority,
    MotionData,
)
from general.entity_db import EntityDB
from level1 import advanced


@dataclass
class History:
    attemps: int = 0
    best_score: int = -1


class L1Manager:
    def __init__(self, context):
        self.context = context
        l1_analyzers = {}
        self._class_mapping: Dict[int, Set] = {}
        self.work_buffer: Dict[str, Dict[int, AdvanceAnalyticPriority]] = {}
        self.analyzer_history: Dict[str, Dict[int, int]] = {}
        self.max_retries = {}
        self.max_required_retries = {}
        self._default_max_required_retries = 3  # max retries if the request is not high priority

        self.l1_config = context.config["l1_models"]
        if self.l1_config is not None:
            factory = advanced.AdvanceFactory()
            # sort models by priority so results will be rewritten by the highest priority
            models = sorted(self.l1_config["models"], key=lambda x: advance_analyzer_priority[x])
            for l1_name in models:
                conf = self.l1_config["attributes"].get(l1_name, {})
                analyzer = factory.create(l1_name, conf, self, context.analyzer_id)
                self.max_retries[l1_name] = analyzer.max_retries
                self.max_required_retries[l1_name] = getattr(
                    analyzer, 'max_required_retries', self._default_max_required_retries
                )
                if analyzer is None:
                    logger.error(f"Non supported L1: {l1_name}")
                else:
                    analyzer.encoder_address = context.nvjpeg_encoder_address
                    l1_analyzers[l1_name] = analyzer
                    self.work_buffer[l1_name] = {}
                    self.analyzer_history[l1_name] = {}
                    for c in analyzer.supported_classes:
                        if c not in self._class_mapping:
                            self._class_mapping[c] = {l1_name}
                        else:
                            self._class_mapping[c].add(l1_name)

        self.unhandled_classes = set()
        self.unhandled_ents = set()
        self.l1_analyzers = l1_analyzers
        self.entity_db: EntityDB = context.entity_db
        self.entity_db.on_entities_update += self.on_entities_update
        self.entity_db.on_entities_removed += self.on_entities_removed
        self.entity_db.on_entities_purged += self.on_entities_purged
        context.on_night_mode_changed += self.on_night_mode_changed
        self._update_unhandled_classes()

    def on_entities_removed(self, ent_ids: List[int]):
        for ent in ent_ids:
            self.request_attributes(ent)

    def on_entities_update(self, ent_ids: List[int]):
        # called every time entity metadata is changed (new or update)
        pass

    def on_entities_purged(self, ent_ids: List[int]):
        for l1_name in self.analyzer_history:
            for ent in ent_ids:
                self.analyzer_history[l1_name].pop(ent, None)
                self.work_buffer[l1_name].pop(ent, None)
                self.unhandled_ents.discard(ent)
            self.l1_analyzers[l1_name].cleanup(ent_ids)

    def on_night_mode_changed(self, night_mode: bool):
        for analyzer in self.l1_analyzers.values():
            analyzer.is_night_mode = night_mode

    def request_attributes(
        self,
        ent_id,
        priority: AdvanceAnalyticPriority = AdvanceAnalyticPriority.REQUIRED,
        l1_analyzers: Set[str] = None,
    ):
        ent_data = self.entity_db.get_entity(ent_id)
        if ent_data.object_id in self.unhandled_classes:
            # if the class is not handled by any analyzer, we only need to request its image
            self.unhandled_ents.add(ent_id)
            return

        if l1_analyzers is None:
            l1_analyzers = self._class_mapping.get(ent_data.object_id, set())
        for l1 in l1_analyzers:
            if l1 is not None and (not ent_data.has_l1(l1) or priority > AdvanceAnalyticPriority.REQUIRED):
                if (
                    self.analyzer_history[l1].get(ent_id, 0) >= self.max_retries[l1]
                    and priority > AdvanceAnalyticPriority.REQUIRED
                ):
                    del self.analyzer_history[l1][ent_id]
                curr = self.work_buffer[l1].get(ent_id, AdvanceAnalyticPriority.REQUIRED)
                self.work_buffer[l1][ent_id] = max(curr, priority)

    def track(self, image_batch, batch_data, motion_data: MotionData):
        for l1_name in self.l1_analyzers:
            self.l1_analyzers[l1_name].track(image_batch, batch_data, motion_data)

    def analyze(self, image_batch, batch_data, alerted_ents: Dict[int, Set[str]], motion_data: MotionData):
        l1_results = []
        stats = {}
        alerted_not_none = set()
        for ent_id in alerted_ents:
            if alerted_ents[ent_id] is not None:
                self.request_attributes(ent_id, AdvanceAnalyticPriority.HIGH, alerted_ents[ent_id])
                alerted_not_none.add(ent_id)

        for ent_id, ent_data in self.entity_db.active_objects.items():
            if ent_id not in alerted_not_none and ent_data.dwell > 60000 and not ent_data.has_any_l1:
                self.request_attributes(ent_id)
                alerted_not_none.add(ent_id)

        for l1_name in self.l1_analyzers:
            l1_analyzer = self.l1_analyzers[l1_name]
            batch_data_for_l1 = l1_analyzer.prepare_batch_data(batch_data)
            l1_analyzer.track(image_batch, batch_data_for_l1, motion_data)
            work_buffer = self.work_buffer[l1_name]
            if len(work_buffer) == 0 or not l1_analyzer.enabled:
                continue
            results, images, stat, success = l1_analyzer.run(image_batch, batch_data_for_l1, work_buffer)
            l1_results.append(
                {"analyzer": l1_analyzer.analyzer_type, "results": results, "images": images, "success": success}
            )
            if stat:
                stats[l1_analyzer.name] = stat
            else:
                stats[l1_analyzer.name] = {}
            for ent_id in success:
                if success[ent_id] != AdvanceAnalyticResults.NO_VALID_CANDIDATE:
                    self.analyzer_history[l1_name][ent_id] = self.analyzer_history[l1_name].get(ent_id, 0) + 1
                if success[ent_id] >= l1_analyzer.min_require_results:
                    del self.work_buffer[l1_name][ent_id]
                elif (
                    success[ent_id] == AdvanceAnalyticResults.NO_VALID_CANDIDATE and l1_analyzer.is_export_invalid_crops
                ):
                    if ent_id not in images:
                        best_image = self.entity_db.get_largest_image(ent_id)
                        if best_image is not None:
                            images[ent_id] = {"image": best_image}

                else:
                    if self.analyzer_history[l1_name].get(ent_id, 0) == self.max_retries[l1_name]:
                        # demote priority to low
                        self.work_buffer[l1_name][ent_id] = min(
                            self.work_buffer[l1_name][ent_id], AdvanceAnalyticPriority.LOW
                        )
                    elif self.analyzer_history[l1_name].get(ent_id, 0) > self.max_retries[l1_name] or (
                        self.work_buffer[l1_name][ent_id] <= AdvanceAnalyticPriority.REQUIRED
                        and self.analyzer_history[l1_name].get(ent_id, 0) >= self.max_required_retries[l1_name]
                    ):
                        del self.work_buffer[l1_name][ent_id]
        general_images = {}
        general_success = {}
        for ent_id in self.unhandled_ents:
            best_image = self.entity_db.get_largest_image(ent_id)
            if best_image is not None:
                general_images[ent_id] = {"image": best_image}
                general_success[ent_id] = AdvanceAnalyticResults.SUCCESS
        if general_images:
            l1_results.append(
                {
                    "analyzer": AdvanceAnalyzerType.BASE,
                    "results": {},
                    "images": general_images,
                    "success": general_success,
                }
            )
            stats[AdvanceAnalyzerType.BASE] = {}
        self.unhandled_ents.clear()
        return l1_results, stats

    def update_l1_config(self, l1_analyzer: AdvanceAnalyzerType, enable: bool = None, filter_in: dict = None):
        if l1_analyzer in self.l1_analyzers:
            self.l1_analyzers[l1_analyzer].update_filters(enable, filter_in)
            self._update_unhandled_classes()

    def default_filters_analyzer(self):
        return {
            self.class_handler.person_value: AdvanceAnalyzerType.HUMAN_PARSING.value,
            self.class_handler.vehicle_value: AdvanceAnalyzerType.ALPR.value,
        }

    def default_desc_analyzer(self):
        return {
            self.class_handler.person_value: AdvanceAnalyzerType.HUMAN_PARSING.value,
            self.class_handler.vehicle_value: AdvanceAnalyzerType.VEHICLE.value,
            self.class_handler.plate_subclass_value: AdvanceAnalyzerType.LPREC.value,
        }

    def get_default_filter_analyzer(self, object_id: int, filters: Dict):
        # recognition is not using the filter analyzer - this is for attributes only
        analyzer = None
        if object_id == self.class_handler.person_value:
            return AdvanceAnalyzerType.HUMAN_PARSING.value
        elif object_id == self.class_handler.vehicle_value:
            return AdvanceAnalyzerType.VEHICLE.value
        return analyzer

    def get_default_recognition_analyzer(self, object_id: int):
        analyzer = None
        if object_id == self.class_handler.person_value:
            return AdvanceAnalyzerType.FACE.value
        elif object_id == self.class_handler.vehicle_value:
            rec_config = self.context.config.get("alertConfig", {}).get("recognition", {})
            use_external_apis = self.context.config.get("l1_models", {}).get("use_external_apis", False)
            use_external_lpr = rec_config.get("useALPR", True) or use_external_apis
            if use_external_lpr:
                return AdvanceAnalyzerType.ALPR.value
            else:
                return AdvanceAnalyzerType.LPREC.value
        return analyzer

    def update_model(self, l1_name, config):
        self.l1_config = config["l1_models"]
        conf = self.l1_config["attributes"].get(l1_name, {})
        analyzer = advanced.AdvanceFactory().create(l1_name, conf, self, self.context.analyzer_id)
        analyzer.encoder_address = self.context.nvjpeg_encoder_address
        self.l1_analyzers[l1_name] = analyzer

    def get_analyzer(self, analyzer_type: AdvanceAnalyzerType):
        return self.l1_analyzers.get(analyzer_type, None)

    @property
    def default_analyzers(self) -> Dict[int, Set]:
        return self._class_mapping

    @property
    def class_handler(self) -> ClassHandler:
        return self.context.class_handler

    @property
    def analytic_config(self):
        return self.context.config

    def _update_unhandled_classes(self):
        handled = set()
        for k, v in self._class_mapping.items():
            for analyzer in v:
                if self.l1_analyzers[analyzer].enabled:
                    handled.add(k)
                    break
        self.unhandled_classes = set(self.context.class_handler.object_ids) - handled
