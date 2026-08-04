from typing import Dict, Optional

from .gloves import GlovesAlert
from .base_alerts import AlertCandidate
from level1.pa_recognition.phone_detector import PhoneModel

class PhoneAlert(GlovesAlert):
    classifier: PhoneModel = None
    confidence_th = 0.2
    alert_message = "Phone violation detected"
    max_supported_person = 10
    no_activity_timeout = 10 * 1000  # 10 sec
    loiter_duration = 0 * 1000  # 0 sec
    classifier_timeout = 3 * 1000  # 3 sec
    latency = 5 # for majority vote of 3 out of 5 events

    def __init__(self, alert_dict: Dict, context):
        super(PhoneAlert, self).__init__(alert_dict, context)
        self.success_required_filter = False
        analytic_config = self.context.get_config()
        # override classifier config from l1_models.attributes.phone
        if "l1_models" in analytic_config and "attributes" in analytic_config["l1_models"]:
            phone_model_cfg = analytic_config["l1_models"]["attributes"].get("phone", {})
            if phone_model_cfg:
                self.classifier_config.update(phone_model_cfg)
                self.classifier = self._create_classifier()
                self.crop_margins = [self.classifier.required_margins] * 2
        # override alert-level params from alertConfig.phoneAlert
        if "alertConfig" in analytic_config:
            if "phoneAlert" in analytic_config["alertConfig"]:
                cfg = analytic_config["alertConfig"]["phoneAlert"]
                self.loiter_duration = cfg.get("loiter_duration", self.loiter_duration)
                self.classifier_timeout = cfg.get("absence_timeout", self.classifier_timeout)
                self.confidence_th = cfg.get("confidence_th", self.confidence_th)

    def _create_classifier(self):
        return PhoneModel(self.classifier_config)

    def _compile_classification_requirements(self):
        return self.classifier.compile_classification_requirements()

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        """Phone alerts skip entity blacklist while preserving normal candidate validation."""
        result = super(PhoneAlert, self).check_alert_candidate(candidate, force=force)
        if result is None:
            return True
        return result

