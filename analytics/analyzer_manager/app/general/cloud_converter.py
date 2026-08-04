from collections import defaultdict
from copy import deepcopy, copy
from typing import Dict, List

from general.core import AttrConfidence

cloud_TrackerTypes_converter = {
    "unknown": 0,
    "person": 1,
    "vehicle": 2,
    "pet": 3,
    "weapon": 4,
    "hazard": 5,
    "shoppingcart": 6,
    "container": 7,
}

# convert vehicle type
cloud_VehicleTypes_converter = {
    "unknown": 0,
    "sedan": 1,
    "suv": 2,
    "van": 3,
    "pickup truck": 4,
    "motorcycle": 5,
    "bus": 6,
    "big truck": 7,
    "bicycle": 8,
    "forklift": 9,
    "boat": 10,
    "minivan": 3,
    "crossover": 2,
    "tractor": 7,
    "golf cart": 1,
}

cloud_GlobalTypes_converter = {
    "unknown": 0,
    "success": 1,
    "failure": 2,
    "blacklist": 3,
}

cloud_ColorTypes_converter = {
    "unknown": 0,
    "black": 1,
    "blue": 2,
    "brown": 3,
    "green": 4,
    "grey": 5,
    "orange": 6,
    "pink": 7,
    "purple": 8,
    "red": 9,
    "yellow": 10,
    "white": 11,
}

cloud_LowerbodyTypes_converter = {
    "unknown": 0,
    "shorts": 1,
    "trousers": 2,
    "skirt_and_dress": 3,
    "not_solid": 4,
    "solid": 5,
}

cloud_UpperbodyTypes_converter = {
    "unknown": 0,
    "short_sleeve": 1,
    "long_sleeve": 2,
    "logo": 3,
    "not_solid": 4,
    "no_logo": 5,
    "solid": 6,
}

cloud_CarryingTypes_converter = {
    "unknown": 0,
    "bag": 1,
    "no_bag": 2,
    "object": 3,
    "no_object": 4,
}

cloud_ProtectiveGearType_converter = {
    "unknown": 0,
    "hard_hat": 1,
    "no_hard_hat": 2,
    "helmet": 1,
    "no_helmet": 2,
    "safety_vest": 3,
    "no_safety_vest": 4,
}

cloud_AccessoryTypes_converter = {
    "unknown": 0,
    "hat": 1,
    "no_glasses": 2,
    "no_hat": 3,
    "glasses": 4,
}

cloud_AgeTypes_converter = {
    "unknown": 0,
    "less_18": 1,
    "18_to_60": 2,
    "over_60": 3,
}

cloud_GenderTypes_converter = {
    "unknown": 0,
    "male": 1,
    "female": 2,
}

cloud_FaceConfidence_converter = {
    "none": 1,
    "low": 3,
    "medium": 5,
    "high": 7,
}

cloud_AttributeConfidence_converter = {
    AttrConfidence.LOW: 3,
    AttrConfidence.MEDIUM: 5,
    AttrConfidence.HIGH: 7,
}


class IdentityMapper:
    def get(self, x, *args, **kwargs):
        return x


identity_mapper = IdentityMapper()

converter_map = {
    "globalType": cloud_GlobalTypes_converter,
    "colors": cloud_ColorTypes_converter,
    "footwearColor": cloud_ColorTypes_converter,
    "lowerbodyColor": cloud_ColorTypes_converter,
    "upperbodyColor": cloud_ColorTypes_converter,
    "hairColor": cloud_ColorTypes_converter,
    "genderType": cloud_GenderTypes_converter,
    "lowerbodyType": cloud_LowerbodyTypes_converter,
    "upperbodyType": cloud_UpperbodyTypes_converter,
    "carryingType": cloud_CarryingTypes_converter,
    "accessoryType": cloud_AccessoryTypes_converter,
    "ageType": cloud_AgeTypes_converter,
    "protectiveGearType": cloud_ProtectiveGearType_converter,
}

alert_person_conversion_map = {
    "genderType": "person_gender",
    "ageType": "person_age",
    "upperbodyType": "person_upper_body_type",
    "lowerbodyType": "person_lower_body_type",
    "upperbodyColor": "person_upper_body_color",
    "lowerbodyColor": "person_lower_body_color",
    "hairColor": "person_hair_color",
    "accessoryType": "person_accessory_type",
    "carryingType": "person_carrying_type",
    "protectiveGearType": "person_protective_gear",
}

alert_vehicle_conversion_map = {
    "make": "vehicle_make",
    "model": "vehicle_model",
    "type": "vehicle_type",
    "colors": "vehicle_color",
    "plate": "vehicle_license_plate",
    "region": "vehicle_license_plate_region",
    "serialNumber": "container_serial",
}


attributes_to_filter = {"perimeter", "position", "markedPositions"}
description_to_promote = {"globalType": "globalType", "plate": "licensePlate"}


def list_dict_to_dict_list(list_of_dicts: List[Dict]) -> Dict[str, List[Dict]]:
    dict_of_lists = defaultdict(list)
    for d in list_of_dicts:
        for key, value in d.items():
            dict_of_lists[key].append(value)
    return dict(dict_of_lists)  # Convert defaultdict back to dict if needed


def convert_to_trackers(info: List[Dict], attributes: Dict) -> List[Dict]:
    info_dict = {obj["idIndex"]: obj for obj in info}
    keys = set(info_dict.keys())
    keys.update(attributes.keys())
    trackers = []
    for key in keys:
        tracker = copy(info_dict.get(key, {}))
        attrs = deepcopy(attributes.get(key, {}))

        # convert to cloud enums and structure
        if attrs:
            desc = attrs.pop("description", {})
            if attrs.get("type", "") == "vehicle" and "type" in desc:
                for t in desc["type"]:
                    t.value = cloud_VehicleTypes_converter.get(t.value, 0)

            keys = list(desc.keys())
            for category in keys:
                value = desc.pop(category)
                if isinstance(value, list):
                    category_values = []
                    for t in value:
                        category_values.append(
                            {
                                "value": converter_map.get(category, identity_mapper).get(t.value, 0),
                                "confidence": cloud_AttributeConfidence_converter.get(t.confidence, 0),
                                "score": t.score,
                            }
                        )
                    if category in description_to_promote:
                        attrs[description_to_promote[category]] = category_values[0].get("value")
                    desc[category] = category_values
                else:
                    attrs[category] = value
            tracker["attributes"] = desc
            if "faceConfidence" in attrs:
                attrs["faceConfidence"] = cloud_FaceConfidence_converter.get(attrs["faceConfidence"], 0)

            tracker.update(attrs)
        for category in attributes_to_filter:
            tracker.pop(category, None)
        if "type" in tracker:
            tracker["type"] = cloud_TrackerTypes_converter.get(tracker["type"], 0)
        trackers.append(tracker)
    return trackers


def convert_to_alert_data(attributes: Dict) -> Dict:
    alert_desc = {}
    desc = attributes.pop("description", {})
    if attributes.get("type", "") == "vehicle":
        conversion_map = alert_vehicle_conversion_map
    elif attributes.get("type", "") == "person":
        conversion_map = alert_person_conversion_map
    else:
        conversion_map = {}
    for category, prop_value in desc.items():
        if category in conversion_map:
            alert_desc[conversion_map[category]] = [p.value for p in prop_value if p.confidence == AttrConfidence.HIGH]
    custom_objects = attributes.get("customObjects", [])
    if custom_objects:
        alert_desc["custom_objects_ids"] = []
    for obj in custom_objects:
        co_id = obj.get("value")
        alert_desc["custom_objects_ids"].append(co_id)
    return alert_desc


def convert_custom_object(custom_object_ids: List[int]) -> Dict:
    object_list = []
    conf = cloud_AttributeConfidence_converter[AttrConfidence.HIGH]
    for obj_id in custom_object_ids:
        object_list.append({"confidence": conf, "score": 1, "value": int(obj_id)})
    return {"customObjects": object_list}
