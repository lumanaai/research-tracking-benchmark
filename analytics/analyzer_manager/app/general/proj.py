import json
import os
import pathlib


def is_in_docker() -> bool:
    return load_bool_from_env("IN_DOCKER", False)


def project_path() -> str:
    this_path = pathlib.Path(__file__)
    return this_path.parent.parent.parent.parent.resolve()


def analyzer_path(postfix_path: str = None) -> str:
    this_path = pathlib.Path(__file__)
    prefix = str(this_path.parent.parent.resolve())
    if postfix_path:
        return os.path.join(prefix, postfix_path)
    else:
        return prefix


def cuda_path(*args) -> str:
    return _build_path("cuda_library", *args)


def resource_path(*args) -> str:
    return _build_path("resources", *args)


def info_path(*args) -> str:
    return _build_path("assets", *args)


def local_weights_path(nn_name: str) -> str:
    return _build_path("inference", "model_repository", nn_name, "1")


def local_inference_repository_path(*args) -> str:
    return _build_path("inference", "model_repository", *args)


def _build_path(second_level: str, *args):
    if len(args) == 0:
        args = []

    path_parts = [_base_path(), second_level] + list(args)
    return str(os.path.join(*path_parts))


def _base_path() -> str:
    this_path = pathlib.Path(__file__)
    return str(this_path.parent.parent.parent.resolve())


def assets_path() -> str:
    return analyzer_path("assets")


def load_config():
    prod_path = "/usr/src/app/configs/config.json"

    if is_in_docker():
        return json.load(open(prod_path))

    else:
        dev_path = os.path.join(_base_path(), "assets", "config.json")
        return json.load(open(dev_path))


def load_configAnalytic(analytics_file_name: str = None):
    if analytics_file_name is None:
        prod_path = "/usr/src/app/configs/configAnalytic.json"
        if is_in_docker():
            config = json.load(open(prod_path))
        else:
            dev_path = os.path.join(_base_path(), "assets", "configAnalytic.json")
            config = json.load(open(dev_path))
    else:
        config = json.load(open(analytics_file_name))

    # versioning control every "if" advances the versions so we can move from each version to the current version
    if "version" not in config:
        config["version"] = 1
        if "pa" in config["l1_models"]["models"]:
            config["l1_models"]["models"] = list(
                map(lambda x: x.replace("pa", "human_parsing"), config["l1_models"]["models"])
            )
        if "pa" in config["l1_models"]["attributes"]:
            pa_attr = config["l1_models"]["attributes"]["pa"]
            del config["l1_models"]["attributes"]["pa"]
            config["l1_models"]["attributes"]["human_parsing"] = {"pa": pa_attr}
        if "reid" not in config["l1_models"]["attributes"]:
            config["l1_models"]["attributes"]["human_parsing"]["reid"] = {"enable": True}
    if config["version"] < 1.1:
        container_configs = {"containerTokenEnable": False, "containertoken": ""}
        for key in container_configs:
            if key not in config["l1_models"]["attributes"]["alpr"]:
                config["l1_models"]["attributes"]["alpr"][key] = container_configs[key]
        if "weights" in config["l1_models"]["attributes"]["human_parsing"]["reid"]:
            if config["l1_models"]["attributes"]["human_parsing"]["reid"]["weights"] == "reid_resnet50_fp16_b2.engine":
                del config["l1_models"]["attributes"]["human_parsing"]["reid"]["weights"]
        config["version"] = 1.1
    if config["version"] < 2.0:
        subcls_to_str_mapping = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            15: "cat",
            16: "dog",
        }

        def convert_subcls_arr(conf_key: str):
            if conf_key in config and "subClasses" in config[conf_key]:
                config[conf_key]["subClasses"] = [subcls_to_str_mapping[i] for i in config[conf_key]["subClasses"]]

        for k in ["l0_detect", "l0_track"]:
            convert_subcls_arr(k)
        config["version"] = 2.0
    if config["version"] < 2.1:
        alpr_change_dict = {
            "min_lpr_width": 70,
            "min_lpr_conf": 0,
            "min_blur_th": 0.375,
        }
        for key in alpr_change_dict.keys():
            if key not in config["l1_models"]["attributes"]["alpr"]:
                config["l1_models"]["attributes"]["alpr"][key] = alpr_change_dict[key]
        config["version"] = 2.1
    if config["version"] < 2.2:
        multiplier = 150  # default
        thumb_duration = 2000  # default
        if "snapshotMultiplier" in config["thumbnailPolicy"]:
            multiplier = config["thumbnailPolicy"]["snapshotMultiplier"]
        if "thumbnailsDuration" in config["thumbnailPolicy"]:
            thumb_duration = config["thumbnailPolicy"]["thumbnailsDuration"]
        config["thumbnailPolicy"]["snapshotDuration"] = multiplier * thumb_duration
        config["version"] = 2.2
    if config["version"] < 2.3:
        if "pre_process" in config:
            config["pre_process"]["checkImageSize"] = False
        config["version"] = 2.3
    if config["version"] < 2.4:
        def_bl = {
            "enable_blacklist": False,
            "blacklist_iou": 0.9,
            "flag_blacklist": True,
            "blacklist_span": 1,
            "blacklist_encode": 0,
        }
        if "entityManagement" not in config:
            config["entityManagement"] = def_bl
        config["version"] = 2.4
    if config["version"] < 2.5:
        if "alertConfig" in config:
            if "success_filter" not in config["alertConfig"]:
                config["alertConfig"]["success_filter"] = False
        else:
            config["alertConfig"] = {"success_filter": False}
        config["version"] = 2.5
    if config["version"] < 2.6:
        config["entityManagement"]["enable_blacklist"] = True
        config["version"] = 2.6
    if config["version"] < 2.7:
        config["l1_models"]["attributes"]["human_parsing"]["ppe"] = {"enable": False}
        config["l1_models"]["attributes"]["doors"] = {"enable": False}
        config["version"] = 2.7
    if config["version"] < 2.8:
        config["load_balance"] = {"mean_skip": 2, "low_motion_th": 15, "high_motion_th": 50}
        if config["motionPolicy"].get("motionVectorsDuration", 20000) == 2000:
            config["motionPolicy"]["motionVectorsDuration"] = 5000
        config["thumbnailPolicy"].update({"mandatoryThumbnailsMultiplier": 1, "minMotionForOptionalThumbnail": 20})
        config["version"] = 2.8
    if config["version"] < 2.9:
        config["trainThumbPolicy"]["useNewSelector"] = True
        config["trainThumbPolicy"]["resize"] = -1
        config["trainThumbPolicy"]["TrainThumbnailDurationCounter"] = {
            "1000": 250,
            "3000": 400,
            "5000": 800,
            "after": 1600,
        }
        config["version"] = 2.9
    if config["version"] < 2.91:
        config["l1_models"]["attributes"]["alpr"]["alprCloudURL"] = "https://lpr.lumix.ai/v1/plate-reader/"
        config["version"] = 2.91
    if config["version"] < 3.1:
        config["preprocess"] = {"name": "cv2"}
        config["version"] = 3.1
    if config["version"] < 3.2:
        config["offline_analytics"] = {"weapon_detection": {"enable": True}}
        config["version"] = 3.2
    if config["version"] < 3.3:
        alpr_conf = config["l1_models"]["attributes"]["alpr"]
        config["l1_models"]["models"].append("container")
        config["l1_models"]["attributes"]["container"] = {
            "api_token": alpr_conf.get("containertoken", "3dec628a0ed8dd1bf83aaa046a8c7487060373f6"),
            "api_url": alpr_conf.get("containerCloudURL", "https://container.lumana.ai/api/v1/predict/"),
        }
        tracker_config = config["l0_track"]
        if "untrack_objects" in tracker_config:
            untrack_objects = tracker_config["untrack_objects"]
            if 8 in untrack_objects:
                untrack_objects.remove(8)
            config["l0_track"]["untrack_objects"] = untrack_objects
        config["version"] = 3.3
    if config["version"] < 3.4:
        thumb_config = config["trainThumbPolicy"]
        thumb_config["crops_ar"] = {"person": 0.625, "vehicle": 1, "face": 1, "license_plate": 2}
        config["version"] = 3.4
    if config["version"] < 3.5:
        config["clip"] = config.get("clip", {"enable": False})
        config["version"] = 3.5
    if config["version"] < 3.6:
        thumb_config = config["trainThumbPolicy"]
        crops_ar = thumb_config.get("crops_ar", {})
        new_keys = {"pet": 1, "shoppingcart": 1}
        for k, v in new_keys.items():
            if k not in crops_ar:
                crops_ar[k] = v
        thumb_config["crops_ar"] = crops_ar
        thumb_config["crop_bw_limits_kb"] = {"350": 70, "500": 50, "750": 30, "100000": 10}
        config["version"] = 3.6
    if config["version"] < 3.7:
        config["clip"] = {"enable": True, "encode_thumbnails": True, "encode_crops": False}
        config["version"] = 3.7
    if config["version"] < 3.8:
        config["countingPolicy"] = {"updateRate": 60}
        config["version"] = 3.8
    if config["version"] < 3.81:
        config["trainThumbPolicy"].update(
            {
                "clip_encode_crops": True,
                "clip_crop_queue_size": 30,
                "clip_crop_rate_limit_sec": 0.5,
            }
        )
        config["version"] = 3.81
    if config["version"] < 3.82:
        internal_alert_config = {
            "internal_alerts": True,
            "internal_alert_type_prob": {"weapon": 0.2, "fire": 0.1, "falling": 0.4, "violence": 0.3},
            "internal_alert_ratio": 0.75,
        }
        config["alertConfig"].update(internal_alert_config)
        config["version"] = 3.82
    if config["version"] < 3.83:
        config["image-quality"] = {"enable": True, "accumulation_period": 288}
        config["version"] = 3.83
    if config["version"] < 3.84:
        if "offline_analytics" in config and "weapon_detection" in config["offline_analytics"]:
            val = config["offline_analytics"]["weapon_detection"].get("min_weapon_confidence", 0.15)
            if val != 0.15:  # default, remove it
                del config["offline_analytics"]["weapon_detection"]["min_weapon_confidence"]
        config["version"] = 3.84
    if config["version"] < 3.85:
        config["alertConfig"]["enable_routing_for_custom_objects"] = False
        config["version"] = 3.85
    if config["version"] < 3.86:
        config["countingPolicy"]["regionUpdateRate"] = 60
        config["version"] = 3.86
    return config


def export_formats():
    import pandas as pd

    # YOLOv5 export formats
    x = [
        ["PyTorch", "-", ".pt", True],
        ["TorchScript", "torchscript", ".torchscript", True],
        ["ONNX", "onnx", ".onnx", True],
        ["OpenVINO", "openvino", "_openvino_model", False],
        ["TensorRT", "engine", ".engine", True],
        ["CoreML", "coreml", ".mlmodel", False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True],
        ["TensorFlow GraphDef", "pb", ".pb", True],
        ["TensorFlow Lite", "tflite", ".tflite", False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False],
        ["TensorFlow.js", "tfjs", "_web_model", False],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "GPU"])


def load_bool_from_env(field_name: str, default: bool = False):
    retval = default
    if os.getenv(field_name):
        retval = os.getenv(field_name).lower().strip("'") == "true"
    return retval


def load_str_from_env(field_name: str):
    retval = ""
    if os.getenv(field_name):
        retval = os.getenv(field_name).lower()
    return retval


def set_environment_var():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


console_log = load_bool_from_env("CONSOLE_LOG", False)


def is_jetson_platform():
    jetson = load_bool_from_env("JETSON", None)
    if jetson is None:
        jetson = get_host() in ["xavier", "orin", "orin-jp5", "orin-jp6"]
    return jetson


def get_host() -> str:
    return load_str_from_env("HOST")


def is_cv2_cuda_available():
    try:
        import cv2

        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except Exception:
        return False
