def apply_from_dict(key, alert_dict, default_value):
    if key in alert_dict and alert_dict[key] is not None:
        return alert_dict[key]
    else:
        return default_value


def apply_from_dict_safe(dict_, key_, default):
    if dict_ is None:
        return default
    ret_val = dict_.get(key_, None)
    if not bool(ret_val):
        return default
    return ret_val


def trim_str(v: str, max_len: int = 256, suffix_len: int = 20) -> str:
    if len(v) <= max_len:
        return v
    prefix_len = max_len - suffix_len - 3  # 3 for "..."
    return v[:prefix_len] + "..." + v[-suffix_len:]
