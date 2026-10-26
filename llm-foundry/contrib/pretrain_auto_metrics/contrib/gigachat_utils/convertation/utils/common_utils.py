import typing as tp


def copy_dict_excluding_keys(data: tp.Dict, keys_to_ignore: tp.List[str], prefix: str = ""):
    if not isinstance(data, dict):
        return data
    result = {}
    for key, value in data.items():
        key_with_prefix = f"{prefix}.{key}" if prefix else f"{key}"
        if key_with_prefix in keys_to_ignore:
            continue
        if isinstance(value, dict):
            result[key] = copy_dict_excluding_keys(value, keys_to_ignore, prefix=key_with_prefix)
        else:
            result[key] = value
    return result