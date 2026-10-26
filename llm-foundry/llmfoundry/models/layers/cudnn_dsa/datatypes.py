"""torch -> cutlass dtype conversion for the vendored DSA CuTeDSL kernels."""

torch_available = None
cutlass_available = None
_torch_to_cutlass_data_type_dict = None


def is_torch_available():
    global torch_available
    if torch_available is None:
        try:
            import torch  # noqa: F401

            torch_available = True
        except ImportError:
            torch_available = False
    return torch_available


def is_cutlass_available():
    global cutlass_available, _torch_to_cutlass_data_type_dict
    if cutlass_available is None:
        try:
            import torch
            import cutlass

            cutlass_available = True
            mapping = {
                torch.half: getattr(cutlass, "Float16", None),
                getattr(torch, "float16", torch.half): getattr(cutlass, "Float16", None),
                getattr(torch, "bfloat16", None): getattr(cutlass, "BFloat16", None),
                torch.float: getattr(cutlass, "Float32", None),
                getattr(torch, "float32", torch.float): getattr(cutlass, "Float32", None),
                torch.double: getattr(cutlass, "Float64", None),
                getattr(torch, "float64", torch.double): getattr(cutlass, "Float64", None),
                getattr(torch, "int8", None): getattr(cutlass, "Int8", None),
                getattr(torch, "int32", None): getattr(cutlass, "Int32", None),
                getattr(torch, "int64", None): getattr(cutlass, "Int64", None),
                getattr(torch, "uint8", None): getattr(cutlass, "Uint8", None),
                getattr(torch, "bool", None): getattr(cutlass, "Boolean", None),
                getattr(torch, "float8_e4m3fn", None): getattr(cutlass, "Float8E4M3FN", None),
                getattr(torch, "float8_e5m2", None): getattr(cutlass, "Float8E5M2", None),
                getattr(torch, "float8_e8m0fnu", None): getattr(cutlass, "Float8E8M0FNU", None),
                getattr(torch, "float4_e2m1fn_x2", None): getattr(cutlass, "Float4E2M1FN", None),
            }
            _torch_to_cutlass_data_type_dict = {t: c for t, c in mapping.items() if t is not None and c is not None}
        except ImportError:
            cutlass_available = False
            _torch_to_cutlass_data_type_dict = {}
    return cutlass_available


def _torch_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2: bool = False):
    if is_cutlass_available() and is_torch_available():
        import torch

        if interpret_uint8_as_fp4x2 and data_type == torch.uint8:
            import cutlass

            return getattr(cutlass, "Float4E2M1FN", None)
        else:
            return _torch_to_cutlass_data_type_dict.get(data_type, None)
    return None


def _convert_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2: bool = False):
    if is_cutlass_available():
        import cutlass

        if isinstance(data_type, type) and issubclass(data_type, cutlass.Numeric):
            return data_type
        elif data_type is not None:
            cutlass_data_type = _torch_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2)
            if cutlass_data_type is None:
                raise ValueError("Unsupported tensor data type.")
            return cutlass_data_type
        else:
            raise ValueError("None is not a valid tensor data type.")
    return None
