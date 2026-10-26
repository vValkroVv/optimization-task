from typing import List, Sequence, Tuple


def stage_layer_index(
    layer_ids: Sequence[int], start_layer: int, end_layer: int
) -> Tuple[int, int]:
    offset = sum(1 for i in layer_ids if i < start_layer)
    count = sum(1 for i in layer_ids if start_layer <= i < end_layer)
    return offset, count


def mamba_pp_dst_indices(
    num_src_ptrs: int, num_dst_ptrs: int, num_local_layers: int, layer_offset: int
) -> List[int]:
    if num_dst_ptrs in (0, num_src_ptrs) or num_local_layers <= 0 or num_src_ptrs == 0:
        return list(range(num_dst_ptrs))
    num_tensors = num_src_ptrs // num_local_layers
    assert (
        num_tensors > 0
        and num_src_ptrs % num_local_layers == 0
        and num_dst_ptrs % num_tensors == 0
    ), (
        f"mamba state layout mismatch: src={num_src_ptrs}, dst={num_dst_ptrs}, "
        f"local_layers={num_local_layers}"
    )
    stride = num_dst_ptrs // num_tensors
    assert layer_offset + num_local_layers <= stride, (
        f"mamba PP window [{layer_offset}, {layer_offset + num_local_layers}) "
        f"exceeds dst layers ({stride})"
    )
    return [
        j * stride + layer_offset + k
        for j in range(num_tensors)
        for k in range(num_local_layers)
    ]
