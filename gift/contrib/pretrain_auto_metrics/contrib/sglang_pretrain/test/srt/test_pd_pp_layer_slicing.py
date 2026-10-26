import importlib.util
import os
import unittest

_PP_LAYER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python",
    "sglang",
    "srt",
    "disaggregation",
    "pp_layer.py",
)
_spec = importlib.util.spec_from_file_location("pp_layer", _PP_LAYER)
pp_layer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp_layer)

NUM_LAYERS = 64
FULL_ATTN_IDS = [i for i in range(NUM_LAYERS) if i % 4 == 3]
MAMBA_IDS = [i for i in range(NUM_LAYERS) if i % 4 != 3]


def uniform_pp_ranges(num_layers, pp_size):
    base, rem = divmod(num_layers, pp_size)
    ranges, start = [], 0
    for rank in range(pp_size):
        n = base + (1 if rank >= pp_size - rem else 0)
        ranges.append((start, start + n))
        start += n
    return ranges


class TestFullAttentionPPSlicing(unittest.TestCase):
    def _dst_slice(self, start_index, num_src):
        dst_ptrs = list(FULL_ATTN_IDS)
        return dst_ptrs[start_index : start_index + num_src]

    def test_pp8_every_stage_hits_its_own_layers(self):
        for pp_rank, (start, end) in enumerate(uniform_pp_ranges(NUM_LAYERS, 8)):
            local = [i for i in FULL_ATTN_IDS if start <= i < end]
            offset, count = pp_layer.stage_layer_index(FULL_ATTN_IDS, start, end)
            self.assertEqual(count, len(local))
            self.assertEqual(offset, 2 * pp_rank)
            self.assertEqual(self._dst_slice(offset, count), local)

    def test_pp8_old_global_layer_id_was_wrong(self):
        _, (start, end) = 6, uniform_pp_ranges(NUM_LAYERS, 8)[6]
        self.assertNotEqual(
            self._dst_slice(start, 2), [i for i in FULL_ATTN_IDS if start <= i < end]
        )
        start7, _ = uniform_pp_ranges(NUM_LAYERS, 8)[7]
        self.assertEqual(self._dst_slice(start7, 2), [])

    def test_pp1_is_identity(self):
        offset, count = pp_layer.stage_layer_index(FULL_ATTN_IDS, 0, NUM_LAYERS)
        self.assertEqual((offset, count), (0, 16))
        self.assertEqual(self._dst_slice(offset, count), FULL_ATTN_IDS)

    def test_uneven_split_pp5(self):
        seen = []
        for start, end in uniform_pp_ranges(NUM_LAYERS, 5):
            offset, count = pp_layer.stage_layer_index(FULL_ATTN_IDS, start, end)
            seen += self._dst_slice(offset, count)
        self.assertEqual(seen, FULL_ATTN_IDS)


class TestMambaPPSlicing(unittest.TestCase):
    NUM_TENSORS = 3

    def _dst(self, layer_ids):
        return [(j, i) for j in range(self.NUM_TENSORS) for i in layer_ids]

    def _src(self, layer_ids):
        return self._dst(layer_ids)

    def test_pp8_maps_to_matching_global_slots(self):
        dst = self._dst(MAMBA_IDS)
        for pp_rank, (start, end) in enumerate(uniform_pp_ranges(NUM_LAYERS, 8)):
            local = [i for i in MAMBA_IDS if start <= i < end]
            offset, count = pp_layer.stage_layer_index(MAMBA_IDS, start, end)
            self.assertEqual((offset, count), (6 * pp_rank, 6))
            src = self._src(local)
            picked = pp_layer.mamba_pp_dst_indices(len(src), len(dst), count, offset)
            self.assertEqual([dst[i] for i in picked], src)

    def test_pp1_is_identity(self):
        dst = self._dst(MAMBA_IDS)
        offset, count = pp_layer.stage_layer_index(MAMBA_IDS, 0, NUM_LAYERS)
        picked = pp_layer.mamba_pp_dst_indices(len(dst), len(dst), count, offset)
        self.assertEqual(picked, list(range(len(dst))))

    def test_uneven_split_pp5_covers_every_slot_once(self):
        dst = self._dst(MAMBA_IDS)
        seen = []
        for start, end in uniform_pp_ranges(NUM_LAYERS, 5):
            local = [i for i in MAMBA_IDS if start <= i < end]
            offset, count = pp_layer.stage_layer_index(MAMBA_IDS, start, end)
            picked = pp_layer.mamba_pp_dst_indices(
                len(self._src(local)), len(dst), count, offset
            )
            self.assertEqual([dst[i] for i in picked], self._src(local))
            seen += picked
        self.assertEqual(sorted(seen), list(range(len(dst))))

    def test_window_past_dst_end_raises(self):
        with self.assertRaises(AssertionError):
            pp_layer.mamba_pp_dst_indices(18, 144, 6, 44)

    def test_no_mamba_state_is_identity(self):
        self.assertEqual(pp_layer.mamba_pp_dst_indices(0, 0, 0, 0), [])

    def test_missing_dst_registration_is_noop(self):
        self.assertEqual(pp_layer.mamba_pp_dst_indices(18, 0, 6, 12), [])


if __name__ == "__main__":
    unittest.main()
