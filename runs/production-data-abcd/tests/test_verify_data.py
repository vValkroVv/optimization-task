"""Behavior checks for the production loader census and rejection gates."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('verify_data', ROOT / 'verify_data.py')
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class CensusTests(unittest.TestCase):
    def test_owner_raw_and_unique(self):
        stats, unique = v.census(np.array([0, 0, 16032, 128255]))
        self.assertEqual(unique.tolist(), [0, 16032, 128255])
        self.assertEqual(stats['owner_raw_requests'], [2, 1, 0, 0, 0, 0, 0, 1])
        self.assertEqual(stats['owner_unique_requests'], [1, 1, 0, 0, 0, 0, 0, 1])

    def test_reject_bounds_and_duplicate_consumption(self):
        for tokens in ([-1], [128256], []):
            with self.assertRaises(AssertionError):
                v.census(np.array(tokens, dtype=np.int64))
        with self.assertRaises(AssertionError):
            v.assert_unique_rows([1, 1], 2)
        v.assert_unique_rows([1, 2], 2)

    def test_manifest_rejects_duplicate_prepared_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'rows.jsonl'
            rows = [{'index': i, 'domain': v.c.DOMAIN_ORDER[i % 4], 'sha256': 'same'} for i in range(4)]
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            with patch.object(v.c, 'ROW_MANIFEST', path), patch.object(v.c, 'NUM_ROWS', 4):
                with self.assertRaisesRegex(AssertionError, 'duplicate prepared'):
                    v.checked_manifest()

    def test_global_summary_and_cross_rank_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for rank in range(8):
                raw = [0] * 8
                raw[rank] = 8192
                records = [{'row_indices': [rank * 4 + i * 2, rank * 4 + i * 2 + 1],
                            'owner_raw_requests': raw} for i in range(2)]
                payload = {'status': 'PASS', 'rank': rank, 'cuda_initialized': False,
                           'workers_stopped': True, 'records': records}
                (directory / f'rank{rank}.json').write_text(json.dumps(payload))
                np.savez(directory / f'rank{rank}-step-tokens.npz',
                         **{'0': np.array([rank * 16032], dtype=np.int64)})
            result = v.summarize(directory, 2)
            self.assertEqual(result['global_unique_rows'], 32)
            self.assertEqual(result['physical_tokens'], 32 * 4096)
            self.assertEqual(result['main_valid_targets'], 32 * 4095)
            self.assertEqual(result['updates'][0]['global_unique_token_ids'], 8)
            self.assertEqual(result['updates'][0]['owner_raw_requests'], [16384] * 8)
            self.assertEqual(result['updates'][0]['owner_raw_max_share'], 0.125)
            path = directory / 'rank1.json'
            payload = json.loads(path.read_text())
            payload['records'][0]['row_indices'][0] = 0
            path.write_text(json.dumps(payload))
            with self.assertRaises(AssertionError):
                v.summarize(directory, 2)

    def test_native_collator_and_shift_position_rejection(self):
        import torch
        from torchtitan.datasets.streaming.text.collators import TextPretrainCollator

        first = torch.arange(4096, dtype=torch.int64) + 3
        first[7] = 2
        second = first + 4096
        native = TextPretrainCollator(seq_len=4096, eos_token_id=2)
        inputs, labels = native([{'tokens': first}, {'tokens': second}])
        lookup = {v.row_hash(row): {'index': i, 'domain': 'fixture', 'sha256': v.row_hash(row)}
                  for i, row in enumerate(inputs['input'].numpy())}
        stats, _ = v.check_batch((inputs, labels, 0, 123), lookup, 2)
        self.assertEqual(stats['main_valid_targets'], 8190)
        self.assertEqual(inputs['positions'][0, 8].item(), 0)
        labels[0, 0] += 1
        with self.assertRaises(AssertionError):
            v.check_batch((inputs, labels, 0, 123), lookup, 2)
        labels[0, 0] -= 1
        inputs['positions'][0, 8] += 1
        with self.assertRaises(AssertionError):
            v.check_batch((inputs, labels, 0, 123), lookup, 2)

    def test_fresh_order_includes_labels_and_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for depth in (0, 32):
                (root / str(depth)).mkdir()
                for rank in range(8):
                    record = {'row_indices': [rank], 'row_hashes': [str(rank)],
                              'labels_sha256': 'labels', 'positions_sha256': 'positions', 'worker_pid': depth}
                    (root / str(depth) / f'rank{rank}.json').write_text(json.dumps({'records': [record]}))
            self.assertEqual(v.order_signature(root / '0'), v.order_signature(root / '32'))
            path = root / '32/rank0.json'
            payload = json.loads(path.read_text())
            payload['records'][0]['positions_sha256'] = 'changed'
            path.write_text(json.dumps(payload))
            self.assertNotEqual(v.order_signature(root / '0'), v.order_signature(root / '32'))


if __name__ == '__main__':
    unittest.main()
