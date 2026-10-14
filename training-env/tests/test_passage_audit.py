"""Tiny CPU fixtures for the report-only passage audit."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import test_data_semantics as fixtures
import audit_passage_overlap as passage

class PassageAuditTests(unittest.TestCase):
    setUp = fixtures.DataSemanticTests.setUp
    tearDown = fixtures.DataSemanticTests.tearDown
    write_real = fixtures.DataSemanticTests.write_real
    # Reuse the canonical real-data fixture builder, without inheriting its tests.
    def test_common_prefix_exact_passage_and_no_mutation(self):
        stories = {k: list(v) for k,v in self.base_stories.items()}
        stories['train'][0] = 'shared passage ' * 8 + 'train ending'
        stories['dev'][0] = 'shared passage ' * 8 + 'dev ending'
        self.write_real(stories)
        result = passage.audit(self.root, profile='smoke')
        self.assertGreater(result['split_pairs']['train:dev']['shared_unique_windows'], 0)
        self.assertTrue(result['arrays_unchanged'])
        self.assertEqual(len(result['audit_source_sha256']),64)
        self.assertEqual(result['status'], 'complete_sampled_scan')
        self.assertIn('Stride sampling can miss shifted', result['limitations'][0])
        self.assertLessEqual(len(result['split_pairs']['train:dev']['examples']), 10)

    def test_short_stories_and_max_budget(self):
        result = passage.audit(self.root, profile='smoke', window=1000)
        self.assertEqual(sum(result['scanned_windows'].values()), 0)
        self.assertEqual(sum(result['skipped_short_stories'].values()), 9)
        limited = passage.audit(self.root, profile='smoke', window=16, max_windows=1)
        self.assertEqual(limited['status'], 'partial')
        self.assertEqual(limited['stop_reason'], 'max_windows')
        self.assertEqual(sum(limited['scanned_windows'].values()), 1)

    def test_digest_collision_does_not_claim_overlap(self):
        with patch.object(passage, 'window_digest', return_value=b'x'*16):
            result = passage.audit(self.root, profile='smoke', window=64)
        self.assertEqual(sum(p['shared_unique_windows'] for p in result['split_pairs'].values()), 0)

    def test_eot_is_derived_from_validated_tokenizer(self):
        import json
        from tokenizers import Tokenizer
        import numpy as np
        spec = json.loads(self.tokenizer.to_str())
        spec['model']['vocab']['<|endoftext|>'] = 3000
        spec['model']['vocab']['<unused-3000>'] = 0
        spec['added_tokens'][0]['id'] = 3000
        tokenizer = Tokenizer.from_str(json.dumps(spec))
        stories = {k: list(v) for k,v in self.base_stories.items()}
        stories['train'][0] = 'shared passage ' * 8 + 'train ending'
        stories['dev'][0] = 'shared passage ' * 8 + 'dev ending'
        arrays = {}
        for split, docs in stories.items():
            ids = []
            for text in docs: ids.extend(tokenizer.encode(text).ids); ids.append(3000)
            arrays[split] = np.asarray(ids,dtype=np.uint16)
        self.write_real(stories, arrays=arrays, tokenizer_json=tokenizer.to_str())
        result = passage.audit(self.root,profile='smoke')
        self.assertEqual(result['eot_id'],3000)
        self.assertGreater(result['split_pairs']['train:dev']['shared_unique_windows'],0)

    def test_invalid_parameters_reject_before_loading_assets(self):
        for kwargs in ({'window':1.5},{'stride':True},{'max_windows':True},{'timeout_seconds':float('nan')},{'timeout_seconds':float('inf')},{'timeout_seconds':-1}):
            with self.subTest(kwargs=kwargs),patch.object(passage,'load_data') as loader:
                with self.assertRaises(ValueError):passage.audit(self.root,**kwargs)
                loader.assert_not_called()

    def test_mutated_asset_snapshot_is_not_reported_as_valid_audit(self):
        original=passage.file_hash
        calls=[0]
        def changed(path):
            calls[0]+=1
            return original(path) if calls[0]<=5 else '0'*64
        with patch.object(passage,'file_hash',side_effect=changed):
            result=passage.audit(self.root,profile='smoke')
        self.assertFalse(result['assets_unchanged'])
        self.assertEqual(result['status'],'invalid_assets')

    def test_synthetic_labeled(self):
        self.write_real(self.base_stories, kind='synthetic')
        result = passage.audit(self.root, profile='smoke')
        self.assertTrue(result['synthetic'])
        self.assertEqual(result['data_kind'], 'synthetic')

if __name__ == '__main__': unittest.main()
