"""Process-owned validation cache cannot replace byte and semantic checks."""
import json
import unittest
from unittest.mock import patch
import numpy as np
from test_data_semantics import DataSemanticTests, encode_stories
import data

class ValidationCacheTests(DataSemanticTests):
    def setUp(self):
        super().setUp()
        data._SEMANTIC_CACHE.clear()

    def load(self, **kwargs):
        return data.load_data(self.root, profile='smoke', **kwargs)[3]['validation']

    def test_warm_report_is_copied_and_bypass_recomputes(self):
        first = self.load()
        self.assertEqual(first['cache_status'], 'cold')
        first['story_counts']['train'] = 999
        with patch.object(data, '_story_validation', wraps=data._story_validation) as scan:
            second = self.load()
            self.assertEqual(second['cache_status'], 'hit')
            self.assertEqual(second['story_counts']['train'], 3)
            self.assertEqual(scan.call_count, 0)
            self.assertEqual(self.load(use_cache=False)['cache_status'], 'bypassed')
            self.assertEqual(scan.call_count, 1)
        self.assertEqual(first['provenance'], second['provenance'])

    def test_warm_cache_does_not_accept_story_count_mutation(self):
        self.load()
        manifest = json.loads((self.root/'manifest.json').read_text())
        manifest['story_counts']['train'] += 1
        (self.root/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'story_counts'):
            self.load()

    def test_warm_cache_rehashed_tokenizer_change_rejected(self):
        self.load()
        config = json.loads(self.tokenizer.to_str())
        config['pre_tokenizer'] = {'type': 'Whitespace'}
        self.write_real(self.base_stories, tokenizer_json=json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'ByteLevel'):
            self.load()

    def test_warm_cache_rehashed_cross_dtype_overlap_rejected(self):
        self.load()
        stream = encode_stories(self.tokenizer, self.base_stories['train'])
        self.write_real(self.base_stories, arrays={s:stream.astype(dtype) for s,dtype in
            zip(data.FILES, (np.uint16,np.uint32,np.uint64))})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load()

    def test_warm_cache_rehashed_decoded_overlap_rejected(self):
        self.load()
        stories = {s:list(v) for s,v in self.base_stories.items()}
        stories['dev'][0] = stories['train'][0].replace(' ', '  ')
        self.write_real(stories)
        with self.assertRaisesRegex(ValueError, 'decoded normalized text'):
            self.load()

    def test_rehashed_partial_story_overlap_recomputes_with_documented_limit(self):
        self.load()
        stories = {s:list(v) for s,v in self.base_stories.items()}
        stories['dev'][0] = stories['train'][0] + 'different ending'
        self.write_real(stories)
        result = self.load()
        self.assertEqual(result['cache_status'], 'cold')
        self.assertIn('partial-story overlap', result['limits'])

    def test_rehashed_valid_dtype_change_recomputes(self):
        self.load()
        arrays = {s:encode_stories(self.tokenizer,self.base_stories[s]).astype(np.uint32)
                  for s in data.FILES}
        self.write_real(self.base_stories, arrays=arrays)
        self.assertEqual(self.load()['cache_status'], 'cold')
        self.assertEqual(self.load()['cache_status'], 'hit')

    def test_basic_range_checked_when_cache_populated(self):
        self.load()
        arrays = {s:encode_stories(self.tokenizer,self.base_stories[s]) for s in data.FILES}
        arrays['train'][0] = data.VOCAB_SIZE
        self.write_real(self.base_stories, arrays=arrays)
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.load()

    def test_mutation_during_analysis_is_rejected_and_not_cached(self):
        real_scan = data._story_validation
        def mutate(*args):
            report = real_scan(*args)
            p = self.root/'manifest.json'
            p.write_bytes(p.read_bytes() + b' ')
            return report
        with patch.object(data, '_story_validation', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'mutated'):
                self.load()
        self.assertEqual(len(data._SEMANTIC_CACHE), 0)

    def test_source_hash_changes_cannot_hit(self):
        self.load()
        actual = data.sha256_file
        def altered(path):
            return 'f'*64 if path == data.Path(data.__file__) else actual(path)
        with patch.object(data, 'sha256_file', side_effect=altered):
            self.assertEqual(self.load()['cache_status'], 'cold')

    def test_lru_bounded_and_raw_manifest_bytes_bind_key(self):
        manifest_path = self.root/'manifest.json'
        for i in range(5):
            manifest_path.write_bytes(manifest_path.read_bytes() + b' ')
            self.assertEqual(self.load()['cache_status'], 'cold')
        self.assertEqual(len(data._SEMANTIC_CACHE), 4)

    def test_disk_cache_claim_ignored(self):
        (self.root/'validation-cache.json').write_text('{"status":"passed"}')
        self.assertEqual(self.load()['cache_status'], 'cold')

if __name__ == '__main__':
    unittest.main()
