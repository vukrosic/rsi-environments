"""Adversarial CPU tests for whole-story and tokenizer data semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data
from data import FILES, MANIFEST_VERSION, VOCAB_SIZE, load_data

EOT = "<|endoftext|>"


def fixture_tokenizer() -> Tokenizer:
    """Construct the prepare.py 4096-entry ByteLevel BPE vocabulary cheaply."""
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    assert len(alphabet) == 256
    vocab = {EOT: 0}
    vocab.update({piece: i + 1 for i, piece in enumerate(sorted(alphabet))})
    vocab.update({f"<unused-{i}>": i for i in range(257, VOCAB_SIZE)})
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([EOT])
    assert tokenizer.token_to_id(EOT) == 0
    assert tokenizer.get_vocab_size() == VOCAB_SIZE
    return tokenizer


def encode_stories(tokenizer: Tokenizer, stories: list[str]) -> np.ndarray:
    out: list[int] = []
    for story in stories:
        out.extend(tokenizer.encode(story).ids)
        out.append(0)
    return np.asarray(out, dtype=np.uint16)


class DataSemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tokenizer = fixture_tokenizer()
        self.base_stories = {
            "train": ["train story alpha " * 4, "train story beta " * 4, "train story gamma " * 4],
            "dev": ["dev story delta " * 4, "dev story epsilon " * 4, "dev story zeta " * 4],
            "final": ["final story eta " * 4, "final story theta " * 4, "final story iota " * 4],
        }
        self.write_real(self.base_stories)

    def tearDown(self):
        self.temp.cleanup()

    def write_real(self, stories: dict[str, list[str]], *, arrays=None, story_counts=None,
                   tokenizer_json: str | None = None, kind: str = "real"):
        self.root.mkdir(parents=True, exist_ok=True)
        tok_json = tokenizer_json if tokenizer_json is not None else self.tokenizer.to_str()
        (self.root / "tok.json").write_text(tok_json, encoding="utf-8")
        arrays = arrays or {name: encode_stories(self.tokenizer, stories[name]) for name in FILES}
        hashes = {}
        counts = {}
        for split, filename in FILES.items():
            np.save(self.root / filename, arrays[split], allow_pickle=False)
            hashes[filename] = hashlib.sha256((self.root / filename).read_bytes()).hexdigest()
            counts[split] = int(len(arrays[split]))
        hashes["tok.json"] = hashlib.sha256((self.root / "tok.json").read_bytes()).hexdigest()
        manifest = {
            "version": MANIFEST_VERSION,
            "kind": kind,
            "vocab_size": VOCAB_SIZE,
            "counts": counts,
            "story_counts": story_counts or {name: len(stories[name]) for name in FILES},
            "sha256": hashes,
            "tokenizer_file": "tok.json" if kind == "real" else None,
        }
        if kind != "real":
            manifest["tokenizer_file"] = None
            manifest["sha256"].pop("tok.json")
        (self.root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return manifest

    def mutate_arrays(self, **changes):
        arrays = {name: np.load(self.root / filename).copy() for name, filename in FILES.items()}
        arrays.update(changes)
        stories = {name: self.base_stories[name] for name in FILES}
        self.write_real(stories, arrays=arrays)

    def test_valid_disjoint_real_fixture_is_accepted(self):
        arrays = load_data(self.root, profile="smoke")
        self.assertEqual([len(array) for array in arrays[:3]], [len(encode_stories(self.tokenizer, self.base_stories[s])) for s in FILES])
        self.assertEqual(arrays[3]["kind"], "real")

    def test_overlap_rejected_even_when_streams_and_manifest_hashes_are_distinct(self):
        overlap = {name: list(values) for name, values in self.base_stories.items()}
        overlap["dev"][0] = overlap["train"][0]
        self.write_real(overlap)
        with self.assertRaisesRegex(ValueError, "overlap|duplicate|story"):
            load_data(self.root, profile="smoke")

    def test_decoded_normalized_overlap_rejected_with_different_tokens(self):
        overlap = {name: list(values) for name, values in self.base_stories.items()}
        overlap['dev'][0] = overlap['train'][0].replace(' ', '  ')
        self.assertFalse(np.array_equal(encode_stories(self.tokenizer,[overlap['train'][0]]), encode_stories(self.tokenizer,[overlap['dev'][0]])))
        self.write_real(overlap)
        with self.assertRaisesRegex(ValueError, 'decoded normalized text'):
            load_data(self.root, profile='smoke')

    def test_missing_eot_rejected(self):
        arrays = {name: encode_stories(self.tokenizer, self.base_stories[name]) for name in FILES}
        arrays["train"] = arrays["train"][:-1]
        self.mutate_arrays(**arrays)
        with self.assertRaises(ValueError):
            load_data(self.root, profile="smoke")

    def test_non_bytelevel_tokenizer_rejected(self):
        config = json.loads(self.tokenizer.to_str())
        config["pre_tokenizer"] = {"type": "Whitespace"}
        self.write_real(self.base_stories, tokenizer_json=json.dumps(config))
        with self.assertRaisesRegex(ValueError, "ByteLevel|tokenizer|byte"):
            load_data(self.root, profile="smoke")

    def test_trailing_incomplete_story_rejected(self):
        arrays = {name: encode_stories(self.tokenizer, self.base_stories[name]) for name in FILES}
        arrays["dev"] = np.concatenate([arrays["dev"], np.asarray([12, 13], dtype=np.uint16)])
        self.mutate_arrays(**arrays)
        with self.assertRaises(ValueError):
            load_data(self.root, profile="smoke")

    def test_empty_eot_only_story_rejected(self):
        arrays = {name: encode_stories(self.tokenizer, self.base_stories[name]) for name in FILES}
        arrays["final"] = np.concatenate([arrays["final"], np.asarray([0], dtype=np.uint16)])
        self.mutate_arrays(**arrays)
        with self.assertRaisesRegex(ValueError, "empty|story|EOT|token"):
            load_data(self.root, profile="smoke")

    def test_tampered_story_counts_rejected(self):
        manifest = self.write_real(self.base_stories)
        manifest["story_counts"]["train"] += 1
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "story|count"):
            load_data(self.root, profile="smoke")

    def test_malformed_existing_tokenizer_rejected(self):
        self.write_real(self.base_stories, tokenizer_json="{ not valid tokenizer json }")
        with self.assertRaisesRegex(ValueError, "tokenizer|inspect|JSON"):
            load_data(self.root, profile="smoke")

    def test_standard_profile_uses_patchable_minimum(self):
        with patch.object(data, "STANDARD_MIN_TOKENS", 68):
            arrays = load_data(self.root, profile="standard")
        self.assertEqual(arrays[3]["kind"], "real")

    def test_same_token_stream_with_different_integer_dtypes_is_rejected(self):
        stream = encode_stories(self.tokenizer, self.base_stories["train"])
        arrays = {
            "train": stream.astype(np.uint16),
            "dev": stream.astype(np.uint32),
            "final": stream.astype(np.uint64),
        }
        self.write_real(self.base_stories, arrays=arrays)
        with patch.object(data, "STANDARD_MIN_TOKENS", 68):
            with self.assertRaisesRegex(ValueError, "same|distinct|duplicate|overlap|story"):
                load_data(self.root, profile="standard")

    def test_synthetic_smoke_bypass_is_explicit(self):
        arrays = {name: np.arange(68, dtype=np.uint16) + i for i, name in enumerate(FILES)}
        manifest = self.write_real(self.base_stories, arrays=arrays, kind="synthetic", story_counts={})
        self.assertEqual(manifest["kind"], "synthetic")
        result = load_data(self.root, profile="smoke")
        self.assertEqual(result[3]["kind"], "synthetic")
        with self.assertRaisesRegex(ValueError, "synthetic"):
            load_data(self.root, profile="standard")


if __name__ == "__main__":
    unittest.main()
