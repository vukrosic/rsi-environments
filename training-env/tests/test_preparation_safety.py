"""Regression tests for preparation output ownership and source preservation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import prepare


class FakeTokenizer:
    def token_to_id(self, token):
        return 0

    def get_vocab_size(self):
        return prepare.VOCAB_SIZE


class PreparationSafetyTests(unittest.TestCase):
    def test_synthetic_preparation_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing-output"
            output.mkdir()
            sentinel = output / "previous-result.json"
            sentinel.write_text("keep this previous result\n")

            with self.assertRaisesRegex(ValueError, "exist|overwrite|preserve"):
                prepare.prepare_synthetic(output, tokens_per_split=8)

            self.assertEqual(sentinel.read_text(), "keep this previous result\n")
            self.assertEqual(sorted(path.name for path in output.iterdir()), [sentinel.name])

    def test_existing_data_refuses_occupied_output_and_preserves_source_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            source.mkdir()
            output.mkdir()
            np.save(source / "train.npy", np.asarray([1, 0, 2, 0], dtype=np.uint16), allow_pickle=False)
            np.save(source / "valid.npy", np.asarray([3, 0, 4, 0], dtype=np.uint16), allow_pickle=False)
            (source / "tok.json").write_text("fixture tokenizer marker\n")
            source_before = {path.name: path.read_bytes() for path in source.iterdir()}
            sentinel = output / "previous-result.json"
            sentinel.write_text("keep this previous result\n")

            with patch.object(prepare.Tokenizer, "from_file", return_value=FakeTokenizer()), \
                 patch.object(prepare, "STANDARD_MIN_TOKENS", 1):
                with self.assertRaisesRegex(ValueError, "exist|overwrite|preserve"):
                    prepare.prepare_from_existing(source, output)

            self.assertEqual({path.name: path.read_bytes() for path in source.iterdir()}, source_before)
            self.assertEqual(sentinel.read_text(), "keep this previous result\n")
            self.assertEqual(sorted(path.name for path in output.iterdir()), [sentinel.name])

    def test_existing_data_same_source_guard_does_not_touch_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            for name, contents in (("train.npy", b"history-train"),
                                   ("valid.npy", b"history-valid"),
                                   ("tok.json", b"history-tokenizer")):
                (source / name).write_bytes(contents)
            before = {path.name: path.read_bytes() for path in source.iterdir()}

            with self.assertRaisesRegex(ValueError, "different directory"):
                prepare.prepare_from_existing(source, source)

            self.assertEqual({path.name: path.read_bytes() for path in source.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
