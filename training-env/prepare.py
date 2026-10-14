"""Prepare verified TinyStories token splits without altering source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

try:
    from .data import FILES, MANIFEST_VERSION, STANDARD_MIN_TOKENS, VOCAB_SIZE, sha256_file
except ImportError:  # Running as ``python prepare.py`` from this directory.
    from data import FILES, MANIFEST_VERSION, STANDARD_MIN_TOKENS, VOCAB_SIZE, sha256_file

EOT = "<|endoftext|>"
EOT_ID_FALLBACK_FOR_SYNTHETIC = 0


def _stories_from_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Match the original preparer's convention: only delimiter-terminated
    # stories are complete; a partial last record is excluded.
    pieces = text.split(EOT)
    return [part.strip() for part in pieces[:-1] if part.strip()]


def _story_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _token_story_hash(tokens: Iterable[int]) -> str:
    values = np.asarray(list(tokens), dtype=np.uint16)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _check_story_overlap(train_hashes: set[str], dev_hashes: set[str], final_hashes: set[str]) -> None:
    overlap = train_hashes & (dev_hashes | final_hashes)
    if overlap:
        raise ValueError(f"held-out story duplicates training data ({len(overlap)} duplicate story hash(es))")
    overlap = dev_hashes & final_hashes
    if overlap:
        raise ValueError(f"dev/final contain duplicate held-out stories ({len(overlap)} duplicate story hash(es))")


def _encode_stories(tokenizer: Tokenizer, stories: list[str], eot_id: int) -> tuple[np.ndarray, list[str]]:
    ids: list[int] = []
    hashes: list[str] = []
    for start in range(0, len(stories), 2000):
        batch = stories[start:start + 2000]
        for story, encoding in zip(batch, tokenizer.encode_batch(batch)):
            ids.extend(encoding.ids)
            ids.append(eot_id)
            hashes.append(_story_hash(story))
    return np.asarray(ids, dtype=np.uint16), hashes


def _split_text_validation(stories: list[str], tokenizer: Tokenizer) -> tuple[list[str], list[str]]:
    if len(stories) < 2:
        raise ValueError("validation source needs at least two complete EOT-terminated stories")
    lengths = [len(tokenizer.encode(story).ids) + 1 for story in stories]
    total = sum(lengths)
    cumulative = 0
    candidates = []
    for i, length in enumerate(lengths[:-1], 1):
        cumulative += length
        candidates.append((abs(total - 2 * cumulative), i))
    cut = min(candidates)[1]
    return stories[:cut], stories[cut:]


def _tokenizer_from_train(stories: list[str]) -> Tokenizer:
    if not stories:
        raise ValueError("training text has no complete EOT-terminated stories")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        stories[::4],
        trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=[EOT],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise ValueError(f"training text produced vocabulary size {tokenizer.get_vocab_size()}, expected {VOCAB_SIZE}")
    return tokenizer


def _segment_legacy(array: np.ndarray, eot_id: int, label: str) -> list[np.ndarray]:
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"legacy {label} must be a rank-1 integral token array")
    if array.size == 0 or int(array.min()) < 0 or int(array.max()) >= VOCAB_SIZE:
        raise ValueError(f"legacy {label} has empty or out-of-range token IDs")
    boundaries = np.flatnonzero(array == eot_id)
    if not len(boundaries) or boundaries[-1] != len(array) - 1:
        raise ValueError(f"legacy {label} must end at an EOT story boundary")
    stories: list[np.ndarray] = []
    start = 0
    for end in boundaries:
        stories.append(np.asarray(array[start:end + 1], dtype=np.uint16))
        start = int(end) + 1
    return stories


def _split_legacy_validation(stories: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(stories) < 2:
        raise ValueError("legacy valid.npy needs at least two EOT-delimited stories")
    lengths = np.asarray([len(story) for story in stories], dtype=np.int64)
    cumulative = np.cumsum(lengths)[:-1]
    cut = int(np.argmin(np.abs(lengths.sum() - 2 * cumulative))) + 1
    return stories[:cut], stories[cut:]


def _join(stories: list[np.ndarray]) -> np.ndarray:
    if not stories:
        return np.empty(0, dtype=np.uint16)
    return np.concatenate(stories).astype(np.uint16, copy=False)


def _write_output(output: Path, arrays: dict[str, np.ndarray], tokenizer_json: str | None,
                  provenance: dict, kind: str = "real", story_counts: dict | None = None) -> dict:
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError('output already exists; preserve it and choose a new directory') from exc
    for split, filename in FILES.items():
        np.save(output / filename, arrays[split], allow_pickle=False)
    hashes = {filename: sha256_file(output / filename) for filename in FILES.values()}
    tokenizer_file = None
    if tokenizer_json is not None:
        tokenizer_file = "tok.json"
        (output / tokenizer_file).write_text(tokenizer_json, encoding="utf-8")
        hashes[tokenizer_file] = sha256_file(output / tokenizer_file)
    manifest = {
        "version": MANIFEST_VERSION,
        "kind": kind,
        "vocab_size": VOCAB_SIZE,
        "counts": {name: int(len(arrays[name])) for name in FILES},
        "story_counts": story_counts or {},
        "sha256": hashes,
        "tokenizer_file": tokenizer_file,
        "source": provenance,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def prepare_from_existing(source: Path, output: Path) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output == source:
        raise ValueError("--output must be a different directory so existing source files remain untouched")
    for name in ("train.npy", "valid.npy", "tok.json"):
        if not (source / name).is_file():
            raise ValueError(f"existing-data source is missing {name}")
    tokenizer = Tokenizer.from_file(str(source / "tok.json"))
    eot_id = tokenizer.token_to_id(EOT)
    if eot_id is None:
        raise ValueError(f"legacy tokenizer has no {EOT} token")
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise ValueError(f"legacy tokenizer vocabulary must be {VOCAB_SIZE}")
    train_array = np.load(source / "train.npy", allow_pickle=False, mmap_mode="r")
    valid_array = np.load(source / "valid.npy", allow_pickle=False, mmap_mode="r")
    train_stories = _segment_legacy(train_array, eot_id, "train.npy")
    valid_stories = _segment_legacy(valid_array, eot_id, "valid.npy")
    dev_stories, final_stories = _split_legacy_validation(valid_stories)
    train_hashes = {_token_story_hash(story) for story in train_stories}
    dev_hashes = {_token_story_hash(story) for story in dev_stories}
    final_hashes = {_token_story_hash(story) for story in final_stories}
    _check_story_overlap(train_hashes, dev_hashes, final_hashes)
    arrays = {"train": _join(train_stories), "dev": _join(dev_stories), "final": _join(final_stories)}
    for split, array in arrays.items():
        if len(array) < STANDARD_MIN_TOKENS:
            raise ValueError(f"{split} has {len(array)} tokens; needs at least {STANDARD_MIN_TOKENS}")
    if len({hashlib.sha256(a.tobytes()).hexdigest() for a in arrays.values()}) != 3:
        raise ValueError("train, dev, and final arrays must be distinct")
    provenance = {
        "method": "split-existing-token-stream-at-eot-story-boundaries",
        "source_directory": str(source),
        "source_sha256": {name: sha256_file(source / name) for name in ("train.npy", "valid.npy", "tok.json")},
        "validation_split": "whole stories, nearest token-count midpoint",
    }
    return _write_output(output, arrays, (source / "tok.json").read_text(encoding="utf-8"), provenance,
                         story_counts={"train": len(train_stories), "dev": len(dev_stories), "final": len(final_stories)})


def prepare_from_text(train_path: Path, valid_path: Path, output: Path) -> dict:
    train_path = train_path.expanduser().resolve()
    valid_path = valid_path.expanduser().resolve()
    output = output.expanduser().resolve()
    train_stories = _stories_from_text(train_path)
    valid_stories = _stories_from_text(valid_path)
    tokenizer = _tokenizer_from_train(train_stories)
    dev_text, final_text = _split_text_validation(valid_stories, tokenizer)
    arrays, train_hashes = {}, None
    arrays["train"], train_hashes_list = _encode_stories(tokenizer, train_stories, tokenizer.token_to_id(EOT))
    arrays["dev"], dev_hashes_list = _encode_stories(tokenizer, dev_text, tokenizer.token_to_id(EOT))
    arrays["final"], final_hashes_list = _encode_stories(tokenizer, final_text, tokenizer.token_to_id(EOT))
    _check_story_overlap(set(train_hashes_list), set(dev_hashes_list), set(final_hashes_list))
    for split, array in arrays.items():
        if len(array) < STANDARD_MIN_TOKENS:
            raise ValueError(f"{split} has {len(array)} tokens; needs at least {STANDARD_MIN_TOKENS}")
    if len({hashlib.sha256(a.tobytes()).hexdigest() for a in arrays.values()}) != 3:
        raise ValueError("train, dev, and final arrays must be distinct")
    provenance = {
        "method": "fit-4096-bytelevel-bpe-on-train-only; split-validation-by-whole-stories",
        "train_text": {"path": str(train_path), "sha256": sha256_file(train_path)},
        "validation_text": {"path": str(valid_path), "sha256": sha256_file(valid_path)},
        "validation_split": "whole stories, nearest token-count midpoint",
    }
    return _write_output(output, arrays, tokenizer.to_str(), provenance,
                         story_counts={"train": len(train_stories), "dev": len(dev_text), "final": len(final_text)})


def prepare_synthetic(output: Path, tokens_per_split: int = 1024) -> dict:
    if tokens_per_split < 1:
        raise ValueError("--synthetic-tokens must be positive")
    output = output.expanduser().resolve()
    # Deterministic, structured smoke sequences. The explicit synthetic kind
    # makes these impossible to load with the measured standard profile.
    arrays = {}
    for split_index, split in enumerate(("train", "dev", "final")):
        positions = np.arange(tokens_per_split, dtype=np.uint32)
        arrays[split] = ((positions * 17 + split_index * 997 + (positions // 7) * 3) % VOCAB_SIZE).astype(np.uint16)
    return _write_output(output, arrays, None,
                         {"method": "deterministic-structured-smoke-tokens", "tokens_per_split": tokens_per_split,
                          "warning": "synthetic smoke data; not suitable for measured training"},
                         kind="synthetic")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-existing", type=Path, help="directory containing train.npy, valid.npy, tok.json")
    parser.add_argument("--train-text", type=Path)
    parser.add_argument("--valid-text", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="write explicitly labeled smoke data")
    parser.add_argument("--synthetic-tokens", type=int, default=1024)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    modes = int(args.from_existing is not None) + int(args.train_text is not None or args.valid_text is not None) + int(args.synthetic)
    if modes != 1:
        parser.error("choose exactly one of --from-existing, --train-text with --valid-text, or --synthetic")
    if (args.train_text is None) != (args.valid_text is None):
        parser.error("--train-text and --valid-text must be supplied together")
    if args.from_existing is not None:
        manifest = prepare_from_existing(args.from_existing, args.output)
    elif args.train_text is not None:
        manifest = prepare_from_text(args.train_text, args.valid_text, args.output)
    else:
        manifest = prepare_synthetic(args.output, args.synthetic_tokens)
    print(json.dumps({"output": str(args.output.expanduser().resolve()), "kind": manifest["kind"],
                      "counts": manifest["counts"], "manifest": str(args.output.expanduser().resolve() / "manifest.json")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
