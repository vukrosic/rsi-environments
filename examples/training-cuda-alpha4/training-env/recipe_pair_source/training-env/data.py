"""Verified loading for the reproducible RSI training environment."""

from __future__ import annotations

import copy
from collections import OrderedDict
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_VERSION = "training-data-v1"
VOCAB_SIZE = 4096
STANDARD_MIN_TOKENS = 60 * 32 * 257
FILES = {"train": "train.npy", "dev": "dev.npy", "final": "final.npy"}
EOT = "<|endoftext|>"
VALIDATION_VERSION = "training-semantic-v1"
SCAN_TOKENS = 65536
MAX_STORY_TOKENS = 1_000_000

# Evaluator-owned process memory only; never deserialize candidate cache claims.
_SEMANTIC_CACHE: OrderedDict[tuple, dict] = OrderedDict()
SEMANTIC_CACHE_MAX = 4

def _file_snapshot(root: Path, names: list[str]) -> tuple:
    try:
        return tuple((name, sha256_file(root / name)) for name in names)
    except OSError as exc:
        raise ValueError(f'missing or unreadable data asset during snapshot: {exc}') from exc


def _real_tokenizer(tokenizer: Any) -> int:
    """Accept the deterministic byte-level BPE convention used by prepare.py.

    This checks serialized semantics, not who fitted the tokenizer or on which
    source. A self-consistent manifest cannot authenticate those claims.
    """
    from tokenizers import pre_tokenizers

    spec = json.loads(tokenizer.to_str())
    model = spec.get("model", {})
    pre = spec.get("pre_tokenizer", {})
    decoder = spec.get("decoder", {})
    if (model.get("type") != "BPE" or model.get("dropout") is not None
            or model.get("unk_token") is not None
            or model.get("continuing_subword_prefix") not in (None, "")
            or model.get("end_of_word_suffix") not in (None, "")
            or any(model.get(k, False) for k in ("fuse_unk", "byte_fallback", "ignore_merges"))
            or pre.get("type") != "ByteLevel" or pre.get("add_prefix_space") is not False
            or pre.get("use_regex", True) is not True
            or decoder.get("type") != "ByteLevel"
            or any(spec.get(k) is not None for k in
                   ("normalizer", "post_processor", "truncation", "padding"))):
        raise ValueError("real data requires preparer-compatible deterministic ByteLevel/BPE tokenizer semantics")
    vocab = tokenizer.get_vocab()
    if set(vocab.values()) != set(range(VOCAB_SIZE)) or not set(pre_tokenizers.ByteLevel.alphabet()) <= vocab.keys():
        raise ValueError("real ByteLevel/BPE vocabulary must have contiguous IDs and the full byte alphabet")
    eot_id = tokenizer.token_to_id(EOT)
    added = spec.get("added_tokens", [])
    if (eot_id is None or len(added) != 1 or added[0].get("content") != EOT
            or added[0].get("id") != eot_id or added[0].get("special") is not True
            or any(added[0].get(k, False) for k in ("single_word", "lstrip", "rstrip", "normalized"))):
        raise ValueError("real tokenizer requires one unmodified special <|endoftext|> token")
    return eot_id


def _story_validation(arrays: list[np.ndarray], tokenizer: Any, manifest: dict) -> dict:
    """Scan mmap slices; retain only compact story digests, never whole streams.

    Per-story decoding is capped to bound Python token-list allocation. Exact
    decoded whitespace-normalized identity catches alternate BPE segmentations;
    paraphrases, partial overlaps and source authenticity are outside this check.
    """
    eot_id = _real_tokenizer(tokenizer)
    declared = manifest.get("story_counts")
    if declared is not None and not isinstance(declared, dict):
        raise ValueError("story_counts must be a mapping when provided")
    counts, identities = {}, []
    for split, array in zip(FILES, arrays):
        count = sum(int(np.count_nonzero(array[i:i + SCAN_TOKENS] == eot_id))
                    for i in range(0, len(array), SCAN_TOKENS))
        if not count or int(array[-1]) != eot_id:
            raise ValueError(f"{split} must end at an EOT story boundary; incomplete trailing story")
        if declared is not None and (type(declared.get(split)) is not int or declared[split] != count):
            raise ValueError(f"{split} story_counts mismatch: found {count} EOT-terminated stories")
        token_hashes = np.empty(count, dtype="V32")
        text_hashes = np.empty(count, dtype="V32")
        start, index = 0, 0
        for offset in range(0, len(array), SCAN_TOKENS):
            for relative in np.flatnonzero(array[offset:offset + SCAN_TOKENS] == eot_id):
                end = offset + int(relative)
                if end == start:
                    raise ValueError(f"{split} contains an empty EOT-delimited story")
                if end - start > MAX_STORY_TOKENS:
                    raise ValueError(f"{split} story exceeds validation limit of {MAX_STORY_TOKENS} tokens")
                story = np.asarray(array[start:end], dtype="<u2")
                token_hashes[index] = hashlib.sha256(story.tobytes()).digest()
                text = " ".join(tokenizer.decode(story.tolist(), skip_special_tokens=False).split())
                if not text:
                    raise ValueError(f"{split} contains an empty decoded story")
                text_hashes[index] = hashlib.sha256(text.encode("utf-8")).digest()
                index += 1
                start = end + 1
        counts[split] = count
        identities.append((np.unique(token_hashes), np.unique(text_hashes)))
    names = list(FILES)
    for i in range(3):
        for j in range(i + 1, 3):
            for basis, a, b in zip(("canonical token", "decoded normalized text"), identities[i], identities[j]):
                if np.intersect1d(a, b, assume_unique=True).size:
                    raise ValueError(f"{names[i]}/{names[j]} contain duplicate whole stories ({basis} identity)")
    return {"status": "passed", "eot_id": eot_id, "story_counts": counts,
            "hash_basis": ["SHA-256 of EOT-excluded little-endian uint16 token IDs",
                           "SHA-256 of UTF-8 decoded story with Python split/join whitespace normalization"],
            "scan_chunk_tokens": SCAN_TOKENS, "max_story_tokens": MAX_STORY_TOKENS,
            "digest_bytes_upper_bound": sum(counts.values()) * 64,
            "limits": "Does not authenticate source origins or tokenizer fitting provenance; does not detect paraphrases or partial-story overlap."}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_array(path: Path, expected_sha: str, expected_count: int) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"missing data file: {path.name}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except Exception as exc:
        raise ValueError(f"cannot load {path.name}: {exc}") from exc
    if array.ndim != 1:
        raise ValueError(f"{path.name} must be rank 1, got shape {array.shape}")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{path.name} must contain integral token IDs, got {array.dtype}")
    if len(array) != expected_count:
        raise ValueError(f"{path.name} count mismatch: manifest says {expected_count}, found {len(array)}")
    if array.size and (int(array.min()) < 0 or int(array.max()) >= VOCAB_SIZE):
        raise ValueError(f"{path.name} contains token IDs outside [0, {VOCAB_SIZE})")
    return array


def load_data(root: str | Path, profile: str = "standard", *, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and verify train/dev/final token arrays and their manifest.

    ``standard`` is the measured-training profile and requires 60*32*257 tokens
    in every split. ``smoke`` is for explicit local smoke checks and accepts
    synthetic manifests with at least 68 tokens per split.
    """
    validation_started = time.perf_counter()
    if profile not in {"standard", "smoke"}:
        raise ValueError("profile must be 'standard' or 'smoke'")
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except Exception as exc:
        raise ValueError(f"invalid manifest JSON: {exc}") from exc
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"manifest version must be {MANIFEST_VERSION!r}")
    kind = manifest.get("kind")
    if kind not in {"real", "synthetic"}:
        raise ValueError("manifest kind must be 'real' or 'synthetic'")
    if kind == "synthetic" and profile == "standard":
        raise ValueError("standard profile rejects synthetic smoke data")
    if manifest.get("vocab_size") != VOCAB_SIZE:
        raise ValueError(f"manifest vocab_size must be {VOCAB_SIZE}")

    hashes = manifest.get("sha256")
    counts = manifest.get("counts")
    if not isinstance(hashes, dict) or not isinstance(counts, dict):
        raise ValueError("manifest must include sha256 and counts mappings")
    snapshot_names = ["manifest.json", *FILES.values()]
    if manifest.get("tokenizer_file") == "tok.json":
        snapshot_names.append("tok.json")
    before = _file_snapshot(root, snapshot_names)
    if dict(before)["manifest.json"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("data files mutated during validation")
    arrays = []
    minimum = STANDARD_MIN_TOKENS if profile == "standard" else 68
    for split in ("train", "dev", "final"):
        filename = FILES[split]
        count = counts.get(split)
        digest = hashes.get(filename)
        if not isinstance(count, int) or isinstance(count, bool) or count < minimum:
            raise ValueError(f"{split} must contain at least {minimum} tokens for profile {profile!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"manifest has no valid SHA-256 for {filename}")
        arrays.append(_read_array(root / filename, digest, count))
    if len({hashes[FILES[split]] for split in ("train", "dev", "final")}) != 3:
        raise ValueError("train, dev, and final must be physically distinct token streams")

    tokenizer_name = manifest.get("tokenizer_file")
    if tokenizer_name is not None:
        if tokenizer_name != "tok.json":
            raise ValueError("tokenizer_file must be 'tok.json' when present")
        tok_hash = hashes.get(tokenizer_name)
        tok_path = root / tokenizer_name
        if not isinstance(tok_hash, str) or not tok_path.is_file() or sha256_file(tok_path) != tok_hash:
            raise ValueError("tokenizer file missing or SHA-256 mismatch")
        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(tok_path))
            actual_vocab_size = tokenizer.get_vocab_size()
        except Exception as exc:
            raise ValueError(f"cannot inspect tokenizer vocabulary: {exc}") from exc
        if actual_vocab_size != VOCAB_SIZE:
            raise ValueError(f"tok.json vocabulary must contain {VOCAB_SIZE} tokens, found {actual_vocab_size}")
    elif kind == "real":
        raise ValueError("real training data requires tok.json and its hash")

    metadata = dict(manifest)
    metadata["root"] = str(root.resolve())
    # Bind the report to raw manifest bytes, every asset byte and validator source.
    # Basic array/tokenizer checks above remain mandatory even on a cache hit.
    source_sha = sha256_file(Path(__file__))
    key = (VALIDATION_VERSION, source_sha, SCAN_TOKENS, MAX_STORY_TOKENS,
           VOCAB_SIZE, before)
    cache_status = "bypassed" if not use_cache else "cold"
    if kind == "real" and use_cache and key in _SEMANTIC_CACHE:
        validation = copy.deepcopy(_SEMANTIC_CACHE[key])
        _SEMANTIC_CACHE.move_to_end(key)
        cache_status = "hit"
    else:
        validation = (_story_validation(arrays, tokenizer, manifest) if kind == "real" else
                      {"status": "skipped-synthetic", "hash_basis": "file SHA-256 only; story semantics intentionally skipped"})
    if _file_snapshot(root, snapshot_names) != before or sha256_file(Path(__file__)) != source_sha:
        raise ValueError("data files or validator source mutated during validation")
    if kind == "real" and use_cache and cache_status == "cold":
        _SEMANTIC_CACHE[key] = copy.deepcopy(validation)
        while len(_SEMANTIC_CACHE) > SEMANTIC_CACHE_MAX:
            _SEMANTIC_CACHE.popitem(last=False)
    validation["cache_status"] = cache_status if kind == "real" else "skipped-synthetic"
    validation["provenance"] = {"validator_source_sha256": source_sha,
                                "file_sha256": dict(before),
                                "cache_scope": "evaluator process memory; max 4 stable semantic reports"}
    validation.update({"version": VALIDATION_VERSION, "elapsed_seconds": time.perf_counter() - validation_started,
                       "tokens_scanned": sum(len(a) for a in arrays), "all_splits": list(FILES)})
    metadata["validation"] = validation
    return arrays[0], arrays[1], arrays[2], metadata
