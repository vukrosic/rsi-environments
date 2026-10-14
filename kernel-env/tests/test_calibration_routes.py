"""Routing fixtures are synthetic JSON; no CPU/GPU timing is performed."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest
from kernel_env import calibration_routes as routes, difficulty


def receipt(task="softmax", case_id="legacy"):
    return {"status": "ok", "task": task, "system": {"device": "cpu", "fixture": True},
            "difficulty": {"id": "legacy"} if case_id == "legacy" else difficulty.choose(task, "standard", 0, case_id),
            "baseline_ms_by_variant": {str(v): 1.0 for v in (0, 4, 5, 6, 10, 11)}}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def index(task="softmax", paths=None):
    return {"schema_version": 1, "kind": routes.INDEX_KIND, "task": task,
            "profile": "standard", "receipts": paths or {"legacy": "raw.json"}}


@pytest.mark.parametrize("task", difficulty.TRAIN_TASKS)
def test_seed_selected_nine_families_route_through_one_path(task, tmp_path):
    paths = {d["id"]: write(tmp_path / f'{d["id"]}.json', receipt(task, d["id"]))
             for d in difficulty.catalog(task, "standard")}
    source = routes.write_index(tmp_path / "index.json", task, paths)
    selected = set()
    for seed in range(100):
        case_id = difficulty.choose(task, "standard", seed)["id"]
        selected.add(case_id)
        result = routes.load(source, task, case_id)
        assert result["receipt"]["difficulty"]["id"] == case_id
        assert result["path"] == str(paths[case_id].resolve())
        assert result["index_path"] == str(source)
    assert selected == set(paths)


@pytest.mark.parametrize("task", ("softmax", "attention", "newton_schulz"))
def test_raw_compatibility_including_heldout_legacy(task, tmp_path):
    raw = write(tmp_path / "raw.json", receipt(task))
    data = raw.read_bytes()
    assert routes.load(raw, task) == {"receipt": receipt(task), "path": str(raw.resolve()),
                                     "sha256": hashlib.sha256(data).hexdigest(),
                                     "snapshots": {"calibration-receipt.json": data}}


def test_raw_route_defers_semantic_validation_to_parent(tmp_path):
    data = {"status": "blocked", "task": "attention", "baseline_ms_by_variant": {"0": -1}}
    raw = write(tmp_path / "raw.json", data)
    assert routes.load(raw, "softmax", "wide/offset")["receipt"] == data


def test_partial_relative_index_exact_snapshots_and_missing_selection(tmp_path):
    case_id = "odd/offset"
    raw = write(tmp_path / "receipts" / "raw.json", receipt(case_id=case_id))
    source = write(tmp_path / "index.json", index(paths={case_id: "receipts/raw.json"}))
    result = routes.load(source, "softmax", case_id)
    assert result["path"] == str(raw.resolve())
    assert result["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert result["index_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["snapshots"] == {"calibration-receipt.json": raw.read_bytes(), "calibration-index.json": source.read_bytes()}
    before = result["snapshots"]["calibration-receipt.json"]
    raw.write_text('{"replaced":true}')
    assert result["receipt"] == json.loads(before)
    assert result["sha256"] == hashlib.sha256(before).hexdigest()
    with pytest.raises(ValueError, match="no receipt"):
        routes.load(source, "softmax", "wide/ordinary")


def test_absolute_operator_path_and_legacy_mapping(tmp_path):
    raw = write(tmp_path / "outside" / "raw.json", receipt())
    source = write(tmp_path / "index.json", index(paths={"legacy": str(raw)}))
    assert routes.load(source, "softmax")["path"] == str(raw)
    assert routes.load(source, "softmax", "legacy")["receipt"] == receipt()


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"schema_version": True}, {"schema_version": 1.0}, {"kind": "other"},
    {"task": "rmsnorm"}, {"task": "unknown"}, {"profile": "smoke"}, {"profile": None}, {"extra": True},
    {"receipts": []}, {"receipts": {"unknown": "raw.json"}}, {"receipts": {"legacy": None}},
    {"receipts": {"legacy": 1}}, {"receipts": {"legacy": ""}}, {"receipts": {"legacy": "   "}},
])
def test_index_schema_and_mapping_validation(change, tmp_path):
    document = index()
    document.update(change)
    source = write(tmp_path / "index.json", document)
    with pytest.raises(ValueError):
        routes.load(source, "softmax")


@pytest.mark.parametrize("task", ("attention", "newton_schulz", "unknown"))
def test_indices_training_only(task, tmp_path):
    source = write(tmp_path / "index.json", index(task))
    with pytest.raises(ValueError, match="training tasks only"):
        routes.load(source, task)
    with pytest.raises(ValueError, match="training tasks only"):
        routes.write_index(tmp_path / "new.json", task, {})


@pytest.mark.parametrize("case_id", ("unknown", "", True, 4, ["legacy"]))
def test_invalid_selected_id(case_id, tmp_path):
    source = write(tmp_path / "index.json", index())
    with pytest.raises(ValueError, match="unknown calibration difficulty"):
        routes.load(source, "softmax", case_id)


@pytest.mark.parametrize("content", [b"{", b"[]", b"null", b"\xff", b'{"status":"ok","status":"blocked"}',
                                     b'{"nested":{"x":1,"x":2}}', b'{"n":NaN}', b'{"n":Infinity}',
                                     b'{"n":-Infinity}', b'{"n":1e9999}'])
@pytest.mark.parametrize("indexed", (False, True))
def test_malformed_and_nonfinite_on_both_routes(content, indexed, tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_bytes(content)
    source = write(tmp_path / "index.json", index()) if indexed else raw
    with pytest.raises(ValueError):
        routes.load(source, "softmax")


def test_duplicate_index_mapping_keys(tmp_path):
    source = tmp_path / "index.json"
    source.write_text('{"schema_version":1,"kind":"kernel_calibration_index","task":"softmax",'
                      '"profile":"standard","receipts":{"legacy":"a","legacy":"b"}}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        routes.load(source, "softmax")


@pytest.mark.parametrize("indexed", (False, True))
def test_size_bound_includes_exact_limit(indexed, tmp_path):
    raw = tmp_path / "raw.json"
    content = b'{"status":"fixture"}'
    raw.write_bytes(content + b" " * (routes.MAX_BYTES - len(content)))
    assert routes.load(raw, "softmax")["receipt"] == {"status": "fixture"}
    raw.write_bytes(raw.read_bytes() + b" ")
    source = write(tmp_path / "index.json", index()) if indexed else raw
    with pytest.raises(ValueError, match="exceeds"):
        routes.load(source, "softmax")


@pytest.mark.parametrize("kind", ("nested", "cycle", "two_file_cycle"))
def test_nested_and_cycles_rejected_without_recursing(kind, tmp_path):
    first = write(tmp_path / "first.json", index(paths={"legacy": "first.json" if kind == "cycle" else "second.json"}))
    write(tmp_path / "second.json", index(paths={"legacy": "first.json" if kind == "two_file_cycle" else "raw.json"}))
    with pytest.raises(ValueError, match="cycle|nested"):
        routes.load(first, "softmax")


@pytest.mark.parametrize("change", [{"status": "blocked"}, {"task": "rmsnorm"}, {"difficulty": {"id": "wide/offset"}},
                                     {"profile": "smoke"}, {"system": {}}, {"system": None}, {"system": {"device": "other"}}])
def test_builder_basic_identity_before_write(change, tmp_path):
    data = receipt()
    data.update(change)
    raw = write(tmp_path / "raw.json", data)
    destination = tmp_path / "new.json"
    with pytest.raises(ValueError):
        routes.write_index(destination, "softmax", {"legacy": raw})
    assert not destination.exists()


@pytest.mark.parametrize("value", [None, {}, {"0": 1.0}, {str(v): True for v in (0, 4, 5, 6, 10, 11)},
                                    {str(v): -1 for v in (0, 4, 5, 6, 10, 11)},
                                    {str(v): "1" for v in (0, 4, 5, 6, 10, 11)}])
def test_builder_rejects_missing_or_invalid_timings(value, tmp_path):
    data = receipt()
    data["baseline_ms_by_variant"] = value
    raw = write(tmp_path / "raw.json", data)
    with pytest.raises(ValueError, match="six declared|finite positive"):
        routes.write_index(tmp_path / "index.json", "softmax", {"legacy": raw})


def test_builder_rejects_tampered_descriptor(tmp_path):
    data = receipt(case_id="nominal/ordinary")
    data["difficulty"]["profile"] = "smoke"
    raw = write(tmp_path / "raw.json", data)
    with pytest.raises(ValueError, match="canonical"):
        routes.write_index(tmp_path / "index.json", "softmax", {"nominal/ordinary": raw})


@pytest.mark.parametrize("mapping", ({}, {"unknown": "raw.json"}, {"legacy": None}, {"legacy": ""}, []))
def test_builder_bad_mapping(mapping, tmp_path):
    with pytest.raises(ValueError):
        routes.write_index(tmp_path / "index.json", "softmax", mapping)


def test_builder_preserves_raw_existing_output_and_dangling_symlinks(tmp_path):
    raw = write(tmp_path / "raw.json", receipt())
    original = raw.read_bytes()
    with pytest.raises(FileExistsError):
        routes.write_index(raw, "softmax", {"legacy": raw})
    assert raw.read_bytes() == original
    destination = routes.write_index(tmp_path / "index.json", "softmax", {"legacy": raw})
    original_index = destination.read_bytes()
    with pytest.raises(FileExistsError):
        routes.write_index(destination, "softmax", {"legacy": raw})
    assert destination.read_bytes() == original_index
    dangling = tmp_path / "dangling.json"
    missing = tmp_path / "missing.json"
    dangling.symlink_to(missing)
    with pytest.raises(FileExistsError):
        routes.write_index(dangling, "softmax", {"legacy": raw})
    assert dangling.is_symlink() and not missing.exists()


def test_builder_rejects_index_as_receipt(tmp_path):
    source = write(tmp_path / "index.json", index())
    with pytest.raises(ValueError, match="nested"):
        routes.write_index(tmp_path / "new.json", "softmax", {"legacy": source})


def test_load_opens_each_selected_file_once(tmp_path, monkeypatch):
    raw = write(tmp_path / "raw.json", receipt())
    source = write(tmp_path / "index.json", index())
    calls = []
    original = Path.open
    def counted(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", counted)
    routes.load(source, "softmax")
    assert calls == [source, raw]


@pytest.mark.parametrize("value", (None, "", "   ", 4, b"raw.json"))
def test_invalid_input_paths(value):
    with pytest.raises(ValueError, match="nonempty path"):
        routes.load(value, "softmax")


def test_oversize_index_itself_is_rejected_before_selection(tmp_path):
    source = write(tmp_path / "index.json", index())
    data = source.read_bytes()
    source.write_bytes(data + b" " * (routes.MAX_BYTES + 1 - len(data)))
    with pytest.raises(ValueError, match="exceeds"):
        routes.load(source, "softmax")


@pytest.mark.parametrize("field", ("schema_version", "kind", "task", "profile"))
def test_missing_index_fields_are_rejected(field, tmp_path):
    data = index()
    del data[field]
    source = write(tmp_path / "index.json", data)
    with pytest.raises(ValueError, match="schema-1 fields"):
        routes.load(source, "softmax")


@pytest.mark.parametrize("value", (0, float("nan"), float("inf"), 10**1000))
def test_builder_rejects_nonpositive_nonfinite_or_overflowing_idle(value, tmp_path):
    data = receipt()
    data["baseline_ms_by_variant"]["0"] = value
    raw = write(tmp_path / "raw.json", data)
    with pytest.raises(ValueError, match="nonfinite|finite positive"):
        routes.write_index(tmp_path / "new.json", "softmax", {"legacy": raw})
    assert not (tmp_path / "new.json").exists()
