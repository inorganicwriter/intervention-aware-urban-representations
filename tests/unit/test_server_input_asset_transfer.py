"""The package receiver must refuse escaped paths and changed payloads."""
import hashlib

import pytest

from scripts.analysis.integrate_server_input_assets import preserve_source_metadata, rebase_asset
from scripts.analysis.receive_server_input_assets import validate_name, write_verified


@pytest.mark.parametrize("name", ["/etc/passwd", "support/../escape", "data/active/causal/treatment_unit_list.parquet",
                                  "data/active/alphaearth_grid_v2/../../escape", "data\\active\\alphaearth_grid_v2\\x"])
def test_receiver_rejects_paths_outside_new_asset_roots(name):
    with pytest.raises(ValueError):
        validate_name(name)


def test_receiver_verifies_source_and_stored_content(tmp_path):
    validate_name("data/active/alphaearth_grid_v2/embeddings.parquet")
    data = b"unchanged source asset bytes"
    record = {"path": "data/active/alphaearth_grid_v2/embeddings.parquet",
              "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    target = tmp_path / "asset.parquet"
    write_verified(target, data, record)
    assert target.read_bytes() == data
    with pytest.raises(ValueError, match="checksum mismatch"):
        write_verified(target, b"changed source asset bytes!!", record)
    assert target.read_bytes() == data


def test_asset_rebase_preserves_relative_aliases_and_rejects_escape():
    project = "/home/nas/lsr/MIT_summer"
    root = project + "/data/active/satellite_event_sequences_v2"
    assert rebase_asset("images/c/a.tif", root, project) == root + "/images/c/a.tif"
    assert rebase_asset("../satellite_grid_images_v1/images/c/a.tif", root, project) == project + "/data/active/satellite_grid_images_v1/images/c/a.tif"
    assert rebase_asset(r"D:\User\Desktop\MIT_summer\data\active\satellite_event_sequences_v2\images\c\a.tif", root, project) == root + "/images/c/a.tif"
    assert rebase_asset(None, root, project) is None
    with pytest.raises(ValueError, match="escapes"):
        rebase_asset("../../../../../../etc/passwd", root, project)


def test_binding_retry_preserves_original_metadata_after_partial_publication(tmp_path):
    roles, backup = tmp_path / "roles", tmp_path / "originals"
    roles.mkdir()
    records = {}
    names = ["state_inputs.parquet", "satellite_state_sequences.parquet", "manifest.json", "table_binding_audit.json"]
    for name in names:
        data = f"original {name}".encode()
        (roles / name).write_bytes(data)
        records[f"data/active/multimodal_inputs_v1/{name}"] = {"sha256": hashlib.sha256(data).hexdigest()}
    sources = preserve_source_metadata(roles, backup, records)
    for name in names:
        (roles / name).write_bytes(b"partially published server version")
    repeated = preserve_source_metadata(roles, backup, records)
    for name in names:
        assert repeated[name] == sources[name]
        assert repeated[name].read_bytes() == f"original {name}".encode()
    (backup / "state_inputs.parquet").write_bytes(b"damaged preserved source")
    with pytest.raises(ValueError, match="Preserved source metadata checksum mismatch"):
        preserve_source_metadata(roles, backup, records)
    assert (roles / "state_inputs.parquet").read_bytes() == b"partially published server version"
