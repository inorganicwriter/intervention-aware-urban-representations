from pathlib import Path

import pytest

from urban_intervention.causal.setup_inputs import validate_frozen_formal_matching_spec


def _write_spec(
    root: Path,
    minimum: int,
    *,
    method: str = "outcome_mahalanobis_m5_static_refine",
    candidates: int = 5,
) -> Path:
    path = root / "data" / "active" / "causal" / "formal_matching_inputs"
    path.mkdir(parents=True)
    spec = (
        'list(schema = "formal_counterfactual_design_v1", '
        f"minimum_complete_families = {minimum}L, "
        f'matching_method = "{method}", '
        f"matching_candidates = {candidates}L)\n"
    )
    (path / "formal_matching_spec.dput").write_text(spec, encoding="utf-8")
    return path / "formal_matching_spec.dput"


def test_frozen_formal_spec_validator_accepts_current_contract(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, 1)

    result = validate_frozen_formal_matching_spec(tmp_path)

    assert result["path"] == str(path)
    assert result["minimum_complete_families"] == 1
    assert result["matching_method"] == "outcome_mahalanobis_m5_static_refine"
    assert result["matching_candidates"] == 5


def test_frozen_formal_spec_validator_rejects_stale_contract(tmp_path: Path) -> None:
    _write_spec(tmp_path, 2)

    with pytest.raises(ValueError, match="Frozen formal matching spec is stale"):
        validate_frozen_formal_matching_spec(tmp_path)


def test_frozen_formal_spec_validator_rejects_stale_matching_method(
    tmp_path: Path,
) -> None:
    _write_spec(tmp_path, 1, method="legacy_panelmatch")

    with pytest.raises(ValueError, match="matching_method"):
        validate_frozen_formal_matching_spec(tmp_path)
