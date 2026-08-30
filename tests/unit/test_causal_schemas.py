from urban_intervention.causal.schemas import (
    CAUSAL_RESPONSE_LABELS_SCHEMA,
    accepts_legacy_version,
)


def test_response_schema_compatibility_is_explicit_and_fail_closed() -> None:
    assert accepts_legacy_version(
        CAUSAL_RESPONSE_LABELS_SCHEMA, CAUSAL_RESPONSE_LABELS_SCHEMA
    )
    assert accepts_legacy_version(
        "causal_response_labels_v1", CAUSAL_RESPONSE_LABELS_SCHEMA
    )
    assert not accepts_legacy_version(
        "causal_response_labels_v999_incompatible", CAUSAL_RESPONSE_LABELS_SCHEMA
    )
