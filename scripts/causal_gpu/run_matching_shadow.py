"""Run GPU matching against artifacts exported from the formal R design."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.gpu.contracts import (  # noqa: E402
    GPU_IMPLEMENTATION_VERSION,
    SHADOW_SCHEMA,
)
from urban_intervention.causal.gpu.matching import MatchingConfig, fit_matching  # noqa: E402
from urban_intervention.causal.gpu.matching_io import (  # noqa: E402
    compare_matching_result,
    load_matching_artifacts,
    matching_result_frames,
)
from urban_intervention.causal.gpu.provenance import (  # noqa: E402
    estimator_code_fingerprint,
    estimator_source_files,
    fingerprint_files,
    python_environment,
)
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--chunk-size", type=int, default=65_536)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-9)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = load_matching_artifacts(args.input)
    runtime = TorchRuntime(
        RuntimeConfig(
            device=args.device,
            dtype=args.dtype,
            deterministic=True,
            allow_tf32=False,
            chunk_size=args.chunk_size,
        )
    )
    config = MatchingConfig(
        candidates=int(artifacts.metadata["matching_candidates"]),
        placebo_sample=int(artifacts.metadata["placebo_sample"]),
        placebo_quantile=float(artifacts.metadata["placebo_quantile"]),
        chunk_size=args.chunk_size,
    )
    started = time.perf_counter()
    result = fit_matching(artifacts.data, config=config, runtime=runtime)
    elapsed = time.perf_counter() - started
    if artifacts.reference_candidates is not None:
        parity = compare_matching_result(
            artifacts,
            result,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
    else:
        parity = {"available": False, "passed": None}
    selected_control = artifacts.donor_ids[result.selected_index]
    manifest = {
        "schema": SHADOW_SCHEMA,
        "implementation_version": GPU_IMPLEMENTATION_VERSION,
        "code_fingerprint": estimator_code_fingerprint("matching"),
        "estimator": "matching",
        "backend": "pytorch",
        "mode": "shadow",
        "formal_eligible": False,
        "input": str(args.input.resolve()),
        "elapsed_seconds": elapsed,
        "candidate_count": len(artifacts.donor_ids),
        "selected_control": selected_control,
        "quality_passed": result.quality_passed,
        "runtime": runtime.metadata(),
        "estimator_numerical_policy": result.provenance.numerical_policy,
        "parity": parity,
        "source_fingerprints": fingerprint_files(
            [
                *[
                    path
                    for path in args.input.iterdir()
                    if path.is_file()
                    and path.name
                    in {
                        "matching_input.parquet",
                        "metadata.csv",
                        "reference_candidates.csv",
                        "reference_selection.csv",
                    }
                ],
                Path(__file__).resolve(),
                *estimator_source_files("matching"),
            ]
        ),
        "environment": python_environment(),
        "promotion_status": (
            "eligible_for_stratified_qualification_audit"
            if parity.get("passed") is True
            else "blocked_by_matching_parity"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    candidates, selection = matching_result_frames(artifacts, result)
    candidates.to_csv(args.output / "gpu_candidates.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(args.output / "gpu_selection.csv", index=False, encoding="utf-8-sig")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
