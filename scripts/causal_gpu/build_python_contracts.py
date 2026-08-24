"""Build versioned GSC/MC tuning contracts without invoking R or fect."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.gpu.contract_builder import build_python_contract  # noqa: E402
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime  # noqa: E402


def _infer_estimator(path: Path, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    parts = {part.lower() for part in path.parts}
    if "xu_gsc" in parts:
        return "gsc"
    if "matrix_completion" in parts:
        return "mc"
    raise ValueError(f"cannot infer estimator from path; pass --estimator: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--panel", type=Path, action="append")
    source.add_argument("--root", type=Path)
    parser.add_argument("--estimator", choices=("auto", "gsc", "mc"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-tasks", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    panels = args.panel or sorted(args.root.rglob("estimation_panel.parquet"))
    if args.max_tasks is not None:
        panels = panels[: args.max_tasks]
    if not panels:
        raise ValueError("no estimation panels were discovered")
    runtime: TorchRuntime | None = None
    completed = 0
    for panel in panels:
        estimator = _infer_estimator(panel, args.estimator)
        if estimator == "mc" and runtime is None:
            runtime = TorchRuntime(
                RuntimeConfig(device=args.device, dtype="float64", seed=20260725)
            )
        manifest = build_python_contract(
            panel,
            estimator,  # type: ignore[arg-type]
            runtime=runtime,
            force=args.force,
        )
        completed += 1
        print(f"Built {estimator} contract: {manifest}")
    print(f"Completed {completed}/{len(panels)} contract(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
