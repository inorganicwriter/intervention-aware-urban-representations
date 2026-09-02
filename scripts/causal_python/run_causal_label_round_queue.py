"""Run the modular queue with resilient recovery and optional global barriers."""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.label_queue import cli, orchestrator  # noqa: E402
from urban_intervention.causal.label_queue.rounds import (  # noqa: E402
    ROUND_STATUSES,
    eligible_round_indices,
    recover_completed_task_or_recompute_stale_qualification,
    run_global_round,
)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--global-round", choices=sorted(ROUND_STATUSES))
    args, remaining = parser.parse_known_args()
    round_name = args.global_round
    orchestrator.recover_completed_task = (
        recover_completed_task_or_recompute_stale_qualification
    )
    prevalidated_sha256 = os.environ.get(
        "MIT_CAUSAL_QUALIFICATION_PREVALIDATED_SHA256", ""
    )
    if prevalidated_sha256:
        cli.validate_formal_qualification_receipt = partial(
            cli.validate_formal_qualification_receipt,
            verify_bound_sources=False,
            expected_sha256=prevalidated_sha256,
        )
    if round_name is None:
        sys.argv = [sys.argv[0], *remaining]
        return cli.main()

    def eligible(*values, **keywords):
        return eligible_round_indices(*values, round_name=round_name, **keywords)

    cli.eligible_indices = eligible
    if round_name == "same_city_matching":
        phase = "matching"
    elif round_name == "same_city_gsc":
        phase = "gsc"
    else:
        orchestrator.run_mc_stage = partial(run_global_round, round_name=round_name)
        phase = "mc"
    sys.argv = [sys.argv[0], *remaining, "--phase", phase]
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
