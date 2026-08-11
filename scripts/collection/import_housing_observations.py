"""Import an authorized housing export without filtering cities or years."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.data.paths import (  # noqa: E402
    RAW_PLATFORM_EXPORT_DIR,
    STAGING_HOUSING_STANDARDIZED_DIR,
)
from urban_intervention.pipelines.housing.importer import (  # noqa: E402
    import_authorized_export,
    import_large_xlsx_authorized_export,
    resume_large_xlsx_authorized_export,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_PLATFORM_EXPORT_DIR / "authorized_imports",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=STAGING_HOUSING_STANDARDIZED_DIR,
    )
    parser.add_argument(
        "--large-xlsx-chunk-rows",
        type=int,
        default=0,
        help="Use bounded-memory streaming import for large .xlsx files.",
    )
    parser.add_argument(
        "--resume-large-xlsx",
        action="store_true",
        help="Finalize already-written streaming parts after an interruption.",
    )
    args = parser.parse_args()
    if args.resume_large_xlsx:
        output, manifest = resume_large_xlsx_authorized_export(
            args.input, args.mapping, args.raw_root, args.staging_root
        )
    elif args.large_xlsx_chunk_rows:
        output, manifest = import_large_xlsx_authorized_export(
            args.input,
            args.mapping,
            args.raw_root,
            args.staging_root,
            chunk_rows=args.large_xlsx_chunk_rows,
        )
    else:
        output, manifest = import_authorized_export(
            args.input, args.mapping, args.raw_root, args.staging_root
        )
    print(f"observations={output}")
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
