"""Content-addressed provenance helpers for causal GPU artifacts."""

from __future__ import annotations

import hashlib
import platform
import sys
from collections.abc import Iterable
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from urban_intervention.utils import sha256_file

NUMERICAL_PACKAGES = ("numpy", "pandas", "pyarrow", "torch")


def estimator_source_files(estimator: str) -> tuple[Path, ...]:
    """Return numerical source files that a qualification receipt must bind."""
    package = Path(__file__).resolve().parent
    common = (
        package / "contracts.py",
        package / "formal_runner.py",
        package / "inference.py",
        package / "io.py",
        package / "panel_builder.py",
        package / "provenance.py",
        package / "qualification.py",
        package / "runtime.py",
        package / "tuning_cache.py",
    )
    specific = {
        "matching": (
            package / "matching.py",
            package / "matching_io.py",
            package / "control_design.py",
            package / "fixed_control.py",
        ),
        "gsc": (package / "gsc.py", package / "linalg.py"),
        "mc": (package / "matrix_completion.py", package / "linalg.py"),
    }
    if estimator not in specific:
        raise ValueError(f"unknown causal estimator source set: {estimator}")
    return tuple(sorted({*common, *specific[estimator]}, key=str))


def source_code_fingerprint(
    paths: Iterable[str | Path], *, root: str | Path
) -> str:
    """Hash normalized source bytes and relative paths deterministically.

    Line endings are normalized so the same archive has one identity on
    Windows and Linux. Absolute paths and timestamps never enter the digest.
    """
    resolved_root = Path(root).resolve()
    resolved_paths = sorted({Path(path).resolve() for path in paths}, key=str)
    if not resolved_paths:
        raise ValueError("cannot fingerprint an empty source set")
    digest = hashlib.sha256(b"urban-intervention-estimator-source\0")
    for path in resolved_paths:
        if not path.is_file():
            raise FileNotFoundError(f"cannot fingerprint missing source: {path}")
        try:
            relative = path.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"source is outside fingerprint root: {path}") from exc
        content = path.read_bytes().replace(b"\r\n", b"\n")
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


@lru_cache(maxsize=3)
def estimator_code_fingerprint(estimator: str) -> str:
    """Return the immutable numerical source identity for one estimator."""
    package = Path(__file__).resolve().parent
    return source_code_fingerprint(
        estimator_source_files(estimator),
        root=package.parent,
    )


def all_estimator_code_fingerprints() -> dict[str, str]:
    """Return stable-name to source-fingerprint bindings for qualification."""
    return {
        estimator: estimator_code_fingerprint(estimator)
        for estimator in ("matching", "gsc", "mc")
    }


def file_sha256(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Compatibility name for the package-wide streaming SHA-256 helper."""
    return sha256_file(path, block_size=block_size)


def fingerprint_files(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Fingerprint required input files in deterministic resolved-path order."""
    unique = sorted({Path(path).resolve() for path in paths}, key=str)
    result: list[dict[str, Any]] = []
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(f"cannot fingerprint missing causal input: {path}")
        stat = path.stat()
        result.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(path),
            }
        )
    return result


def fingerprints_match(records: object, *, verify_content: bool = False) -> bool:
    """Validate sources, rehashing when metadata changed or explicitly requested."""
    if not isinstance(records, list) or not records:
        return False
    try:
        for record in records:
            if not isinstance(record, dict):
                return False
            path = Path(str(record["path"]))
            if not path.is_file():
                return False
            stat = path.stat()
            if stat.st_size != int(record["size_bytes"]):
                return False
            metadata_unchanged = stat.st_mtime_ns == int(record.get("mtime_ns", -1))
            if (verify_content or not metadata_unchanged) and (
                file_sha256(path) != str(record["sha256"])
            ):
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def python_environment() -> dict[str, Any]:
    """Capture exact Python packages that materially affect numerical artifacts."""
    packages: dict[str, str | None] = {}
    for package in NUMERICAL_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": packages,
    }


def numerical_environment_contract(
    environment: object,
    runtime: object,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize the numerical environment bound by a qualification receipt.

    Qualification is tied to the environment that actually passed R/Python
    parity.  Paths, GPU indices and platform strings are intentionally omitted:
    they do not affect the numerical contract and would prevent a qualified
    environment from being reproduced on another worker.
    """
    if not isinstance(environment, dict) or not isinstance(runtime, dict):
        return None, ["environment/runtime metadata is missing"]
    differences: list[str] = []
    python_version = str(environment.get("python_version") or "")
    python_implementation = str(environment.get("python_implementation") or "")
    if not python_version:
        differences.append("Python version is missing")
    if not python_implementation:
        differences.append("Python implementation is missing")
    packages = environment.get("packages")
    normalized_packages: dict[str, str] = {}
    if not isinstance(packages, dict):
        differences.append("package versions are missing")
    else:
        for package in NUMERICAL_PACKAGES:
            version = str(packages.get(package) or "")
            if not version:
                differences.append(f"{package} version is missing")
            normalized_packages[package] = version
    resolved_device = str(runtime.get("resolved_device") or "")
    device_type = resolved_device.partition(":")[0]
    if device_type != "cuda":
        differences.append("formal qualification must run on CUDA")
    cuda_version = str(runtime.get("cuda_version") or "")
    if not cuda_version:
        differences.append("CUDA runtime version is missing")
    cudnn_version = runtime.get("cudnn_version")
    if cudnn_version is None:
        differences.append("cuDNN version is missing")
    contract = {
        "python_version": python_version,
        "python_implementation": python_implementation,
        "packages": normalized_packages,
        "device_type": device_type,
        "dtype": runtime.get("dtype"),
        "deterministic": runtime.get("deterministic"),
        "allow_tf32": runtime.get("allow_tf32"),
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
    }
    return contract, differences


def qualified_environment_differences(
    qualified: object,
    environment: object,
    runtime: object,
) -> list[str]:
    """Compare a production worker with the environment recorded at parity."""
    current, differences = numerical_environment_contract(environment, runtime)
    if differences:
        return differences
    if not isinstance(qualified, dict) or current is None:
        return ["qualification receipt lacks a numerical environment"]
    result: list[str] = []
    for field in (
        "python_version",
        "python_implementation",
        "device_type",
        "dtype",
        "deterministic",
        "allow_tf32",
        "cuda_version",
        "cudnn_version",
    ):
        if current.get(field) != qualified.get(field):
            result.append(f"{field} differs from the qualified environment")
    qualified_packages = qualified.get("packages")
    if not isinstance(qualified_packages, dict):
        result.append("qualification receipt lacks package versions")
    else:
        for package in NUMERICAL_PACKAGES:
            if current["packages"].get(package) != qualified_packages.get(package):
                result.append(f"{package} differs from the qualified environment")
    return result
