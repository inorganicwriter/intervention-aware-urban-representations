from pathlib import Path

from urban_intervention.causal.gpu.provenance import source_code_fingerprint


def _write_source(root: Path, content: bytes) -> Path:
    path = root / "gpu" / "estimator.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path


def test_source_fingerprint_is_line_ending_independent(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = _write_source(first_root, b"x = 1\r\ny = 2\r\n")
    second = _write_source(second_root, b"x = 1\ny = 2\n")

    assert source_code_fingerprint([first], root=first_root) == source_code_fingerprint(
        [second], root=second_root
    )


def test_source_fingerprint_changes_with_source_content(tmp_path: Path) -> None:
    root = tmp_path / "source"
    path = _write_source(root, b"x = 1\n")
    before = source_code_fingerprint([path], root=root)
    path.write_bytes(b"x = 2\n")

    assert source_code_fingerprint([path], root=root) != before
