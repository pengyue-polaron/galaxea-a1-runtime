import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_runtime_source_remains_python311_parseable_for_openpi_backend():
    failures = []
    for path in sorted((REPO / "galaxea_a1_runtime").rglob("*.py")):
        try:
            ast.parse(path.read_text(), filename=str(path), feature_version=(3, 11))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(REPO)}:{exc.lineno}: {exc.msg}")

    assert not failures, "OpenPI Python 3.11 syntax failures:\n" + "\n".join(failures)
