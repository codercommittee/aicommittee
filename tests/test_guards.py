"""Guards: no study imports inside the package, an import from outside works, and no barred
wording in prose, documentation or source."""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "codercommittee"
STUDY_MODULES = ("pipeline", "figures", "extension", "datasets", "open_weight", "recombine", "rescore")
# Wording that never reaches prose, documentation or output: the study is exploratory and descriptive.
BARRED = (
    "pre-declared",
    "pre_declared",
    "predeclared",
    "pre-registered",
    "preregistered",
    "confirmatory",
    "exploratory-selected",
    "fixed before",
    "fixed in advance",
    "in advance",
    "re-tested",
    "retested",
    "tested out of sample",
    "out of sample",
    "chosen by hand",
    "replication",
    "recommended",
    "primary",
    "frozen",
)
PROSE = [
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "DESIGN.md",
    ROOT / "quickstart.py",
    *sorted((ROOT / "docs").rglob("*.md")),
    ROOT / "mkdocs.yml",
]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_package_imports_nothing_from_the_study() -> None:
    for path in sorted(PACKAGE.rglob("*.py")):
        hits = _imports(path) & set(STUDY_MODULES)
        assert not hits, f"{path.relative_to(ROOT)} imports {sorted(hits)}"


def test_import_from_a_temporary_directory_outside_the_repository(tmp_path: Path) -> None:
    # Drop the study's code directory, wherever it sits on the path; keep site-packages (numpy, the package itself).
    code = (
        "import sys; sys.path = [p for p in sys.path if not p.rstrip('/').endswith('/code')]; "
        "import codercommittee; print(codercommittee.__version__)"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.2.4"


def _prose_of_source(path: Path) -> str:
    """Docstrings and comments of a module: the prose, not the identifiers or the keys of bound inputs."""
    source = path.read_text(encoding="utf-8")
    parts: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            docstring = ast.get_docstring(node)
            if docstring:
                parts.append(docstring)
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    parts.extend(token.string for token in tokens if token.type == tokenize.COMMENT)
    return "\n".join(parts)


def test_no_barred_wording_in_prose_or_source() -> None:
    texts = {path: path.read_text(encoding="utf-8") for path in PROSE if path.is_file()}
    for path in sorted(PACKAGE.rglob("*.py")) + sorted((ROOT / "parity").glob("*.py")):
        texts[path] = _prose_of_source(path)
    hits: list[str] = []
    for path, text in texts.items():
        lowered = text.lower()
        # Whole words: ``frozenset`` is a type, not the word.
        hits.extend(
            f"{path.relative_to(ROOT)}: {term!r}"
            for term in BARRED
            if re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", lowered)
        )
    assert not hits, "\n".join(hits)
