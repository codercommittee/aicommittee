"""The README and the documentation site are executable truth.

Every fenced Python block of README.md and of every page under docs/ runs as written, in mock mode,
without a network; a Python block whose fence carries a ``title="..."`` names what it needs (a
running Ollama server, one of the study's run directories) and is compiled but not run. Every
``codercommittee`` command line in the README and the pages parses with the real parser; every
``pip install`` line names the package or a declared extra; every tool the development block names
is importable; every backticked public name in README.md and DESIGN.md resolves; the arm names,
statuses, kinds, seeds and constants the prose lists are the code's; every page is in the site's
navigation; the reference lists every exported name and every module. Prose claims that other
tests hold, by test module:

    quickstart under a minute ....................... test_committee (quickstart_runs_end_to_end)
    an item without a reference is never scored ...... test_data (run_complete_and_scored_populations)
    any status other than ok removes the item ........ test_arms (incomplete_items_leave_every_arm)
    the pool aligns at 0.70, k of nine ............... test_combine (pool_members_count_support)
    Jaccard for a label set, macro-F1 for a label .... test_scoring (labels_jaccard, macro_f1 oracles)
    ties by seat order ............................... test_compare (best_single_ties_fall_to_seat_order)
    gap = premium - spread ........................... test_compare, test_properties_tables
    three coders and three draws, refused elsewhere .. test_arms (other_committee_sizes_are_refused)
    a forgotten fake fails loudly .................... test_providers (no_real_sdk_client)
    resume asks only for what is missing ............. test_store, test_integration (crashed_run_resumes)
    stored replies re-validated ...................... test_store (stored_values_are_never_authoritative)
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import runpy
import shlex
import sys
import tomllib
import types
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest

import codercommittee
from codercommittee import Item, MockCoder, MockEmbedder, Task, arm_names, collect, study
from codercommittee.cli import build_parser
from codercommittee.data import KINDS, AnswerStatus

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.rglob("*.md"))
STUDY_DOC = DOCS / "study.md"
DESIGN = ROOT / "DESIGN.md"
FENCE = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)


def blocks(path: Path) -> list[tuple[str, str]]:
    """Every fenced block of a document as (language, body); the language is the fence's first word."""
    return [(info.split(" ", 1)[0], body) for info, body in FENCE.findall(path.read_text(encoding="utf-8"))]


def python_blocks(path: Path) -> list[tuple[str, str]]:
    """Every Python block as (the whole fence info string, body)."""
    found = FENCE.findall(path.read_text(encoding="utf-8"))
    return [(info, body) for info, body in found if info.split(" ", 1)[0] == "python"]


def shell_lines(body: str) -> list[str]:
    """Logical lines of a shell block, continuations joined, comments stripped."""
    joined = re.sub(r"\\\n\s*", " ", body)
    lines = []
    for raw in joined.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


@pytest.fixture
def sentence_transformers_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """The README's sentence-transformers example runs against the real package when installed, else a stand-in."""
    if importlib.util.find_spec("sentence_transformers") is not None:
        return
    module = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, name: str) -> None:
            self.name = name

        def encode(self, texts: list[str]) -> Any:
            return np.asarray([MockEmbedder(16).embed(text) for text in texts])

    module.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_every_python_block_of_the_readme_runs_as_written(
    sentence_transformers_stub: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    python_blocks = [body for language, body in blocks(README) if language == "python"]
    assert len(python_blocks) >= 3
    script = tmp_path / "readme_blocks.py"
    script.write_text("\n\n".join(python_blocks), encoding="utf-8")
    namespace = runpy.run_path(str(script))
    out = capsys.readouterr().out
    assert "blind pick" in out and "vote" in out  # the first block prints the comparison table
    assert namespace["run"].coders == ("alpha", "beta", "gamma")
    assert len(namespace["committee"].coders) == 3  # the provider block built three live adapters, no call made
    embedder = namespace["embedder"]  # the sentence-transformers block
    space = embedder.space(Item("x", "t"), ["heavy rain", "the bridge"])
    assert len(space) == 2 and 0.0 <= space.cosine("heavy rain", "the bridge") <= 1.0


@pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(DOCS)))
def test_every_python_block_of_every_page_runs_as_written(
    page: Path, sentence_transformers_stub: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fence of exactly ``python`` runs; ``python title="..."`` names what the block needs and is only compiled."""
    runnable = [body for info, body in python_blocks(page) if info == "python"]
    for info, body in python_blocks(page):
        if info != "python":
            assert 'title="' in info, f"{page.name}: a Python block that is not run names what it needs in its title"
            compile(body, f"{page.name}:{info}", "exec")
    if not runnable:
        return
    monkeypatch.chdir(tmp_path)  # the pages write items, runs and results into the working directory
    script = tmp_path / "page.py"
    script.write_text("\n\n".join(runnable), encoding="utf-8")
    runpy.run_path(str(script))


def test_every_command_line_in_the_docs_parses() -> None:
    parser = build_parser()
    seen = 0
    for path in (README, *PAGES):
        for language, body in blocks(path):
            if language == "python":
                continue
            for line in shell_lines(body):
                if not line.startswith("codercommittee "):
                    continue
                argv = shlex.split(line)[1:]
                try:
                    parser.parse_args(argv)
                except SystemExit as exc:  # argparse reports usage errors by exiting
                    pytest.fail(f"{path.name}: {line!r} does not parse ({exc})")
                seen += 1
    assert seen >= 10


def test_install_and_development_lines_name_what_exists() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = set(project["optional-dependencies"])
    checked = 0
    for language, body in blocks(README) + blocks(DOCS / "getting-started.md"):
        if language == "python":
            continue
        for line in shell_lines(body):
            if line.startswith("pip install"):
                spec = shlex.split(line)[-1]
                name, _, extra_list = spec.partition("[")
                assert name in ("codercommittee", "."), line
                if extra_list:
                    assert set(extra_list.rstrip("]").split(",")) <= extras, line
                checked += 1
            elif line.startswith("python quickstart.py"):
                assert (ROOT / "quickstart.py").is_file()
                checked += 1
            elif "pytest" in line.split():
                assert importlib.util.find_spec("pytest")
                checked += 1
            elif line.startswith("ruff"):
                assert importlib.util.find_spec("ruff") and importlib.util.find_spec("mypy")
                checked += 1
            elif line.startswith("mkdocs"):
                assert importlib.util.find_spec("mkdocs") and (ROOT / "mkdocs.yml").is_file()
                checked += 1
    assert checked >= 6


def package_modules() -> list[Any]:
    """Every module of the package, for resolving names the documents mention."""
    import pkgutil

    found = [
        importlib.import_module(info.name) for info in pkgutil.walk_packages(codercommittee.__path__, "codercommittee.")
    ]
    return [codercommittee, *found]


CLASSES = {
    "run": codercommittee.Run,
    "scores": codercommittee.Scores,
    "Task": codercommittee.Task,
    "Run": codercommittee.Run,
    "Scores": codercommittee.Scores,
    "Comparison": codercommittee.Comparison,
    "Unit": codercommittee.Unit,
}
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*(\(\))?$")


def resolves(token: str, modules: list[Any]) -> bool:
    """Whether a backticked name is a name of the package: a class, function, method or dotted path."""
    name = token.rstrip("()")
    if name.startswith("codercommittee."):
        try:
            importlib.import_module(name)
            return True
        except ImportError:
            module_name, _, attribute = name.rpartition(".")
            try:
                return hasattr(importlib.import_module(module_name), attribute)
            except ImportError:
                return False
    head, _, tail = name.partition(".")
    if tail and head in CLASSES:
        return hasattr(CLASSES[head], tail.rstrip("()"))
    if any(hasattr(module, name) for module in modules):
        return True
    return any(hasattr(cls, name) for cls in CLASSES.values())


def public_tokens(path: Path) -> set[str]:
    """Backticked tokens of a document that look like Python names: CamelCase classes, calls, dotted paths."""
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    tokens = set()
    for token in re.findall(r"`([^`\n]+)`", text):
        if not IDENTIFIER.match(token) or re.search(r"\.(json|jsonl|csv|md|py|toml|txt)$", token):
            continue
        is_class = re.match(r"^[A-Z][A-Za-z0-9]+$", token) is not None
        is_call = token.endswith("()")
        is_path = token.startswith("codercommittee.") or (token.split(".")[0] in CLASSES and "." in token)
        if is_class or is_call or is_path:
            tokens.add(token)
    return tokens


def test_every_backticked_public_name_in_the_readme_resolves() -> None:
    modules = package_modules()
    unresolved = sorted(token for token in public_tokens(README) if not resolves(token, modules))
    assert not unresolved, unresolved


def test_every_class_the_design_names_exists() -> None:
    """DESIGN.md also names the study pipeline's functions (its module map); the package's own classes must exist."""
    modules = package_modules()
    # The module map's study column names the pipeline's own classes; those are not the package's.
    study_classes = {"SpanSimilarity", "EmbeddingStore", "PanelPredictionResult", "ValidationResult", "MockClient"}
    classes = {token for token in public_tokens(DESIGN) if re.match(r"^[A-Z][A-Za-z0-9]+$", token)} - study_classes
    assert len(classes) >= 5
    unresolved = sorted(token for token in classes if not resolves(token, modules))
    assert not unresolved, unresolved


def test_the_prose_lists_the_codes_arm_names_statuses_kinds_and_seeds() -> None:
    readme = README.read_text(encoding="utf-8")
    run = collect([Item("a", "t", True)], Task("label", {"y": ""}), [MockCoder("c1"), MockCoder("c2"), MockCoder("c3")])
    names = arm_names(run)
    assert "vote" in names and all(f"single:{c}" in names and f"self:{c}" in names for c in run.coders)
    relations = collect(
        [Item("a", "heavy rain filled the river so the bridge washed out")],
        Task("tuples"),
        [MockCoder("c1"), MockCoder("c2"), MockCoder("c3")],
        embedder=MockEmbedder(),
    )
    assert [a for a in arm_names(relations) if a.startswith("pool:")] == ["pool:1", "pool:2", "pool:3"]
    for arm in ("pool:1", "pool:2", "pool:3", "`vote`", "single:<coder>", "self:<coder>"):
        assert arm in readme
    for status in get_args(AnswerStatus):
        assert f"`{status}`" in readme
    for kind in KINDS:
        assert f"`{kind}`" in readme
    doc = STUDY_DOC.read_text(encoding="utf-8")
    assert "42019 for FinCausal" in doc and "42017 for CausalNews" in doc
    assert study.BOOTSTRAP_SEED == {"miric": 42, "gvfc": 42, "politicause": 42, "fincausal": 42019, "causalnews": 42017}
    assert "3,072 dimensions" in readme and MockEmbedder().dimensions == 3072


def nav_pages(node: Any) -> list[str]:
    """The page paths of a mkdocs navigation tree."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [page for value in node.values() for page in nav_pages(value)]
    return [page for entry in node for page in nav_pages(entry)]


def test_every_page_is_in_the_navigation_and_every_navigation_entry_is_a_page() -> None:
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    in_nav = set(nav_pages(config["nav"]))
    on_disk = {str(page.relative_to(DOCS)) for page in PAGES}
    assert in_nav == on_disk, {"not in nav": on_disk - in_nav, "not on disk": in_nav - on_disk}
    assert config["repo_url"] == "https://github.com/codercommittee/aicommittee"
    assert any("pypi.org/project/codercommittee" in link["link"] for link in config["extra"]["social"])


def test_the_reference_lists_every_exported_name_and_every_module() -> None:
    import pkgutil

    package_page = (DOCS / "reference" / "codercommittee.md").read_text(encoding="utf-8")
    missing = [name for name in codercommittee.__all__ if name != "__version__" and f"[{name}][" not in package_page]
    assert not missing, missing
    documented = set()
    for page in (DOCS / "reference").glob("*.md"):
        documented.update(re.findall(r"^::: (codercommittee[\w.]*)", page.read_text(encoding="utf-8"), re.MULTILINE))
    modules = {info.name for info in pkgutil.walk_packages(codercommittee.__path__, "codercommittee.")}
    assert modules <= documented, sorted(modules - documented)
    assert "codercommittee" in documented


def test_the_concepts_page_lists_the_codes_constants() -> None:
    text = (DOCS / "concepts.md").read_text(encoding="utf-8")
    shown = {
        ("dedupe", "DEDUP_THRESHOLD"): "0.90",
        ("dedupe", "LINKAGE_FLOOR"): "0.85",
        ("align", "LOW_THRESHOLD"): "0.60",
        ("combine", "VOTE_ACCEPT_AT"): "0.80",
        ("arms", "POOL_ACCEPT_AT"): "0.70",
        ("scoring", "MATCH_THRESHOLD"): "0.60",
        ("similarity", "DECIMALS"): "12",
        ("matching", "ENUMERATION_GUARD"): "24",
        ("matching", "VALUE_TOLERANCE"): "1e-9",
        ("compare", "TIE_TOLERANCE"): "1e-12",
        ("compare", "REPLICATES"): "2,000",
        ("compare", "PICK_DRAWS"): "500",
        ("arms", "COMMITTEE_SIZE"): "3",
        ("arms", "DRAWS"): "3",
    }
    for (module_name, name), value in shown.items():
        module = importlib.import_module(f"codercommittee.{module_name}")
        assert f"`codercommittee.{module_name}.{name}`" in text, (module_name, name)
        forms = (f"| {value} ", f"and {value} |", f"({value},")
        assert any(form in text for form in forms), (name, value)
        assert float(value.replace(",", "")) == float(getattr(module, name)), (name, value)
    from codercommittee.compare import PERCENTILES

    assert PERCENTILES == (2.5, 97.5) and "2.5 and 97.5" in text
