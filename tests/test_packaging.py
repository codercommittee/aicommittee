"""Packaging: the sdist and wheel hold what they must, pass twine's check, and install into a clean
environment where the command line, the quickstart and the import work without extras.

Marked ``release``: deselected by default (``pytest -m release`` runs it; CI's package job does).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.release


def run(*argv: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), cwd=cwd, env=env, capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def dist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dist")
    built = run(sys.executable, "-m", "build", "--outdir", str(out), str(ROOT))
    assert built.returncode == 0, built.stderr[-3000:]
    return out


def test_the_wheel_and_sdist_hold_what_they_must(dist: Path) -> None:
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
    assert "codercommittee/py.typed" in names and "codercommittee/referee_prompt.md" in names
    for dataset in ("miric", "gvfc", "politicause", "fincausal", "causalnews"):
        assert f"codercommittee/study_prompts/{dataset}.md" in names
    assert not any(name.startswith(("tests/", "parity/")) for name in names)
    assert any(name.endswith("entry_points.txt") for name in names)
    with tarfile.open(sdists[0]) as sdist:
        members = [member.name.split("/", 1)[1] for member in sdist.getmembers() if "/" in member.name]
    for required in (
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "DESIGN.md",
        "quickstart.py",
        "docs/study.md",
        "tests/conftest.py",
        "tests/fixtures/mock_cycles.tar.gz",
        "parity/full_cycles.py",
        "src/codercommittee/study_prompts/causalnews.md",
    ):
        assert required in members, required
    assert not any(name.startswith("docs/api") for name in members)


def test_twine_accepts_the_distributions(dist: Path) -> None:
    checked = run(sys.executable, "-m", "twine", "check", *(str(path) for path in dist.iterdir()))
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_the_wheel_installs_and_works_in_a_clean_environment(dist: Path, tmp_path: Path) -> None:
    wheel = next(dist.glob("*.whl"))
    venv = tmp_path / "venv"
    if shutil.which("uv"):
        assert run("uv", "venv", "--python", sys.executable, str(venv)).returncode == 0
        python = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
        installed = run("uv", "pip", "install", "--python", str(python), str(wheel))
    else:
        assert run(sys.executable, "-m", "venv", str(venv)).returncode == 0
        python = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
        installed = run(str(python), "-m", "pip", "install", "--quiet", str(wheel))
    assert installed.returncode == 0, installed.stderr[-2000:]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    script = venv / ("Scripts" if os.name == "nt" else "bin") / "codercommittee"
    version = run(str(script), "--version", env=env)
    assert version.returncode == 0 and "codercommittee" in version.stdout
    tasks = run(str(script), "tasks", env=env)
    assert tasks.returncode == 0 and "causalnews" in tasks.stdout
    quickstart = tmp_path / "quickstart.py"
    shutil.copy(ROOT / "quickstart.py", quickstart)
    ran = run(str(python), "-W", "error", str(quickstart), cwd=tmp_path, env=env)
    assert ran.returncode == 0, ran.stderr[-2000:]
    assert "two runs compared" in ran.stdout
    probe = "\n".join(
        [
            "import codercommittee, importlib.util",
            "assert importlib.util.find_spec('pandas') is None, 'pandas must not be installed by the bare wheel'",
            "from codercommittee import Committee, Item, MockCoder, Task",
            "committee = Committee([MockCoder('a'), MockCoder('b'), MockCoder('c')])",
            "run = committee.code([Item('x', 't', True)], Task('label', {'y': ''}))",
            "exc = None",
            "try:",
            "    run.to_frame()",
            "except ImportError as e:",
            "    exc = e",
            "assert exc is not None, 'to_frame must need pandas'",
            "print('ok', codercommittee.__version__)",
        ]
    )
    checked = run(str(python), "-c", probe, env=env)
    assert checked.returncode == 0 and checked.stdout.startswith("ok"), checked.stderr[-2000:]
