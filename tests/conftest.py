from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parity"))
FIXTURE = ROOT / "tests" / "fixtures" / "mock_cycles.tar.gz"


@pytest.fixture(scope="session")
def mock_cycles(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The portable parity fixture: three mock cycles of the study pipeline, unpacked once."""
    out = tmp_path_factory.mktemp("fixture")
    with tarfile.open(FIXTURE) as tar:
        tar.extractall(out, filter="data")
    root = out / "mock_cycles"
    return {
        "root": root,
        "expected": json.loads((root / "expected.json").read_text(encoding="utf-8")),
        "items": json.loads((root / "items.json").read_text(encoding="utf-8")),
    }


@pytest.fixture(autouse=True)
def no_provider_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may construct a real SDK client: the adapters take fakes, and a forgotten fake fails loudly."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no provider client in tests")

    for module_name, attribute in (("anthropic", "Anthropic"), ("openai", "OpenAI"), ("google.genai", "Client")):
        try:
            module = __import__(module_name, fromlist=[attribute])
        except ImportError:
            continue
        monkeypatch.setattr(module, attribute, refuse)
