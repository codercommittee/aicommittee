"""The command line on mock coders: code, resume, score, compare, agreement, pick, tasks; usage errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codercommittee import Item, Relation, Task, write_items
from codercommittee.cli import CLIError, load_task, main, parse_coder, parse_embedder, parse_model, parse_referee
from codercommittee.coders import MockCoder, ModelCoder
from codercommittee.embedders import MockEmbedder, OpenAICompatibleEmbedder, OpenAIEmbedder
from codercommittee.io import read_jsonl
from codercommittee.providers import AnthropicModel, GoogleModel, OpenAICompatibleModel, OpenAIModel
from codercommittee.referee import BandRule, LLMReferee
from codercommittee.store import LoggedReferee, RunStore, task_to_json

TUPLES = Task("tuples")
TEXTS = [
    "heavy rain filled the river so the bridge washed out",
    "the council cut the subsidy and the buses stopped running",
    "frost hit the orchard early and the press stood idle",
    "the mill closed in march so forty workers left town",
    "a new tariff raised prices and the bakery closed its shop",
    "early warnings prevent a harvest collapse in april",
]


@pytest.fixture
def items_file(tmp_path: Path) -> Path:
    items = [
        Item(n, text, [Relation(cause=" ".join(text.split()[:3]), effect=" ".join(text.split()[-3:]))])
        for n, text in enumerate(TEXTS)
    ]
    path = tmp_path / "items.jsonl"
    write_items(items, path, TUPLES)
    return path


def run(*argv: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    status = main(list(argv))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_code_score_compare_agreement_pick_and_resume(
    tmp_path: Path, items_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"kind": "tuples", "prompt_file": "prompt.md"}))
    (tmp_path / "prompt.md").write_text("Relations in: {{TEXT}}")
    coders = ["--coder", "a=mock", "--coder", "b=mock:1", "--coder", "c=mock:2"]
    status, out, _ = run(
        "code",
        "--task",
        str(task),
        "--items",
        str(items_file),
        "--out",
        str(tmp_path / "one"),
        *coders,
        "--embedder",
        "mock:64",
        "--referee",
        "band",
        capsys=capsys,
    )
    assert status == 0 and "6 of 6 items complete" in out and "best single" in out
    one = tmp_path / "one"
    assert (one / "run.json").is_file() and (one / "predictions.jsonl").is_file() and (one / "levels.json").is_file()
    assert {row["arm"] for row in read_jsonl(one / "predictions.jsonl")} >= {"vote", "pool:2"}
    answered = len((one / "answers.jsonl").read_text().splitlines())
    status, out, _ = run(
        "code",
        "--task",
        str(task),
        "--items",
        str(items_file),
        "--out",
        str(one),
        *coders,
        "--embedder",
        "mock:64",
        capsys=capsys,
    )
    assert (
        status == 0 and len((one / "answers.jsonl").read_text().splitlines()) == answered
    )  # resumed, nothing re-asked
    status, out, _ = run(
        "code",
        "--task",
        "study:causalnews",
        "--items",
        str(items_file),
        "--out",
        str(tmp_path / "two"),
        "--coder",
        "a=mock:7",
        "--coder",
        "b=mock:8",
        "--coder",
        "c=mock:9",
        "--embedder",
        "mock:64",
        capsys=capsys,
    )
    assert status == 0
    status, out, _ = run("score", str(one), "--out", str(tmp_path / "scores.csv"), capsys=capsys)
    assert status == 0 and "vote" in out and (tmp_path / "scores.csv").is_file()
    status, out, _ = run(
        "compare",
        str(one),
        str(tmp_path / "two"),
        "--seed",
        "3",
        "--draws",
        "30",
        "--out",
        str(tmp_path / "table.json"),
        capsys=capsys,
    )
    assert status == 0 and "blind pick" in out and json.loads((tmp_path / "table.json").read_text())["runs"] == 2
    status, out, _ = run("agreement", str(one), "--out", str(tmp_path / "agreement.json"), capsys=capsys)
    assert status == 0 and "cross-coder disagreement" in out and "matched Jaccard" in out
    status, out, _ = run(
        "pick",
        str(one),
        "--n",
        "2",
        "--seed",
        "3",
        "--draws",
        "10",
        "--out",
        str(tmp_path / "pick.json"),
        capsys=capsys,
    )
    assert status == 0 and "held-out gap" in out and json.loads((tmp_path / "pick.json").read_text())["n"] == 2
    status, out, _ = run("tasks", capsys=capsys)
    assert status == 0 and "causalnews" in out and "tuples" in out
    status, out, _ = run("tasks", "miric", "--out", str(tmp_path / "miric.json"), capsys=capsys)
    assert status == 0 and json.loads((tmp_path / "miric.json").read_text())["key"] == "is_ai_patent"


def test_usage_errors_exit_with_status_two(
    tmp_path: Path, items_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, _, err = run(
        "code",
        "--task",
        "study:nothing",
        "--items",
        str(items_file),
        "--out",
        str(tmp_path / "r"),
        "--coder",
        "a=mock",
        capsys=capsys,
    )
    assert status == 2 and "unknown study task" in err
    status, _, err = run(
        "code",
        "--task",
        "study:causalnews",
        "--items",
        str(items_file),
        "--out",
        str(tmp_path / "r"),
        "--coder",
        "a=mock",
        "--coder",
        "b=mock",
        "--coder",
        "c=mock",
        capsys=capsys,
    )
    assert status == 2 and "needs an embedder" in err
    status, _, err = run(
        "code",
        "--task",
        "study:causalnews",
        "--items",
        str(items_file),
        "--out",
        str(tmp_path / "r"),
        "--coder",
        "a=mock",
        "--coder",
        "a=mock",
        "--coder",
        "c=mock",
        "--embedder",
        "mock",
        capsys=capsys,
    )
    assert status == 2 and "distinct" in err
    status, _, err = run(
        "code",
        "--task",
        "study:causalnews",
        "--items",
        str(tmp_path / "none.jsonl"),
        "--out",
        str(tmp_path / "r"),
        "--coder",
        "a=mock",
        "--embedder",
        "mock",
        capsys=capsys,
    )
    assert status == 2
    status, _, err = run("score", str(tmp_path / "nowhere"), capsys=capsys)
    assert status == 2 and "holds no run" in err
    status, _, err = run(
        "code",
        "--task",
        "study:miric",
        "--items",
        str(items_file),
        "--out",
        str(tmp_path / "r"),
        "--coder",
        "a=mock",
        "--coder",
        "b=mock",
        capsys=capsys,
    )
    assert status == 2  # the items file carries relation references, not booleans


def test_specs_build_the_right_objects() -> None:
    assert isinstance(parse_coder("a=mock"), MockCoder) and parse_coder("a=mock:5").seed == 5  # type: ignore[attr-defined]
    coder = parse_coder("gpt=openai:gpt-5.6-terra")
    assert (
        isinstance(coder, ModelCoder) and isinstance(coder.model, OpenAIModel) and coder.model.model == "gpt-5.6-terra"
    )
    assert isinstance(parse_coder("c=anthropic:claude-sonnet-5").model, AnthropicModel)  # type: ignore[attr-defined]
    assert isinstance(parse_coder("g=google:gemini-3.8-flash").model, GoogleModel)  # type: ignore[attr-defined]
    local = parse_coder("l=compatible:llama@http://localhost:8000/v1")
    assert isinstance(local.model, OpenAICompatibleModel) and local.model.base_url == "http://localhost:8000/v1"  # type: ignore[attr-defined]
    for bad in ("nameless", "a=", "a=mock:x", "a=compatible:llama", "a=unknown:m", "a=openai"):
        with pytest.raises(CLIError):
            parse_coder(bad)
    assert isinstance(parse_embedder("mock", TUPLES), MockEmbedder) and parse_embedder("mock:8", TUPLES).dimensions == 8  # type: ignore[union-attr]
    openai = parse_embedder("openai:text-embedding-3-small:256", TUPLES)
    assert isinstance(openai, OpenAIEmbedder) and openai.dimensions == 256
    assert isinstance(parse_embedder("compatible:bge@http://x/v1", TUPLES), OpenAICompatibleEmbedder)
    assert parse_embedder(None, Task("label", {"y": ""})) is None
    with pytest.raises(CLIError, match="needs an embedder"):
        parse_embedder(None, TUPLES)
    with pytest.raises(CLIError):
        parse_embedder("other", TUPLES)
    assert isinstance(parse_referee("band", None), BandRule) and parse_referee("band:0.7", None).accept_at == 0.7  # type: ignore[union-attr]
    assert isinstance(parse_referee("llm:openai:gpt-x", None), LLMReferee)
    assert isinstance(parse_referee("llm:openai:gpt-x", RunStore("x")), LoggedReferee)
    assert parse_referee(None, None) is None
    with pytest.raises(CLIError):
        parse_referee("judge", None)
    assert load_task("study:gvfc").key == "frames"
    with pytest.raises(CLIError, match="not found"):
        load_task("/nowhere/task.json")


def test_bad_paths_and_task_files_exit_two_with_a_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from codercommittee.io import write_items

    label = Task("label", {"yes": ""})
    (tmp_path / "task.json").write_text(json.dumps(task_to_json(label)))
    write_items([Item(1, "t", True)], tmp_path / "items.jsonl", label)
    (tmp_path / "file").write_text("x")
    base = [
        "code",
        "--task",
        str(tmp_path / "task.json"),
        "--items",
        str(tmp_path / "items.jsonl"),
        "--coder",
        "a=mock",
    ]
    assert main([*base, "--out", str(tmp_path / "file")]) == 2
    assert main([*base, "--out", str(tmp_path / "file" / "run")]) == 2
    assert main(["score", str(tmp_path / "file")]) == 2
    assert main([*base[:3], "--items", str(tmp_path), "--coder", "a=mock", "--out", str(tmp_path / "out")]) == 2
    for record in (
        {"kind": "tuples", "per_answer": [1]},
        {"kind": "tuples", "prompt_file": 5},
        {"kind": "tuples", "prompt_file": "."},
    ):
        (tmp_path / "bad.json").write_text(json.dumps(record))
        assert (
            main(
                [
                    "code",
                    "--task",
                    str(tmp_path / "bad.json"),
                    "--items",
                    str(tmp_path / "items.jsonl"),
                    "--coder",
                    "a=mock",
                    "--out",
                    str(tmp_path / "out"),
                ]
            )
            == 2
        )
    err = capsys.readouterr().err
    assert err.count("codercommittee:") == 7 and "Traceback" not in err


def test_model_options_reach_every_adapter_and_the_referee(tmp_path: Path) -> None:
    from codercommittee.cli import build_parser, model_settings

    args = build_parser().parse_args(
        [
            "code",
            "--task",
            "study:miric",
            "--items",
            "x",
            "--out",
            "y",
            "--coder",
            "a=openai:gpt-x",
            "--max-output-tokens",
            "16384",
            "--temperature",
            "0",
            "--timeout",
            "300",
        ]
    )
    settings = model_settings(args)
    assert settings == {"max_output_tokens": 16384, "temperature": 0.0, "timeout": 300.0}
    for spec in ("anthropic:claude-x", "openai:gpt-x", "google:gemini-x", "compatible:m@http://localhost:11434/v1"):
        model = parse_model(spec, settings)
        record = model.describe()  # type: ignore[attr-defined]
        assert record["max_output_tokens"] == 16384 and record["temperature"] == 0.0 and record["timeout"] == 300.0
        sent = model.arguments("p", {"type": "object"}, "s")  # type: ignore[attr-defined]
        assert sent.get("temperature", sent.get("config", {}).get("temperature")) == 0.0
    plain = parse_model("openai:gpt-x")
    assert "temperature" not in plain.arguments("p", {"type": "object"}, "s")  # type: ignore[attr-defined]
    referee = parse_referee("llm:openai:gpt-x", None, settings)
    assert isinstance(referee, LLMReferee) and referee.describe()["max_output_tokens"] == 16384
    coder = parse_coder("a=compatible:m@http://localhost:11434/v1", settings)
    assert isinstance(coder, ModelCoder) and coder.describe()["timeout"] == 300.0
    mock = parse_coder("a=mock", settings)
    assert isinstance(mock, MockCoder) and mock.describe() == {"kind": "mock", "model": "mock", "seed": 0}
