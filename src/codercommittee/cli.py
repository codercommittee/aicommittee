r"""The command line: code items into a run directory, then score, compare, measure agreement and pick.

    codercommittee code --task study:causalnews --items items.jsonl --out runs/one \\
        --coder claude=anthropic:claude-sonnet-5 --coder gpt=openai:gpt-5.6-terra \\
        --coder gemini=google:gemini-3.8-flash --embedder openai --referee llm:anthropic:claude-opus-5-5
    codercommittee score runs/one
    codercommittee compare runs/one runs/two runs/three --seed 42 --out table.csv
    codercommittee agreement runs/one
    codercommittee pick runs/one --n 20 --seed 42
    codercommittee tasks causalnews --out task.json

Specs. Coder: ``NAME=mock[:SEED]``, ``NAME=anthropic:MODEL``, ``NAME=openai:MODEL``, ``NAME=google:MODEL``,
``NAME=compatible:MODEL@BASE_URL``. Embedder: ``mock[:DIMS]``, ``openai[:MODEL[:DIMS]]``,
``compatible:MODEL@BASE_URL``. Referee: ``band[:ACCEPT_AT]`` or ``llm:PROVIDER:MODEL[@BASE_URL]``.
``--max-output-tokens``, ``--temperature`` and ``--timeout`` apply to every model built from a spec. Task:
``study:NAME`` or a JSON file with the keys of Task (``prompt`` or ``prompt_file``). A run directory
resumes when ``code`` is run again with the same task, items, coders (each seat's model) and referee;
without ``--referee`` a resume keeps the referee the run was made with.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__, study
from .arms import arms
from .coders import MockCoder, ModelCoder
from .compare import Scores, agreement, compare, pick_on_sample
from .data import SPAN_KINDS, Item, Run, Task
from .embedders import MockEmbedder, OpenAICompatibleEmbedder, OpenAIEmbedder
from .io import (
    dump,
    read_items,
    write_agreement,
    write_comparison,
    write_levels,
    write_pick,
    write_predictions,
    write_scores,
)
from .protocols import Coder, Embedder, Model, Referee
from .referee import BandRule, LLMReferee, RefereeError
from .replay import ReplayError
from .store import LoggedReferee, RunStore, StoreError, load_run, stored_referee, task_from_json, task_to_json
from .strictjson import StrictJSONError, strict_json_loads


class CLIError(Exception):
    """A usage error: printed on stderr, exit status 2."""


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------


def load_task(spec: str) -> Task:
    """The task a ``--task`` spec names: ``study:NAME`` or a JSON file."""
    if spec.startswith("study:"):
        name = spec[len("study:") :]
        if name not in study.DATASETS:
            raise CLIError(f"unknown study task {name!r}; the study tasks are {', '.join(study.DATASETS)}")
        return study.TASKS[name]
    path = Path(spec)
    if not path.is_file():
        raise CLIError(f"task file not found: {path}")
    try:
        record = strict_json_loads(path.read_text(encoding="utf-8"))
    except StrictJSONError as exc:
        raise CLIError(f"{path}: not strict JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise CLIError(f"{path}: a task file holds a JSON object")
    if "prompt_file" in record:
        if not isinstance(record["prompt_file"], str) or not record["prompt_file"]:
            raise CLIError(f"{path}: prompt_file names a file")
        prompt_path = Path(record["prompt_file"])
        if not prompt_path.is_absolute():
            prompt_path = path.parent / prompt_path
        record = {**record, "prompt": prompt_path.read_text(encoding="utf-8")}
        record.pop("prompt_file")
    try:
        return task_from_json(record)
    except (KeyError, ValueError, TypeError) as exc:
        raise CLIError(f"{path}: {exc}") from exc


ModelSettings = dict[str, Any]


def model_settings(args: argparse.Namespace | None) -> ModelSettings:
    """The adapter settings the command line gives every model: output budget, temperature, timeout."""
    settings: ModelSettings = {}
    for name in ("max_output_tokens", "temperature", "timeout"):
        value = getattr(args, name, None) if args is not None else None
        if value is not None:
            settings[name] = value
    return settings


def parse_model(spec: str, settings: ModelSettings | None = None) -> Model:
    """``anthropic:MODEL``, ``openai:MODEL``, ``google:MODEL`` or ``compatible:MODEL@BASE_URL``."""
    settings = dict(settings or {})
    provider, _, rest = spec.partition(":")
    if not rest:
        raise CLIError(f"model spec {spec!r} needs provider:model")
    if provider == "anthropic":
        from .providers.anthropic import AnthropicModel

        return AnthropicModel(rest, **settings)
    if provider == "openai":
        from .providers.openai import OpenAIModel

        return OpenAIModel(rest, **settings)
    if provider == "google":
        from .providers.google import GoogleModel

        return GoogleModel(rest, **settings)
    if provider == "compatible":
        from .providers.compatible import OpenAICompatibleModel

        model, _, base_url = rest.partition("@")
        if not model or not base_url:
            raise CLIError(f"model spec {spec!r} needs compatible:MODEL@BASE_URL")
        return OpenAICompatibleModel(model, base_url=base_url, **settings)
    raise CLIError(f"unknown provider {provider!r} in {spec!r}; use anthropic, openai, google or compatible")


def parse_coder(spec: str, settings: ModelSettings | None = None) -> Coder:
    """A coder from ``NAME=SPEC``."""
    name, _, rest = spec.partition("=")
    if not name or not rest:
        raise CLIError(f"coder spec {spec!r} needs NAME=SPEC")
    if rest == "mock" or rest.startswith("mock:"):
        seed = rest[len("mock:") :] if rest.startswith("mock:") else "0"
        if not seed.lstrip("-").isdigit():
            raise CLIError(f"coder spec {spec!r}: the mock seed is an integer")
        return MockCoder(name, seed=int(seed))
    return ModelCoder(name, parse_model(rest, settings))


def parse_embedder(spec: str | None, task: Task) -> Embedder | None:
    """An embedder from its spec, or None for the label kinds; the span kinds need one."""
    if spec is None:
        if task.kind in SPAN_KINDS:
            raise CLIError(
                f"a {task.kind} task needs an embedder: --embedder mock, openai[:MODEL[:DIMS]] or compatible:MODEL@URL"
            )
        return None
    kind, _, rest = spec.partition(":")
    if kind == "mock":
        return MockEmbedder(int(rest)) if rest else MockEmbedder()
    if kind == "openai":
        model, _, dims = rest.partition(":")
        return OpenAIEmbedder(model or "text-embedding-3-large", int(dims) if dims else 3072)
    if kind == "compatible":
        model, _, base_url = rest.partition("@")
        if not model or not base_url:
            raise CLIError(f"embedder spec {spec!r} needs compatible:MODEL@BASE_URL")
        return OpenAICompatibleEmbedder(base_url, model)
    raise CLIError(f"unknown embedder {spec!r}; use mock, openai or compatible")


def parse_referee(spec: str | None, store: RunStore | None, settings: ModelSettings | None = None) -> Referee | None:
    """A referee from its spec; an LLM referee is logged into the store when one is given."""
    if spec is None:
        return None
    kind, _, rest = spec.partition(":")
    if kind == "band":
        return BandRule(float(rest)) if rest else BandRule()
    if kind == "llm":
        referee: Referee = LLMReferee(parse_model(rest, settings))
        return LoggedReferee(referee, store) if store is not None else referee
    raise CLIError(f"unknown referee {spec!r}; use band[:ACCEPT_AT] or llm:PROVIDER:MODEL")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def levels_text(scores: Scores) -> str:
    lines = [f"{'arm':<22}{'level':>9}"]
    for arm, level in scores.levels().items():
        lines.append(f"{arm:<22}{level:>9.3f}")
    lines.append(f"best single: {scores.best_single()}; {len(scores.items)} scored items")
    return "\n".join(lines)


def _loaded(directory: str, referee: Referee | None = None) -> Run:
    try:
        return load_run(directory, referee)
    except StoreError as exc:
        raise CLIError(str(exc)) from exc


def command_code(args: argparse.Namespace) -> int:
    from .collect import collect

    task = load_task(args.task)
    items: list[Item] = read_items(args.items, task, int_ids=args.int_ids)
    settings = model_settings(args)
    coders = [parse_coder(spec, settings) for spec in args.coder]
    names = [coder.name for coder in coders]
    if len(set(names)) != len(names):
        raise CLIError("coder names must be distinct")
    embedder = parse_embedder(args.embedder, task)
    store = RunStore(args.out)
    referee = parse_referee(args.referee, store, settings)
    if referee is None and store.exists:
        referee = stored_referee(store)  # a resume keeps the referee the run was made with
    try:
        run = collect(
            items,
            task,
            coders,
            draws=args.draws,
            attempts=args.attempts,
            embedder=embedder,
            store=store,
            referee=referee,
        )
    except StoreError as exc:
        raise CLIError(str(exc)) from exc
    complete = run.complete()
    print(f"{len(complete)} of {len(run.items)} items complete; {store.answered()} answers stored in {store.dir}")
    if len(run.coders) == 3 and run.draws == 3:
        try:
            predictions = arms(run, referee)
        except StoreError as exc:
            if args.referee is None and str(exc).startswith("no stored decision"):
                raise CLIError(
                    f"{exc}; the answers are stored, but a realization this run never decided needs the referee: "
                    "run again with --referee llm:PROVIDER:MODEL (the recorded decisions are served, only the new "
                    "ones are asked)"
                ) from exc
            raise CLIError(str(exc)) from exc
        except RefereeError as exc:
            hint = (
                " (a model that thinks before it answers may need --max-output-tokens 16384)"
                if "length" in str(exc)
                else ""
            )
            raise CLIError(f"{exc}; the answers are stored, run again to decide the rest{hint}") from exc
        write_predictions(predictions, task, store.dir / "predictions.jsonl")
        print(f"predictions written to {store.dir / 'predictions.jsonl'}")
        if run.scored():
            scores = Scores.of(run, predictions)
            write_scores(scores, store.dir / "scores.csv")
            write_levels(scores, store.dir / "levels.json")
            print(levels_text(scores))
    else:
        print("the four structures need three coders and three draws; the answers are stored, nothing else is built")
    return 0


def command_score(args: argparse.Namespace) -> int:
    run = _loaded(args.run, parse_referee(args.referee, RunStore(args.run), model_settings(args)))
    if not run.scored():
        raise CLIError("no item carries a reference; nothing to score")
    scores = run.scores()
    print(levels_text(scores))
    if args.out:
        write_scores(scores, args.out)
        print(f"written to {args.out}")
    return 0


def command_compare(args: argparse.Namespace) -> int:
    tables = []
    for directory in args.runs:
        run = _loaded(directory)
        if not run.scored():
            raise CLIError(f"{directory}: no item carries a reference")
        tables.append(run.scores())
    try:
        table = compare(tables, draws=args.draws, seed=args.seed)
    except ValueError as exc:
        raise CLIError(str(exc)) from exc
    print(table)
    if args.out:
        write_comparison(table, args.out)
        print(f"written to {args.out}")
    return 0


def command_agreement(args: argparse.Namespace) -> int:
    run = _loaded(args.run)
    result = agreement(run)
    print(f"items {result.n_items}; cross-coder disagreement {result.cross:.3f}")
    for coder, value in result.within.items():
        print(f"  within {coder:<12} {value:.3f}")
    for pair, value in result.pairwise.items():
        print(f"  identical {pair:<16} {value:.3f}")
    if result.graded:
        print(f"  matched Jaccard across coders {result.graded['cross_matched_jaccard']:.3f}")
    if args.out:
        write_agreement(result, args.out)
    return 0


def command_pick(args: argparse.Namespace) -> int:
    run = _loaded(args.run)
    scores = run.scores()
    try:
        pick = pick_on_sample(scores, args.n, args.among or None, draws=args.draws, seed=args.seed)
    except ValueError as exc:
        raise CLIError(str(exc)) from exc
    share = f"{pick.pick_share:.0%} of {pick.draws} draws"
    print(
        f"a labelled sample of {pick.n} items picks the best single coder in {share}; "
        f"held-out gap {pick.heldout_gap:+.3f}; blind pick {pick.blind_gap:+.3f}"
    )
    if args.out:
        write_pick(pick, args.out)
    return 0


def command_tasks(args: argparse.Namespace) -> int:
    if not args.name:
        for name in study.DATASETS:
            task = study.TASKS[name]
            print(f"{name:<12} {task.kind:<7} key={task.key}")
        return 0
    if args.name not in study.DATASETS:
        raise CLIError(f"unknown study task {args.name!r}")
    record = task_to_json(study.TASKS[args.name])
    if args.out:
        dump(record, args.out)
        print(f"written to {args.out}")
    else:
        import json

        print(json.dumps(record, ensure_ascii=False, indent=1))
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def _model_options(parser: argparse.ArgumentParser) -> None:
    """The settings every model built from a spec takes (coders and the LLM referee alike)."""
    parser.add_argument(
        "--max-output-tokens", type=int, help="output budget per request (default 4096; a thinking model needs more)"
    )
    parser.add_argument("--temperature", type=float, help="sampling temperature (default: the provider's own)")
    parser.add_argument("--timeout", type=float, help="seconds per request before a transport failure (default 120)")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser with every subcommand."""
    parser = argparse.ArgumentParser(prog="codercommittee", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"codercommittee {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    code = commands.add_parser("code", help="code items into a run directory (resumes if it exists)")
    code.add_argument("--task", required=True, help="study:NAME or a task JSON file")
    code.add_argument("--items", required=True, help="items file (.jsonl, .json or .csv)")
    code.add_argument("--out", required=True, help="run directory")
    code.add_argument("--coder", action="append", required=True, help="NAME=SPEC, repeatable")
    code.add_argument("--embedder", help="mock[:DIMS], openai[:MODEL[:DIMS]] or compatible:MODEL@URL")
    code.add_argument("--referee", help="band[:ACCEPT_AT] or llm:PROVIDER:MODEL[@URL]")
    code.add_argument("--draws", type=int, default=3)
    code.add_argument("--attempts", type=int, default=3)
    code.add_argument("--int-ids", action="store_true", help="read numeric ids as integers")
    _model_options(code)
    code.set_defaults(func=command_code)

    score = commands.add_parser("score", help="levels and per-item scores of a saved run")
    score.add_argument("run")
    score.add_argument(
        "--referee", help="band[:ACCEPT_AT] or llm:PROVIDER:MODEL (default: the referee the run was made with)"
    )
    score.add_argument("--out", help="scores file (.csv or .json)")
    _model_options(score)
    score.set_defaults(func=command_score)

    cmp = commands.add_parser("compare", help="the comparison table across saved runs")
    cmp.add_argument("runs", nargs="+")
    cmp.add_argument("--seed", type=int, required=True)
    cmp.add_argument("--draws", type=int, default=2000)
    cmp.add_argument("--out", help="table file (.csv or .json)")
    cmp.set_defaults(func=command_compare)

    agree = commands.add_parser("agreement", help="cross- and within-coder disagreement of a saved run")
    agree.add_argument("run")
    agree.add_argument("--out", help="JSON file")
    agree.set_defaults(func=command_agreement)

    pick = commands.add_parser("pick", help="what a labelled sample of n items would pick")
    pick.add_argument("run")
    pick.add_argument("--n", type=int, required=True)
    pick.add_argument("--seed", type=int, required=True)
    pick.add_argument("--draws", type=int, default=500)
    pick.add_argument("--among", nargs="*", help="arms to pick among (default: the single coders)")
    pick.add_argument("--out", help="JSON file")
    pick.set_defaults(func=command_pick)

    tasks = commands.add_parser("tasks", help="list the study tasks, or write one as a JSON file to start from")
    tasks.add_argument("name", nargs="?")
    tasks.add_argument("--out", help="task JSON file")
    tasks.set_defaults(func=command_tasks)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit status: 0 for success, 2 for a usage error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result: int = args.func(args)
        return result
    except (CLIError, StoreError, ReplayError, RefereeError, ValueError, OSError) as exc:
        print(f"codercommittee: {exc}", file=sys.stderr)
        return 2


def entry() -> None:  # pragma: no cover - the console script
    sys.exit(main())


__all__ = [
    "CLIError",
    "build_parser",
    "load_task",
    "main",
    "model_settings",
    "parse_coder",
    "parse_embedder",
    "parse_model",
    "parse_referee",
]


if __name__ == "__main__":  # pragma: no cover
    entry()
