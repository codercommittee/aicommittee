"""Build the portable parity fixture from the study's own mock runs.

Run inside the study's virtual environment, from this repository::

    python parity/build_fixture.py --study /path/to/multi-model-llm-qualitative-coding

The script drives the study pipeline in mock mode over small item sets (three cycles, four
datasets, scripted relation and tuple answers, the pipeline's own mock answers for the label
kinds), lets the study's figure code derive every expected number, and writes
``tests/fixtures/mock_cycles.tar.gz``: the run directories (answers, feature-hashed vectors,
mock referee decisions, in the study's checkpoint layout), ``items.json`` and
``expected.json``. The library's parity test replays the stored evidence and must reproduce
the expected values; it never re-embeds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "mock_cycles.tar.gz"
DATASETS = ("causalnews", "miric", "gvfc", "politicause")
CYCLES = ("c1", "c2", "c3")
DRAWS = ("1", "2", "3")

# Relation items: text, two reference relations, and a pool of candidate answers: the two
# references A and B, boundary variants VA and VB, and a spurious relation C.
RELATION_ITEMS: list[dict[str, Any]] = [
    {
        "id": "r1",
        "text": "alpha beta gamma delta epsilon delta zeta alpha beta eta theta iota kappa",
        "A": ("alpha beta gamma", "delta epsilon"),
        "B": ("alpha beta eta", "delta zeta"),
        "VA": ("beta gamma", "delta epsilon"),
        "VB": ("alpha beta eta", "zeta"),
        "C": ("theta iota", "kappa"),
    },
    {
        "id": "r2",
        "text": "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega",
        "A": ("lambda mu", "nu xi omicron"),
        "B": ("rho sigma", "tau upsilon"),
        "VA": ("lambda mu nu", "xi omicron"),
        "VB": ("pi rho sigma", "tau upsilon"),
        "C": ("phi chi", "psi omega"),
    },
    {
        "id": "r3",
        "text": "a red kite rose over the harbour, blue boats drifted to the quay and the gulls scattered",
        "A": ("a red kite rose over the harbour", "blue boats drifted to the quay"),
        "B": ("blue boats drifted to the quay", "the gulls scattered"),
        "VA": ("a red kite rose", "blue boats drifted to the quay"),
        "VB": ("blue boats drifted", "the gulls scattered"),
        "C": ("the harbour", "the gulls"),
    },
    {
        "id": "r4",
        "text": "the mill closed in march, so forty workers left town and the bakery lost its trade",
        "A": ("the mill closed in march", "forty workers left town"),
        "B": ("forty workers left town", "the bakery lost its trade"),
        "VA": ("the mill closed", "forty workers left town"),
        "VB": ("forty workers left", "the bakery lost its trade"),
        "C": ("the bakery", "its trade"),
    },
    {
        "id": "r5",
        "text": "heavy rain filled the river, the bridge washed out, and the market moved uphill",
        "A": ("heavy rain filled the river", "the bridge washed out"),
        "B": ("the bridge washed out", "the market moved uphill"),
        "VA": ("heavy rain", "the bridge washed out"),
        "VB": ("the bridge washed out", "the market moved"),
        "C": ("the market", "uphill"),
    },
    {
        "id": "r6",
        "text": "frost hit the orchard early and the cider press stood idle until the late apples came in",
        "A": ("frost hit the orchard early", "the cider press stood idle"),
        "B": ("the late apples came in", "the cider press stood idle"),
        "VA": ("frost hit the orchard", "the cider press stood idle"),
        "VB": ("the late apples came", "the cider press stood idle"),
        "C": ("the orchard", "the late apples"),
    },
]

# Which pool entries each coder answers on each draw, per cycle. Odd items mirror A with B.
RELATION_PATTERNS: dict[str, dict[str, tuple[list[str], ...]]] = {
    "c1": {
        "anthropic": (["A"], ["A", "C"], ["VA"]),
        "openai": (["B"], ["B"], ["B", "C"]),
        "google": (["A", "B"], ["A", "B"], ["VA", "B"]),
    },
    "c2": {
        "anthropic": (["A"], ["A", "C"], ["VA"]),
        "openai": (["A", "B"], ["A", "B"], ["VA", "B"]),
        "google": (["B"], ["B"], ["B", "C"]),
    },
    "c3": {
        "anthropic": (["A", "B"], ["A"], ["A", "B", "C"]),
        "openai": (["B", "VA"], ["B"], ["A", "B"]),
        "google": (["VB"], ["B", "C"], ["A", "VB"]),
    },
}
MIRROR = {"A": "B", "B": "A", "VA": "VB", "VB": "VA", "C": "C"}

# Tuple items: text, the references of two coders (cause, effect, signal), and a pool: R1, R2,
# a boundary variant V, the non-causal answer N and, on non-causal items, a spurious S.
TUPLE_ITEMS: list[dict[str, Any]] = [
    {
        "id": 101,
        "text": "Because the council cut the bus subsidy, evening services stopped across the valley.",
        "refs": [
            ("the council cut the bus subsidy", "evening services stopped across the valley", "Because"),
            ("the council cut the bus subsidy", "evening services stopped", "Because"),
        ],
        "V": ("council cut the bus subsidy", "evening services stopped across the valley", ""),
    },
    {
        "id": 102,
        "text": "The new tariff raised import prices, which pushed the bakery to close its second shop.",
        "refs": [
            ("The new tariff raised import prices", "the bakery to close its second shop", "pushed"),
            ("The new tariff", "the bakery to close its second shop", "pushed"),
        ],
        "V": ("The new tariff raised import prices", "close its second shop", ""),
    },
    {
        "id": 103,
        "text": "Early warnings can prevent a harvest collapse when the frost arrives in April.",
        "refs": [
            ("Early warnings", "a harvest collapse", "prevent"),
            ("Early warnings", "a harvest collapse when the frost arrives in April", "prevent"),
        ],
        "V": ("warnings", "a harvest collapse", ""),
    },
    {
        "id": 104,
        "text": "We must widen the harbour so that the larger ferries can dock in winter.",
        "refs": [
            ("widen the harbour", "the larger ferries can dock in winter", "so that"),
            ("We must widen the harbour", "the larger ferries can dock", "so that"),
        ],
        "V": ("widen the harbour", "the larger ferries can dock", ""),
    },
    {"id": 105, "text": "The council met on Tuesday and published its minutes the next morning.", "refs": []},
    {"id": 106, "text": "The report lists every bridge in the county with its year of construction.", "refs": []},
]
TUPLE_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "c1": {"anthropic": ("R1", "R1", "V"), "openai": ("R2", "N", "R2"), "google": ("R1", "R2", "R1")},
    "c2": {"anthropic": ("R1", "R1", "V"), "openai": ("R1", "R2", "R1"), "google": ("R2", "N", "R2")},
    "c3": {"anthropic": ("R2", "V", "R1"), "openai": ("N", "R2", "R2"), "google": ("R1", "R1", "V")},
}
TUPLE_PATTERNS_NEGATIVE: dict[str, dict[str, tuple[str, ...]]] = {
    "c1": {"anthropic": ("N", "N", "S"), "openai": ("N", "N", "N"), "google": ("S", "N", "N")},
    "c2": {"anthropic": ("N", "S", "N"), "openai": ("N", "N", "N"), "google": ("N", "N", "S")},
    "c3": {"anthropic": ("N", "N", "N"), "openai": ("S", "N", "N"), "google": ("N", "N", "N")},
}

MIRIC_TEXTS = [
    "A method that trains a neural network to predict machine failures from sensor data.",
    "A window latch with a spring-loaded bolt and a weather seal.",
    "A recommendation engine that learns user preferences from browsing behaviour.",
    "A bicycle frame made of welded aluminium tubes.",
    "A speech recognition module that adapts its acoustic model to the speaker.",
    "A garden hose reel with an automatic retraction mechanism.",
    "An image classifier that detects defects on a production line.",
    "A folding chair with a locking hinge for outdoor use.",
    "A fraud detection system fitted to historical transaction records.",
    "A kitchen timer with a rotating dial and a bell.",
    "A traffic forecast computed from historical flow data at each junction.",
    "A ceramic tile with a slip-resistant glaze.",
    "A chatbot that classifies customer messages by intent.",
    "A door hinge with a self-closing damper.",
    "A drone that learns a landing policy by reinforcement.",
    "A coffee grinder with adjustable burrs.",
    "A greenhouse controller that fits a moisture model to its sensors.",
    "A tent peg made of recycled plastic.",
    "A text summariser trained on parliamentary records.",
    "A staircase handrail with an integrated light strip.",
    "A medical scan reader that scores images for tumour likelihood.",
    "A lunch box with separate compartments and a cooling pack.",
    "A loan scoring model that updates with repayment histories.",
    "A hammock with a steel stand.",
]
GVFC_HEADLINES = [
    ("Senate Republicans block background check bill after shooting", ["politics", "gun_control"]),
    ("Students walk out of class to demand safer schools", ["school_public_space_safety", "public_opinion"]),
    ("NRA says the Second Amendment is not up for debate", ["gun_rights"]),
    ("Poll finds most voters back stricter gun laws", ["public_opinion", "gun_control"]),
    ("Governor signs bill raising the age to buy rifles", ["gun_control"]),
    ("Shooting renews debate over mental health funding", ["mental_health"]),
    ("Black communities bear the brunt of gun deaths, report says", ["race_ethnicity"]),
    ("Gun sales surge after election night", ["economic_consequences"]),
    ("Hollywood faces questions over gun violence on screen", ["society_culture"]),
    ("Mayor orders metal detectors at city parks", ["school_public_space_safety"]),
    ("Democrats and Republicans trade blame after the rally shooting", ["politics"]),
    ("Gun owners rally at the capitol to defend their rights", ["gun_rights", "public_opinion"]),
    ("Insurers raise premiums for schools after string of shootings", ["economic_consequences"]),
    ("Church shooting leaves town questioning its gun culture", ["society_culture"]),
    ("Candidate makes gun control the centre of her campaign", ["politics", "gun_control"]),
    ("Study links shootings to untreated mental illness", ["mental_health"]),
    ("Latino groups call for action on gun deaths", ["race_ethnicity", "public_opinion"]),
    ("Court upholds right to carry handguns in public", ["gun_rights"]),
    ("Teachers union opposes arming staff", ["school_public_space_safety", "politics"]),
    ("Retailer stops selling assault-style rifles", ["economic_consequences", "gun_control"]),
]


def relation(cause: str, effect: str) -> dict[str, str]:
    return {"cause": cause, "effect": effect, "signal": ""}


def items() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {
        "causalnews": [
            {
                "id": item["id"],
                "text": item["text"],
                "gold_label": [{**relation(*item["A"]), "signals": []}, {**relation(*item["B"]), "signals": []}],
                "metadata": {"num_relations": 2},
            }
            for item in RELATION_ITEMS
        ],
        "miric": [
            {"id": f"p{index:02d}", "text": text, "gold_label": 1 if index % 3 == 0 else 0, "metadata": {}}
            for index, text in enumerate(MIRIC_TEXTS)
        ],
        "gvfc": [
            {"id": 1000 + index, "text": text, "gold_label": frames, "metadata": {"focus": 1 + index % 2}}
            for index, (text, frames) in enumerate(GVFC_HEADLINES)
        ],
        "politicause": [
            {
                "id": item["id"],
                "text": item["text"],
                "gold_label": {
                    "causal": bool(item["refs"]),
                    "references": [
                        {"coder_id": f"c{n + 1}", "cause": cause, "effect": effect, "signal": signal}
                        for n, (cause, effect, signal) in enumerate(item["refs"])
                    ],
                },
                "metadata": {"frac_accept": 1.0 if item["refs"] else 0.0},
            }
            for item in TUPLE_ITEMS
        ],
    }
    return out


def relation_answer(cycle: str, index: int, model: str, draw: str) -> dict[str, Any]:
    item = RELATION_ITEMS[index]
    keys = RELATION_PATTERNS[cycle][model][int(draw) - 1]
    if index % 2 == 1:
        keys = [MIRROR[key] for key in keys]
    return {"relations": [relation(*item[key]) for key in keys]}


def tuple_answer(cycle: str, index: int, model: str, draw: str) -> dict[str, Any]:
    item = TUPLE_ITEMS[index]
    negative = {"causal": False, "cause": "", "effect": "", "signal": ""}
    if item["refs"]:
        key = TUPLE_PATTERNS[cycle][model][int(draw) - 1]
        if key == "N":
            return negative
        cause, effect, signal = item["refs"][0] if key == "R1" else item["refs"][1] if key == "R2" else item["V"]
        return {"causal": True, "cause": cause, "effect": effect, "signal": signal}
    key = TUPLE_PATTERNS_NEGATIVE[cycle][model][int(draw) - 1]
    if key == "N":
        return negative
    words = item["text"].rstrip(".").split()
    return {"causal": True, "cause": " ".join(words[:3]), "effect": " ".join(words[-3:]), "signal": ""}


def build(study: Path) -> Path:
    sys.path.insert(0, str(study / "code"))
    from extension import construction_study as cs
    from figures import derive, load, panels, pool_analysis
    from pipeline import models, run_study
    from pipeline.models import ModelResponse, request_envelope, request_fingerprint

    mock_items = items()
    relation_index = {item["text"]: index for index, item in enumerate(RELATION_ITEMS)}
    tuple_index = {item["text"]: index for index, item in enumerate(TUPLE_ITEMS)}
    tmp = Path(tempfile.mkdtemp(prefix="codercommittee_fixture_"))
    run_study.ROOT = tmp
    run_study.load_items = lambda dataset, *args, **kwargs: [dict(item) for item in mock_items[dataset]]
    run_study.load_prompt = lambda name: "{{TEXT}}"
    run_study.validate_dataset_inventory = lambda *args, **kwargs: []
    original = run_study.call_model
    state = {"cycle": ""}

    def scripted(
        spec: Any,
        prompt: str,
        system: str = run_study.SYSTEM_PROMPT,
        mode: str = "mock",
        *,
        response_schema: Any = None,
        cache_namespace: str = models.DEFAULT_CACHE_NAMESPACE,
        max_output_tokens: int = models.MAX_OUTPUT_TOKENS,
        repeat_index: int = 1,
        before_attempt: Any = None,
        after_attempt: Any = None,
        max_attempts: int = models.MAX_TRANSPORT_ATTEMPTS,
    ) -> Any:
        properties = (response_schema or {}).get("properties", {})
        answer: dict[str, Any] | None = None
        if spec.model_id != "alignment_agent":
            if "relations" in properties and prompt in relation_index:
                answer = relation_answer(state["cycle"], relation_index[prompt], spec.model_id, str(repeat_index))
            elif {"causal", "cause", "effect"} <= set(properties) and prompt in tuple_index:
                answer = tuple_answer(state["cycle"], tuple_index[prompt], spec.model_id, str(repeat_index))
        if answer is None:
            return original(
                spec,
                prompt,
                system,
                mode,
                response_schema=response_schema,
                cache_namespace=cache_namespace,
                max_output_tokens=max_output_tokens,
                repeat_index=repeat_index,
                before_attempt=before_attempt,
                after_attempt=after_attempt,
                max_attempts=max_attempts,
            )
        envelope = request_envelope(
            spec,
            prompt,
            system,
            response_schema,
            "mock",
            cache_namespace,
            max_output_tokens=max_output_tokens,
            repeat_index=repeat_index,
        )
        raw = json.dumps(answer, ensure_ascii=False)
        response = ModelResponse(
            model_id=spec.model_id,
            family=spec.family,
            raw_text=raw,
            parsed=json.loads(raw),
            latency_ms=1.0,
            requested_model_id=spec.provider_model,
            returned_model_id=spec.provider_model,
            attempts=1,
            cache_namespace=cache_namespace,
            tokens_in=10,
            tokens_out=10,
        )
        response.request_fingerprint = request_fingerprint(envelope)
        return response

    run_study.call_model = scripted
    cycles: list[dict[str, Any]] = []
    dirs_by_cycle: dict[str, dict[str, Path]] = {}
    for cycle in CYCLES:
        state["cycle"] = cycle
        dirs = {d: tmp / run_study.run(d, mode="mock", run_id=cycle)["run_directory"] for d in DATASETS}
        cs.run(
            Namespace(
                command="run", runs=[str(dirs["causalnews"])], budget=None, run_id=f"cs_{cycle}", mode="mock", workers=1
            )
        )
        extension = tmp / "results" / "extension" / f"cs_{cycle}"
        cycles.append(
            {
                "id": cycle,
                "runs": {d: load.load_run(dirs[d], allow_mock=True) for d in DATASETS},
                "extension": load.load_extension(extension, allow_mock=True),
                "bases": {"causalnews": load.base_run(dirs["causalnews"])},
            }
        )
        dirs_by_cycle[cycle] = dirs

    derived = derive.derive(cycles)
    spec = cs.load_spec()
    threshold = float(spec["scoring"]["primary_match_threshold"])
    expected: dict[str, Any] = {"cross": derived["cross"], "cycles": {}, "composition": {}}
    expected["match_threshold"] = threshold
    for spec_of_cycle in cycles:
        block, memory = derive.derive_cycle(
            spec_of_cycle["id"], spec_of_cycle["runs"], spec_of_cycle["extension"], spec_of_cycle["bases"]
        )
        per_dataset: dict[str, Any] = {}
        for d in DATASETS:
            entry = block["datasets"][d]
            record: dict[str, Any] = {
                "arms": entry["arms"],
                "best_single": entry["best_single"],
                "n_complete": entry["n_complete"],
                "disagreement": entry["disagreement"],
                "pool": entry.get("pool"),
                "item_ids": memory[d]["items"]["item_ids"],
                "series": memory[d]["items"]["series"],
                "arm_contrasts": entry["arm_contrasts"],
            }
            if d == "causalnews":
                base = memory[d]["base"]
                flat = memory[d]["flat"]
                pool: dict[str, dict[str, Any]] = {}
                for position, item in enumerate(base["items"]):
                    members = {f"{m}:{r}": item["prepared"][m][r] for m in base["models"] for r in DRAWS}
                    pool[str(item["item_id"])] = {
                        str(k): {
                            "score": flat[k]["score"][position],
                            "prediction": pool_analysis._flat_pool(members, item["sim"], spec, k),
                        }
                        for k in range(1, 10)
                    }
                record["pool_items"] = pool
            per_dataset[d] = record
        expected["cycles"][spec_of_cycle["id"]] = per_dataset
    for d in DATASETS:
        entry = derived["cross"][d]
        expected["composition"][d] = {**panels._composition(entry), "best_level": panels._best_level(entry)}

    stage = Path(tempfile.mkdtemp(prefix="codercommittee_stage_")) / "mock_cycles"
    for cycle, dirs in dirs_by_cycle.items():
        for d, source in dirs.items():
            target = stage / cycle / d
            target.mkdir(parents=True)
            for name in ("per_item.jsonl", "summary.json", "canary_pin.json"):
                if (source / name).is_file():
                    shutil.copy2(source / name, target / name)
            for folder in ("responses", "embeddings", "agent"):
                if (source / "checkpoints" / folder).is_dir():
                    shutil.copytree(source / "checkpoints" / folder, target / "checkpoints" / folder)
    (stage / "items.json").write_text(json.dumps(mock_items, ensure_ascii=False, indent=1), encoding="utf-8")
    (stage / "expected.json").write_text(json.dumps(expected, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(stage, arcname="mock_cycles")
    shutil.rmtree(stage.parent)
    shutil.rmtree(tmp)
    return OUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study", required=True, type=Path, help="path to the study repository checkout")
    args = parser.parse_args()
    out = build(args.study.resolve())
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
