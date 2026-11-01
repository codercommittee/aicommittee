"""Export one study dataset's items to the library's items file, through the study's own loaders.

Run inside the study's virtual environment, with the dataset fetched and its path variables set
as the study's ``code/datasets/DATASET_FETCH.md`` and ``.env.example`` describe::

    python parity/export_items.py --study /path/to/multi-model-llm-qualitative-coding --dataset causalnews \\
        --out causalnews.jsonl

The file holds the same items, texts and references the study coded, in the study's order, and
is what ``codercommittee code --task study:causalnews --items causalnews.jsonl ...`` takes. The
datasets' licences stay with their publishers: the export is for a machine that holds them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from codercommittee.io import write_items  # noqa: E402
from codercommittee.study import DATASETS, TASKS  # noqa: E402
from full_cycles import study_items  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study", required=True, type=Path, help="path to the study repository checkout")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--out", required=True, type=Path, help="items file to write (.jsonl or .csv)")
    args = parser.parse_args()
    items = study_items(args.study.resolve(), args.dataset)
    write_items(items, args.out, TASKS[args.dataset])
    with_reference = sum(item.reference is not None for item in items)
    print(f"wrote {len(items)} items ({with_reference} with a reference) to {args.out}")


if __name__ == "__main__":
    main()
