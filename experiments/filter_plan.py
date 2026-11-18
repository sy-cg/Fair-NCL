import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def filter_jobs(
    jobs: Iterable[Dict[str, Any]],
    *,
    methods: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    backbones: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    allowed_methods = set(methods or [])
    allowed_datasets = set(datasets or [])
    allowed_backbones = set(backbones or [])
    allowed_seeds = {int(seed) for seed in seeds or []}
    return [
        job
        for job in jobs
        if (not allowed_methods or job["method"] in allowed_methods)
        and (not allowed_datasets or job["dataset"] in allowed_datasets)
        and (not allowed_backbones or job["backbone"] in allowed_backbones)
        and (not allowed_seeds or int(job["seed"]) in allowed_seeds)
    ]


def load_plan(path: str) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_plan(jobs: Iterable[Dict[str, Any]], path: str) -> int:
    rows = list(jobs)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in rows),
        encoding="utf-8",
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a JSONL experiment plan without changing its jobs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--backbones", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    jobs = filter_jobs(
        load_plan(args.input),
        methods=args.methods,
        datasets=args.datasets,
        backbones=args.backbones,
        seeds=args.seeds,
    )
    count = save_plan(jobs, args.output)
    print(f"Wrote {count} jobs to {args.output}.")


if __name__ == "__main__":
    main()
