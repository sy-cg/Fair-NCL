import argparse
import json
from typing import Any, Dict, List

from .job_utils import result_artifact_is_complete


def summarize_plan(plan_path: str, results_root: str) -> Dict[str, Any]:
    jobs = _load_plan(plan_path)
    done_count = 0
    todo_indices = []
    for index, job in enumerate(jobs):
        if _job_result_exists(job, results_root):
            done_count += 1
        else:
            todo_indices.append(index)
    return {
        "total": len(jobs),
        "done": done_count,
        "todo_indices": todo_indices,
        "phase": jobs[0].get("phase") if jobs else None,
    }


def is_job_done(plan_path: str, results_root: str, job_index: int) -> bool:
    for index, job in enumerate(_iter_plan(plan_path)):
        if index == job_index:
            return _job_result_exists(job, results_root)
    raise IndexError(f"Job index out of range: {job_index}")


def _load_plan(plan_path: str) -> List[Dict[str, Any]]:
    return list(_iter_plan(plan_path))


def _iter_plan(plan_path: str):
    with open(plan_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _job_result_exists(job: Dict[str, Any], results_root: str) -> bool:
    return result_artifact_is_complete(job, results_root)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Inspect JSONL experiment plan completion status.")
    parser.add_argument("command", choices=["summary", "field", "todo-indices", "job-done"])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--field", choices=["total", "done", "phase"], default=None)
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args(argv)

    if args.command == "summary":
        print(json.dumps(summarize_plan(args.plan, args.results_root)))
    elif args.command == "field":
        if args.field is None:
            parser.error("--field is required for field")
        value = summarize_plan(args.plan, args.results_root)[args.field]
        print("" if value is None else value)
    elif args.command == "todo-indices":
        for index in summarize_plan(args.plan, args.results_root)["todo_indices"]:
            print(index)
    elif args.command == "job-done":
        if args.index is None:
            parser.error("--index is required for job-done")
        print("1" if is_job_done(args.plan, args.results_root, args.index) else "0")


if __name__ == "__main__":
    main()
