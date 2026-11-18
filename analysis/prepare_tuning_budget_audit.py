"""Audit initial tuning runs and export the backbone validation trace."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ml-1m", "taobao")
BACKBONES = ("sasrec", "bert4rec", "gru4rec", "caser")
EXPECTED = {"backbone": 8, "augment": 18, "loss": 12}
AUDIT_METHODS = {"fair_ncl", "adv_debias", "grl", "sm_pcfr", "a_fsr", "afrl", "pfrec"}


def export_rows(source: Path, destination: Path, expected: int, drop: tuple[str, ...] = ()) -> None:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [field for field in reader.fieldnames if field not in drop]
        rows = [{field: row[field] for field in fields} for row in reader
                if row["method"] in AUDIT_METHODS]
    assert len(rows) == expected, (source, len(rows))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    observed: Counter[tuple[str, str, str]] = Counter()
    completed: Counter[tuple[str, str, str]] = Counter()
    test_outputs: Counter[tuple[str, str, str]] = Counter()
    backbone_runs: dict[tuple[str, str], list[dict]] = {}
    for path in (ROOT / "results").rglob("job.json"):
        job = json.loads(path.read_text(encoding="utf-8"))
        phase, dataset, backbone = job.get("phase"), job.get("dataset"), job.get("backbone")
        if phase not in EXPECTED or dataset not in DATASETS or backbone not in BACKBONES:
            continue
        assert job.get("seed") == 42, path
        key = dataset, backbone, phase
        observed[key] += 1
        if path.with_name("history.json").exists() and path.with_name("best.pt").exists():
            completed[key] += 1
        if path.with_name("test_results.json").exists():
            test_outputs[key] += 1
        if phase == "backbone":
            history = json.loads(path.with_name("history.json").read_text(encoding="utf-8"))
            backbone_runs.setdefault((dataset, backbone), []).append({
                "params": {name: job["params"][name]
                           for name in ("learning_rate", "hidden_units", "dropout_rate")},
                "max_validation_hr10": max(history["val_HitRate@10"]),
            })

    rows = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for phase, expected in EXPECTED.items():
                key = dataset, backbone, phase
                assert observed[key] == completed[key] == expected, (key, observed[key], completed[key])
                rows.append({"dataset": dataset, "backbone": backbone, "phase": phase,
                             "seed": 42, "jobs": observed[key], "completed": completed[key],
                             "test_outputs": test_outputs[key]})
    output = ROOT / "tables/supplementary/reviewer_protocol_cost/initial_search_audit.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    supplement = ROOT / "els-cas-templates/source_data/table_source_data"
    supplement.joinpath("Supplementary_Data_S4_initial_search.csv").write_bytes(output.read_bytes())
    fixed = json.loads((ROOT / "configs/backbone_best.json").read_text(encoding="utf-8"))[
        "__dataset_backbones__"]
    backbone_rows = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            trials = backbone_runs[(dataset, backbone)]
            assert len(trials) == 8
            chosen = fixed[dataset][backbone]
            assert sum(trial["params"] == chosen for trial in trials) == 1
            for trial in trials:
                backbone_rows.append({
                    "dataset": dataset, "backbone": backbone, **trial["params"],
                    "max_validation_hr10": trial["max_validation_hr10"],
                    "validation_hr_rank": 1 + sum(
                        other["max_validation_hr10"] > trial["max_validation_hr10"]
                        for other in trials),
                    "fixed_backbone_config": trial["params"] == chosen,
                })
    trace = ROOT / "tables/supplementary/reviewer_protocol_cost/backbone_search_trace.csv"
    with trace.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(backbone_rows[0]))
        writer.writeheader()
        writer.writerows(backbone_rows)
    supplement.joinpath("Supplementary_Data_S4_backbone_search.csv").write_bytes(trace.read_bytes())
    export_rows(ROOT / "tables/supplementary/validation_candidates.csv",
                supplement / "Supplementary_Data_S4_candidates.csv", 280)
    export_rows(ROOT / "tables/supplementary/validation_selection.csv",
                supplement / "Supplementary_Data_S4_selected.csv", 56, ("all_candidates",))
    print(f"Wrote {len(rows)} completed phase counts to {output}")


if __name__ == "__main__":
    main()
