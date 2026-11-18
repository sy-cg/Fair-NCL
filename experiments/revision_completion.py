"""Generate the missing, paired revision experiments from completed Fair-NCL jobs.

Run with: python -m experiments.revision_completion plan --results-root results_revision_kbs_v2
After tuning: python -m experiments.revision_completion finalize --results-root results_revision_kbs_v2
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from .job_utils import result_artifact_is_complete, stable_job_id
from .specs import ExperimentJob, save_plan

SEEDS = (42, 2025, 2026, 2027, 2028)
TUNE_SEEDS = SEEDS[:3]
CO_WEIGHTS = (0.03, 0.1, 0.3)
LEAN_SENSITIVITY_AXES = ("epsilon", "augment_ratio", "utility_beta")
REVISION_DATASETS = ("ml-1m", "taobao")
REVISION_BACKBONES = ("sasrec", "bert4rec", "gru4rec", "caser")


def _fair_jobs(results_root, *, required_backbones=None, reference_plan=None):
    backbones = tuple(required_backbones or REVISION_BACKBONES)
    expected_pairs = {
        (dataset, backbone)
        for dataset in REVISION_DATASETS
        for backbone in backbones
    }
    found = {}
    if reference_plan is not None:
        reference_path = Path(reference_plan)
        sources = [
            (json.loads(line), reference_path)
            for line in reference_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        sources = [
            (json.loads(path.read_text(encoding="utf-8")), path)
            for path in sorted(Path(results_root).glob("*/supplementary_*fair_ncl_*/job.json"))
        ]
    for job, source_path in sources:
        if job.get("phase") != "supplementary" or job.get("method") != "fair_ncl":
            continue
        key = (job["dataset"], job["backbone"], int(job["seed"]))
        if key[:2] not in expected_pairs:
            continue
        if key in found:
            raise ValueError(f"Duplicate Fair-NCL job for {key}")
        if reference_plan is None and not result_artifact_is_complete(job, results_root):
            raise ValueError(f"Fair-NCL result incomplete: {source_path.parent}")
        params = job["params"]
        if params.get("bias_metric") != "prior_jsd" or params.get("bias_event_source") != "unique_train_events":
            raise ValueError(f"Old skew definition in {source_path}")
        found[key] = job
    pairs = {(dataset, backbone) for dataset, backbone, _ in found}
    for dataset, backbone in expected_pairs:
        missing = set(SEEDS) - {seed for ds, bb, seed in found if (ds, bb) == (dataset, backbone)}
        if missing:
            raise ValueError(f"Missing Fair-NCL seeds for {(dataset, backbone)}: {sorted(missing)}")
    if pairs != expected_pairs:
        raise ValueError(f"Expected Fair-NCL pairs {sorted(expected_pairs)}, found {sorted(pairs)}")
    return found


def _job(phase, dataset, backbone, seed, method, params):
    return ExperimentJob(
        job_id=stable_job_id(phase, dataset, method, backbone, seed, params),
        phase=phase, dataset=dataset, method=method, backbone=backbone,
        seed=seed, params=params,
    )


def factorial_jobs(fair):
    jobs = []
    for (dataset, backbone, seed), source in sorted(fair.items()):
        for bits in range(15):
            # bit 0: similarity term; 1: skew term; 2: augmented CE; 3: geometry.
            flags = tuple(bool(bits & (1 << i)) for i in range(4))
            params = dict(source["params"])
            if not flags[0]:
                params["utility_alpha"] = 0.0
            if not flags[1]:
                params["utility_beta"] = 0.0
            if not flags[2]:
                params["fair_ncl_aug_rec_weight"] = 0.0
            if not flags[3]:
                for key in ("fair_ncl_align_weight", "fair_ncl_var_weight", "fair_ncl_cov_weight"):
                    params[key] = 0.0
            params["factorial_cell"] = "".join(str(int(flag)) for flag in flags)
            params["export_mechanism_analysis"] = False
            jobs.append(_job("revision_factorial", dataset, backbone, seed,
                             "fair_ncl_factorial", params))
    return jobs


def random_pool_jobs(fair):
    jobs = []
    for (dataset, backbone, seed), source in sorted(fair.items()):
        params = dict(source["params"])
        params["export_mechanism_analysis"] = False
        jobs.append(_job("revision_random_pool", dataset, backbone, seed,
                         "random_aug", params))
    return jobs


def sensitivity_jobs(fair, *, seeds=SEEDS, axes=None):
    jobs = []
    selected_seeds = {int(seed) for seed in seeds}
    selected_axes = set(axes) if axes is not None else None
    for (dataset, backbone, seed), source in sorted(fair.items()):
        if backbone != "sasrec" or seed not in selected_seeds:
            continue
        base = source["params"]
        beta = float(base["utility_beta"])
        axis_values = {
            "epsilon": [0.5 * float(base["epsilon"]), 2.0 * float(base["epsilon"])],
            "augment_ratio": [0.5 * float(base["augment_ratio"]), 2.0 * float(base["augment_ratio"])],
            "utility_beta": [max(0.0, beta - 0.2), min(1.0, beta + 0.2)],
            "fair_ncl_align_weight": [0.5 * float(base["fair_ncl_align_weight"]),
                                      2.0 * float(base["fair_ncl_align_weight"])],
            "fair_ncl_var_weight": [0.5 * float(base["fair_ncl_var_weight"]),
                                    2.0 * float(base["fair_ncl_var_weight"])],
            "fair_ncl_cov_weight": [0.5 * float(base["fair_ncl_cov_weight"]),
                                    2.0 * float(base["fair_ncl_cov_weight"])],
        }
        for axis, values in axis_values.items():
            if selected_axes is not None and axis not in selected_axes:
                continue
            for value in values:
                params = dict(base)
                params[axis] = round(value, 8)
                if axis == "utility_beta":
                    params["utility_alpha"] = round(1.0 - value, 8)
                params["sensitivity_axis"] = axis
                params["export_mechanism_analysis"] = False
                jobs.append(_job("revision_sensitivity", dataset, backbone, seed,
                                 "fair_ncl", params))
    return jobs


def _existing_mce_weights(results_root, dataset, backbone):
    for method in ("fair_aug_mce", "ncl_sr_mce"):
        paths = sorted((Path(results_root) / dataset).glob(
            f"supplementary_{dataset}_{method}_{backbone}_s2025_*/job.json"
        ))
        if paths:
            params = json.loads(paths[0].read_text(encoding="utf-8"))["params"]
            return (float(params.get("mce_uniform_weight", 1.0)),
                    float(params.get("mce_align_weight", 1.0)))
    return 1.0, 1.0


def mce_candidates(results_root, dataset, backbone):
    uniform, align = _existing_mce_weights(results_root, dataset, backbone)
    return tuple(dict(mce_uniform_weight=round(u, 8), mce_align_weight=round(a, 8))
                 for u, a in ((uniform, align), (uniform / 2, align),
                              (uniform * 2, align), (uniform, align / 2),
                              (uniform, align * 2)))


def mce_tune_jobs(fair, results_root):
    jobs = []
    for (dataset, backbone, seed), source in sorted(fair.items()):
        if seed not in TUNE_SEEDS:
            continue
        for index, candidate in enumerate(mce_candidates(results_root, dataset, backbone)):
            params = dict(source["params"])
            params.update(candidate)
            params.update(ncl_objective="mce", evaluate_test=False,
                          export_per_user_metrics=False, export_mechanism_analysis=False,
                          tune_candidate=f"mce_{index}")
            jobs.append(_job("revision_mce_tune", dataset, backbone, seed,
                             "fair_aug_mce", params))
    return jobs


def coserec_tune_jobs(fair):
    jobs = []
    for (dataset, backbone, seed), source in sorted(fair.items()):
        if backbone != "sasrec" or seed not in TUNE_SEEDS:
            continue
        for weight in CO_WEIGHTS:
            params = _coserec_params(source["params"], weight)
            params.update(evaluate_test=False, export_per_user_metrics=False,
                          export_mechanism_analysis=False,
                          tune_candidate=f"cf_{weight:g}")
            jobs.append(_job("revision_coserec_tune", dataset, backbone, seed,
                             "coserec", params))
    return jobs


def _coserec_params(fair_params, weight):
    return {
        key: fair_params[key] for key in ("learning_rate", "hidden_units", "dropout_rate")
    } | {
        "coserec_cf_weight": weight,
        "train_negative_history_only": True,
        "coserec_crop_rate": 0.2,
        "coserec_mask_rate": 0.7,
        "coserec_reorder_rate": 0.2,
        "coserec_insert_rate": 0.4,
        "coserec_max_insert_num_per_pos": 1,
        "coserec_substitute_rate": 0.1,
        "coserec_short_threshold": 4,
        "coserec_temperature": 1.0,
        "bias_metric": "prior_jsd",
        "bias_event_source": "unique_train_events",
        "report_intersectional_groups": True,
    }


def _select_validation(tune_jobs, results_root):
    grouped = defaultdict(list)
    expected_pairs = {(job.dataset, job.backbone) for job in tune_jobs}
    for job in tune_jobs:
        run_dir = Path(results_root) / job.dataset / job.job_id
        result_path = run_dir / "validation_results.json"
        marker_path = run_dir / "test_results.json"
        history_path = run_dir / "history.json"
        checkpoint_path = run_dir / "best.pt"
        for required in (result_path, marker_path, history_path, checkpoint_path):
            if not required.exists():
                raise FileNotFoundError(f"Tuning must finish before finalizing: {required}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "not_run_by_protocol" or "selected" in marker:
            raise ValueError(f"Tuning result violates validation-only protocol: {marker_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        metric = float(result["utility"]["HitRate@10"])
        if not math.isfinite(metric):
            continue
        skipped = sum(int(value) for value in history.get("skipped_nonfinite_updates", []))
        train_metrics = [
            values for key, values in history.items()
            if key.startswith("train_") and isinstance(values, list) and values
        ]
        if skipped > 0 and not train_metrics:
            continue
        key = (job.dataset, job.backbone, job.params["tune_candidate"])
        grouped[key].append(metric)
    selected = {}
    for pair in expected_pairs:
        scores = {
            candidate: sum(values) / len(values)
            for (dataset, backbone, candidate), values in grouped.items()
            if (dataset, backbone) == pair and len(values) == len(TUNE_SEEDS)
        }
        selected[pair] = (
            max(scores, key=lambda candidate: (scores[candidate], candidate))
            if scores else None
        )
    return selected


def finalize_jobs(fair, results_root):
    mce_tune = mce_tune_jobs(fair, results_root)
    mce_selection = _select_validation(mce_tune, results_root)
    mce_final = []
    for (dataset, backbone, seed), source in sorted(fair.items()):
        candidates = {f"mce_{index}": candidate for index, candidate in enumerate(
            mce_candidates(results_root, dataset, backbone))}
        selected_mce = mce_selection[(dataset, backbone)]
        if selected_mce is not None:
            params = dict(source["params"])
            params.update(candidates[selected_mce])
            params.update(ncl_objective="mce", export_mechanism_analysis=False)
            mce_final.append(_job("revision_mce_final", dataset, backbone, seed,
                                  "fair_aug_mce", params))
    return {
        "mce_final": mce_final,
        "coserec_final": finalize_coserec_jobs(fair, results_root),
    }


def finalize_coserec_jobs(fair, results_root):
    tune_jobs = coserec_tune_jobs(fair)
    selected = _select_validation(tune_jobs, results_root)
    jobs = []
    for (dataset, backbone, seed), source in sorted(fair.items()):
        if backbone != "sasrec":
            continue
        candidate = selected[(dataset, backbone)]
        if candidate is None:
            continue
        weight = float(candidate.removeprefix("cf_"))
        params = _coserec_params(source["params"], weight)
        params.update(evaluate_test=True, export_per_user_metrics=True,
                      export_mechanism_analysis=False)
        jobs.append(_job("revision_coserec_final", dataset, backbone, seed,
                         "coserec", params))
    return jobs


def coserec_selection_manifest(fair, results_root, final_jobs):
    tune_jobs = coserec_tune_jobs(fair)
    selected = _select_validation(tune_jobs, results_root)
    params_by_candidate = {
        (job.dataset, job.backbone, job.params["tune_candidate"]): job.params
        for job in tune_jobs
    }
    rows = []
    for (dataset, backbone), candidate in sorted(selected.items()):
        row = {
            "dataset": dataset,
            "backbone": backbone,
            "status": "selected" if candidate is not None else "excluded",
            "selected_candidate": candidate,
        }
        if candidate is None:
            row["reason"] = (
                "no candidate had a finite validation HitRate@10 and usable "
                "training updates for all three tuning seeds"
            )
        else:
            params = params_by_candidate[(dataset, backbone, candidate)]
            row["selected_params"] = {
                "coserec_cf_weight": params["coserec_cf_weight"],
            }
        rows.append(row)
    expected = sorted({
        (dataset, backbone) for dataset, backbone, _ in fair if backbone == "sasrec"
    })
    completed = sorted({(job.dataset, job.backbone) for job in final_jobs})
    return {
        "coserec_final_pairs": completed,
        "excluded_coserec_pairs": [pair for pair in expected if pair not in completed],
        "validation_selection": {"coserec": rows},
    }


def selection_manifest(fair, results_root, plans):
    """Describe validation choices and exclusions without consulting test metrics."""
    mce_tune = mce_tune_jobs(fair, results_root)
    coserec_tune = coserec_tune_jobs(fair)
    selections = {
        "mce": (_select_validation(mce_tune, results_root), mce_tune),
        "coserec": (_select_validation(coserec_tune, results_root), coserec_tune),
    }

    rows = {}
    for name, (selected, tune_jobs) in selections.items():
        params_by_candidate = {
            (job.dataset, job.backbone, job.params["tune_candidate"]): job.params
            for job in tune_jobs
        }
        method_rows = []
        for (dataset, backbone), candidate in sorted(selected.items()):
            row = {
                "dataset": dataset,
                "backbone": backbone,
                "status": "selected" if candidate is not None else "excluded",
                "selected_candidate": candidate,
            }
            if candidate is None:
                row["reason"] = (
                    "no candidate had a finite validation HitRate@10 and usable "
                    "training updates for all three tuning seeds"
                )
            else:
                params = params_by_candidate[(dataset, backbone, candidate)]
                if name == "mce":
                    row["selected_params"] = {
                        "mce_uniform_weight": params["mce_uniform_weight"],
                        "mce_align_weight": params["mce_align_weight"],
                    }
                else:
                    row["selected_params"] = {
                        "coserec_cf_weight": params["coserec_cf_weight"],
                    }
            method_rows.append(row)
        rows[name] = method_rows

    selected_pairs = {
        "mce_final_pairs": sorted({(job.dataset, job.backbone) for job in plans["mce_final"]}),
        "coserec_final_pairs": sorted({(job.dataset, job.backbone) for job in plans["coserec_final"]}),
    }
    expected_mce = sorted({(dataset, backbone) for dataset, backbone, _ in fair})
    expected_coserec = sorted({
        (dataset, backbone) for dataset, backbone, _ in fair if backbone == "sasrec"
    })
    selected_pairs["excluded_mce_pairs"] = [
        pair for pair in expected_mce if pair not in selected_pairs["mce_final_pairs"]
    ]
    selected_pairs["excluded_coserec_pairs"] = [
        pair for pair in expected_coserec if pair not in selected_pairs["coserec_final_pairs"]
    ]
    selected_pairs["validation_selection"] = rows
    return selected_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "finalize"))
    parser.add_argument("--results-root", default="results_revision_kbs_v2")
    parser.add_argument("--output-dir", default="experiments/revision_completion_plans")
    parser.add_argument("--profile", choices=("full", "lean"), default="full")
    parser.add_argument(
        "--reference-plan",
        default=None,
        help="Optional JSONL source for completed Fair-NCL job parameters.",
    )
    args = parser.parse_args()
    fair = _fair_jobs(
        args.results_root,
        required_backbones=("sasrec",) if args.profile == "lean" else None,
        reference_plan=args.reference_plan,
    )
    if args.stage == "plan":
        if args.profile == "lean":
            plans = {
                "sensitivity": sensitivity_jobs(
                    fair,
                    seeds=TUNE_SEEDS,
                    axes=LEAN_SENSITIVITY_AXES,
                ),
                "coserec_tune": coserec_tune_jobs(fair),
            }
        else:
            plans = {
                "factorial": factorial_jobs(fair),
                "random_pool": random_pool_jobs(fair),
                "sensitivity": sensitivity_jobs(fair),
                "mce_tune": mce_tune_jobs(fair, args.results_root),
                "coserec_tune": coserec_tune_jobs(fair),
            }
    else:
        plans = (
            {"coserec_final": finalize_coserec_jobs(fair, args.results_root)}
            if args.profile == "lean"
            else finalize_jobs(fair, args.results_root)
        )
    for name, jobs in plans.items():
        save_plan(jobs, str(Path(args.output_dir) / f"{name}.jsonl"))
    if args.stage == "finalize":
        selected = (
            coserec_selection_manifest(fair, args.results_root, plans["coserec_final"])
            if args.profile == "lean"
            else selection_manifest(fair, args.results_root, plans)
        )
        manifest = Path(args.output_dir) / "selection_manifest.json"
        manifest.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()



