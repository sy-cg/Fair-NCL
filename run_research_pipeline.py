import argparse
import json
import os

from analysis.mechanism_reporting import export_mechanism_report
from analysis.reporting import export_experiment_report, PRIMARY_REPORT_METRICS
from data.unified_preprocessing import DatasetBuildConfig, UnifiedSequentialPreprocessor, default_raw_dir
from experiments.job_utils import merge_nested_params
from experiments.specs import build_experiment_plan, save_plan
from experiments.runner import load_job_from_jsonl, run_experiment_job


def preprocess_command(args):
    datasets = _expand_datasets(args.datasets)
    for dataset in datasets:
        cfg = DatasetBuildConfig(
            dataset=dataset,
            raw_dir=args.raw_dir or default_raw_dir(dataset),
            output_dir=args.output_dir,
            max_rows=args.max_rows,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
            max_seq_len=args.max_seq_len,
            lastfm_item_level=args.lastfm_item_level,
        )
        UnifiedSequentialPreprocessor(cfg).build()


def plan_command(args):
    base_params = _load_base_params(args.params_file, args.params_json)
    jobs = build_experiment_plan(
        phase=args.phase,
        datasets=args.datasets,
        backbones=args.backbones,
        seeds=args.seeds,
        base_params=base_params,
        pairs=args.pairs,
        ablation_methods=args.ablation_methods,
    )
    save_plan(jobs, args.output)


def run_command(args):
    job = load_job_from_jsonl(args.jobs, args.index)
    run_experiment_job(
        job,
        processed_root=args.processed_root,
        results_root=args.results_root,
        debug=args.debug,
        skip_existing=args.skip_existing,
    )


def summarize_command(args):
    datasets = _expand_datasets(args.datasets)
    backbones = _expand_backbones(args.backbones)
    export_experiment_report(
        jobs_path=args.jobs,
        results_root=args.results_root,
        output_path=args.output,
        phase=args.phase,
        datasets=datasets,
        backbones=backbones,
    )
    print(f"Saved Excel report to {args.output}")


def summarize_mechanism_command(args):
    datasets = _expand_datasets(args.datasets)
    backbones = _expand_backbones(args.backbones)
    export_mechanism_report(
        jobs_path=args.jobs,
        results_root=args.results_root,
        output_path=args.output,
        phase=args.phase,
        datasets=datasets,
        backbones=backbones,
    )
    print(f"Saved mechanism Excel report to {args.output}")


def plot_command(args):
    from analysis.plotting import export_parameter_plots

    datasets = _expand_datasets(args.datasets)
    backbones = _expand_backbones(args.backbones)
    report = export_parameter_plots(
        jobs_path=args.jobs,
        results_root=args.results_root,
        output_dir=args.output_dir,
        phase=args.phase,
        datasets=datasets,
        backbones=backbones,
        metrics=args.metrics or list(PRIMARY_REPORT_METRICS),
        file_format=args.format,
    )
    figure_count = len(report["figure_paths"]) if "figure_paths" in report else 0
    print(f"Saved {figure_count} figures and curve CSVs to {args.output_dir}")


def _expand_datasets(datasets):
    if len(datasets) == 1 and datasets[0] == "all":
        return ["ml-1m", "lastfm-1k", "taobao"]
    return datasets


def _expand_backbones(backbones):
    if len(backbones) == 1 and backbones[0] == "all":
        return ["sasrec", "bert4rec", "gru4rec", "caser"]
    return backbones


def main():
    parser = argparse.ArgumentParser(description="Fair-NCL research pipeline helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Build unified processed datasets.")
    preprocess.add_argument("--datasets", nargs="+", default=["all"])
    preprocess.add_argument("--raw-dir", default=None, help="Only use with a single dataset.")
    preprocess.add_argument("--output-dir", default="processed_data")
    preprocess.add_argument("--max-rows", type=int, default=None)
    preprocess.add_argument("--min-user-interactions", type=int, default=5)
    preprocess.add_argument("--min-item-interactions", type=int, default=5)
    preprocess.add_argument("--max-seq-len", type=int, default=100)
    preprocess.add_argument("--lastfm-item-level", choices=["artist", "track"], default="artist")
    preprocess.set_defaults(func=preprocess_command)

    plan = subparsers.add_parser("plan", help="Generate staged experiment jobs.")
    plan.add_argument("--phase", choices=["backbone", "augment", "loss", "ablation", "comparison", "all"], required=True)
    plan.add_argument("--datasets", nargs="+", default=None, help="Omit to use the phase default matrix.")
    plan.add_argument("--backbones", nargs="+", default=None, help="Omit to use the phase default matrix.")
    plan.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Optional explicit dataset:backbone list. When provided, it overrides the Cartesian dataset/backbone matrix.",
    )
    plan.add_argument(
        "--ablation-methods",
        nargs="+",
        default=None,
        help=(
            "Ablation methods or presets used only with --phase ablation/all. "
            "Presets: legacy, requested/new, main, alpha_tradeoff/experimental, "
            "semantic_alpha_tradeoff/semantic, semantic_hybrid, all/full, all_with_experimental."
        ),
    )
    plan.add_argument("--seeds", nargs="+", type=int, default=[42])
    plan.add_argument(
        "--params-file",
        default=None,
        help="JSON file with fixed parameters. Supports a flat object or scoped keys like __default__/__backbones__/__datasets__/__dataset_backbones__.",
    )
    plan.add_argument(
        "--params-json",
        default=None,
        help="Inline JSON object with fixed parameters. Supports a flat object or scoped keys like __default__/__backbones__/__datasets__/__dataset_backbones__.",
    )
    plan.add_argument("--output", default=os.path.join("experiments", "jobs.jsonl"))
    plan.set_defaults(func=plan_command)

    run = subparsers.add_parser("run", help="Run one generated experiment job.")
    run.add_argument("--jobs", required=True)
    run.add_argument("--index", type=int, default=0)
    run.add_argument("--processed-root", default="processed_data")
    run.add_argument("--results-root", default="results")
    run.add_argument("--debug", action="store_true")
    run.add_argument("--skip-existing", action="store_true")
    run.set_defaults(func=run_command)

    summarize = subparsers.add_parser("summarize", help="Export ablation/comparison results to Excel.")
    summarize.add_argument("--jobs", required=True, help="JSONL plan file for one experiment phase.")
    summarize.add_argument("--results-root", default="results")
    summarize.add_argument("--phase", choices=["ablation", "comparison"], required=True)
    summarize.add_argument("--datasets", nargs="+", default=["all"])
    summarize.add_argument("--backbones", nargs="+", default=["all"])
    summarize.add_argument("--output", default=os.path.join("results", "paper_report.xlsx"))
    summarize.set_defaults(func=summarize_command)

    summarize_mechanism = subparsers.add_parser(
        "summarize-mechanism",
        help="Export RQ4 mechanism-analysis results to Excel.",
    )
    summarize_mechanism.add_argument("--jobs", required=True, help="JSONL plan file for one experiment phase.")
    summarize_mechanism.add_argument("--results-root", default="results")
    summarize_mechanism.add_argument("--phase", default=None)
    summarize_mechanism.add_argument("--datasets", nargs="+", default=["all"])
    summarize_mechanism.add_argument("--backbones", nargs="+", default=["all"])
    summarize_mechanism.add_argument("--output", default=os.path.join("results", "rq4_mechanism_report.xlsx"))
    summarize_mechanism.set_defaults(func=summarize_mechanism_command)

    plot = subparsers.add_parser("plot", help="Export parameter analysis plots.")
    plot.add_argument("--jobs", required=True, help="JSONL plan file for one tuning phase.")
    plot.add_argument("--results-root", default="results")
    plot.add_argument("--phase", choices=["backbone", "augment", "loss"], required=True)
    plot.add_argument("--datasets", nargs="+", default=["all"])
    plot.add_argument("--backbones", nargs="+", default=["all"])
    plot.add_argument("--metrics", nargs="+", default=list(PRIMARY_REPORT_METRICS))
    plot.add_argument("--output-dir", default=os.path.join("figures", "parameter_analysis"))
    plot.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    plot.set_defaults(func=plot_command)

    args = parser.parse_args()
    args.func(args)


def _load_base_params(params_file, params_json):
    params = {}
    if params_file:
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Params file not found: {params_file}")
        with open(params_file, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"Params file must contain a JSON object: {params_file}")
        params = merge_nested_params(params, loaded)
    if params_json:
        loaded = json.loads(params_json)
        if not isinstance(loaded, dict):
            raise ValueError("params-json must decode to a JSON object.")
        params = merge_nested_params(params, loaded)
    return params


if __name__ == "__main__":
    main()
