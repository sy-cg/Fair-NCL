import json
import unittest
from pathlib import Path

from experiments.specs import build_experiment_plan


FINAL_SEEDS = (42, 2025, 2026, 2027, 2028)
DECLARED_EPSILON_GRID = {0.5, 1.0, 2.0}
DECLARED_BETA_GRID = {0.2, 0.3, 0.4}
BASE_PARAMS = {
    "learning_rate": 0.001,
    "hidden_units": 64,
    "dropout_rate": 0.2,
    "epsilon": 1.0,
    "augment_ratio": 0.2,
    "utility_alpha": 0.7,
    "utility_beta": 0.3,
}


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SupplementaryPlanTests(unittest.TestCase):
    def setUp(self):
        self.jobs = build_experiment_plan(
            "supplementary",
            seeds=FINAL_SEEDS,
            base_params=BASE_PARAMS,
        )

    def test_revision_plan_uses_only_approved_datasets_and_five_seeds(self):
        self.assertEqual({job.dataset for job in self.jobs}, {"ml-1m", "taobao"})
        self.assertEqual({job.seed for job in self.jobs}, set(FINAL_SEEDS))

    def test_revision_phase_defaults_to_final_five_seeds(self):
        jobs = build_experiment_plan("supplementary", base_params=BASE_PARAMS)
        self.assertEqual({job.seed for job in jobs}, set(FINAL_SEEDS))

    def test_direct_controls_cover_all_four_backbones(self):
        direct_methods = {
            "baseline",
            "fair_ncl",
            "ncl_only",
            "random_aug",
            "similarity_aug",
            "ncl_sr_mce",
        }
        expected_pairs = {
            (dataset, backbone)
            for dataset in ("ml-1m", "taobao")
            for backbone in ("sasrec", "bert4rec", "gru4rec", "caser")
        }
        for method in direct_methods:
            actual_pairs = {(job.dataset, job.backbone) for job in self.jobs if job.method == method}
            self.assertEqual(actual_pairs, expected_pairs, method)

    def test_focused_factorial_controls_are_sasrec_only(self):
        methods = {
            "skew_only_aug",
            "fair_aug_no_geometry",
            "fair_aug_no_augmented_ce",
            "raw_skew_fair_ncl",
        }
        for method in methods:
            jobs = [job for job in self.jobs if job.method == method]
            self.assertTrue(jobs, method)
            self.assertEqual({job.backbone for job in jobs}, {"sasrec"})

    def test_control_params_change_only_declared_factors(self):
        by_method = {}
        for job in self.jobs:
            if job.dataset == "ml-1m" and job.backbone == "sasrec" and job.seed == 42:
                by_method[job.method] = job.params

        self.assertEqual(by_method["fair_ncl"]["bias_metric"], "prior_jsd")
        self.assertTrue(by_method["fair_ncl"]["export_per_user_metrics"])
        self.assertTrue(by_method["fair_ncl"]["report_intersectional_groups"])
        self.assertEqual(by_method["raw_skew_fair_ncl"]["bias_metric"], "raw_range")
        self.assertEqual(by_method["raw_skew_fair_ncl"]["bias_event_source"], "prefix_samples")
        self.assertEqual(by_method["skew_only_aug"]["utility_alpha"], 0.0)
        self.assertEqual(by_method["fair_aug_no_geometry"]["fair_ncl_align_weight"], 0.0)
        self.assertEqual(by_method["fair_aug_no_geometry"]["fair_ncl_var_weight"], 0.0)
        self.assertEqual(by_method["fair_aug_no_geometry"]["fair_ncl_cov_weight"], 0.0)
        self.assertEqual(by_method["fair_aug_no_augmented_ce"]["fair_ncl_aug_rec_weight"], 0.0)
        self.assertEqual(by_method["ncl_sr_mce"]["ncl_objective"], "mce")
        self.assertEqual(by_method["ncl_sr_mce"]["fair_ncl_aug_rec_weight"], 0.0)
        self.assertEqual(by_method["ncl_only"]["fair_ncl_aug_rec_weight"], 0.0)


class ValidationAuditPlanTests(unittest.TestCase):
    def test_revision_base_file_matches_submitted_parameter_provenance(self):
        revision_path = Path("configs/revision_base_params.json")
        self.assertTrue(revision_path.exists(), "missing consolidated revision parameter file")

        revision = _load_json(revision_path)["__dataset_backbones__"]
        backbone = _load_json("configs/backbone_best.json")["__dataset_backbones__"]
        original_loss = _load_json("configs/loss_selected.json")["__dataset_backbones__"]
        adjusted_sasrec = _load_json("configs/loss_ml_taobao_sasrec.json")["__dataset_backbones__"]

        self.assertEqual(set(revision), {"ml-1m", "taobao"})
        for dataset in ("ml-1m", "taobao"):
            self.assertEqual(set(revision[dataset]), {"sasrec", "bert4rec", "gru4rec", "caser"})
            self.assertEqual(revision[dataset]["sasrec"], adjusted_sasrec[dataset]["sasrec"])
            for backbone_name in ("bert4rec", "gru4rec", "caser"):
                self.assertEqual(
                    revision[dataset][backbone_name],
                    original_loss[dataset][backbone_name],
                )
            for backbone_name in ("sasrec", "bert4rec", "gru4rec", "caser"):
                for key in ("learning_rate", "hidden_units", "dropout_rate"):
                    self.assertEqual(
                        revision[dataset][backbone_name][key],
                        backbone[dataset][backbone_name][key],
                    )

    def test_fair_ncl_candidates_keep_submitted_center_and_declared_grid(self):
        submitted = {
            "ml-1m": {
                "learning_rate": 0.0005,
                "hidden_units": 64,
                "dropout_rate": 0.2,
                "epsilon": 2.0,
                "augment_ratio": 0.1,
                "utility_alpha": 0.6,
                "utility_beta": 0.4,
                "fair_ncl_align_weight": 1.0,
                "fair_ncl_var_weight": 0.5,
                "fair_ncl_cov_weight": 0.02,
            },
            "taobao": {
                "learning_rate": 0.0005,
                "hidden_units": 64,
                "dropout_rate": 0.1,
                "epsilon": 0.5,
                "augment_ratio": 0.1,
                "utility_alpha": 0.7,
                "utility_beta": 0.3,
                "fair_ncl_align_weight": 0.5,
                "fair_ncl_var_weight": 0.5,
                "fair_ncl_cov_weight": 0.08,
            },
        }

        for dataset, center in submitted.items():
            jobs = build_experiment_plan(
                "validation_audit",
                datasets=(dataset,),
                backbones=("sasrec",),
                seeds=(42,),
                base_params={"__dataset_backbones__": {dataset: {"sasrec": center}}},
            )
            candidates = [job for job in jobs if job.method == "fair_ncl"]
            by_name = {job.params["audit_candidate"]: job.params for job in candidates}

            self.assertEqual(set(by_name), {"c0", "c1", "c2", "c3", "c4"})
            for key, value in center.items():
                self.assertEqual(by_name["c0"][key], value, f"{dataset}:{key}")

            epsilon_values = {round(float(job.params["epsilon"]), 10) for job in candidates}
            beta_values = {round(float(job.params["utility_beta"]), 10) for job in candidates}
            self.assertEqual(epsilon_values, DECLARED_EPSILON_GRID)
            self.assertEqual(beta_values, DECLARED_BETA_GRID)
            for job in candidates:
                self.assertAlmostEqual(
                    float(job.params["utility_alpha"]) + float(job.params["utility_beta"]),
                    1.0,
                )
                for key in (
                    "fair_ncl_align_weight",
                    "fair_ncl_var_weight",
                    "fair_ncl_cov_weight",
                ):
                    self.assertEqual(job.params[key], center[key])

    def test_validation_reference_is_three_seed_baseline_only_and_never_tests(self):
        jobs = build_experiment_plan("validation_reference", base_params=BASE_PARAMS)

        self.assertEqual(len(jobs), 24)
        self.assertEqual({job.dataset for job in jobs}, {"ml-1m", "taobao"})
        self.assertEqual({job.backbone for job in jobs}, {"sasrec", "bert4rec", "gru4rec", "caser"})
        self.assertEqual({job.seed for job in jobs}, {42, 2025, 2026})
        self.assertEqual({job.method for job in jobs}, {"baseline"})
        self.assertTrue(all(job.params["evaluate_test"] is False for job in jobs))

    def test_each_method_receives_equal_five_candidate_budget(self):
        jobs = build_experiment_plan(
            "validation_audit",
            datasets=("ml-1m",),
            backbones=("sasrec",),
            seeds=(42,),
            base_params=BASE_PARAMS,
        )
        methods = {job.method for job in jobs}
        for method in methods:
            method_jobs = [job for job in jobs if job.method == method]
            self.assertEqual(len(method_jobs), 5, method)
            self.assertEqual(
                {job.params["audit_candidate"] for job in method_jobs},
                {"c0", "c1", "c2", "c3", "c4"},
            )
            self.assertTrue(all(job.params["evaluate_test"] is False for job in method_jobs))

    def test_frozen_fairness_baselines_declare_reference_initialization(self):
        audit_jobs = build_experiment_plan(
            "validation_audit",
            datasets=("ml-1m",),
            backbones=("caser",),
            seeds=(42,),
            base_params=BASE_PARAMS,
        )
        final_jobs = build_experiment_plan(
            "supplementary",
            datasets=("ml-1m",),
            backbones=("caser",),
            seeds=(42,),
            base_params=BASE_PARAMS,
        )

        frozen_jobs = [
            job
            for job in audit_jobs + final_jobs
            if job.method in {"afrl", "pfrec"}
        ]
        self.assertTrue(frozen_jobs)
        self.assertTrue(all(
            job.params["frozen_backbone_init"] == "validation_reference"
            for job in frozen_jobs
        ))

    def test_mce_controls_receive_matched_validation_budget(self):
        jobs = build_experiment_plan(
            "validation_audit",
            datasets=("ml-1m",),
            backbones=("sasrec", "gru4rec"),
            seeds=(42,),
            base_params=BASE_PARAMS,
        )

        sasrec_mce = [job for job in jobs if job.backbone == "sasrec" and job.method == "ncl_sr_mce"]
        sasrec_fair_mce = [job for job in jobs if job.backbone == "sasrec" and job.method == "fair_aug_mce"]
        gru_mce = [job for job in jobs if job.backbone == "gru4rec" and job.method == "ncl_sr_mce"]
        gru_fair_mce = [job for job in jobs if job.backbone == "gru4rec" and job.method == "fair_aug_mce"]

        self.assertEqual(len(sasrec_mce), 5)
        self.assertEqual(len(sasrec_fair_mce), 5)
        self.assertEqual(len(gru_mce), 5)
        self.assertEqual(gru_fair_mce, [])
        self.assertEqual(
            {(job.params["mce_uniform_weight"], job.params["mce_align_weight"]) for job in sasrec_mce},
            {(1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.0, 2.0)},
        )

    def test_caser_mce_stability_plan_is_single_seed_low_weight_diagnostic(self):
        jobs = build_experiment_plan("mce_stability", base_params=BASE_PARAMS)

        self.assertEqual(len(jobs), 8)
        self.assertEqual({job.dataset for job in jobs}, {"ml-1m", "taobao"})
        self.assertEqual({job.backbone for job in jobs}, {"caser"})
        self.assertEqual({job.method for job in jobs}, {"ncl_sr_mce"})
        self.assertEqual({job.seed for job in jobs}, {42})
        self.assertEqual(
            {
                (job.params["mce_uniform_weight"], job.params["mce_align_weight"])
                for job in jobs
            },
            {(0.01, 0.01), (0.05, 0.05), (0.1, 0.1), (0.25, 0.25)},
        )
        self.assertTrue(all(job.params["evaluate_test"] is False for job in jobs))
        self.assertTrue(all(job.params["mce_diagnostic"] is True for job in jobs))

    def test_method_scoped_overrides_are_applied_without_changing_backbone_params(self):
        params = {
            "__default__": BASE_PARAMS,
            "__methods__": {
                "fair_ncl": {"utility_alpha": 0.6, "utility_beta": 0.4},
                "adv_debias": {"adv_debias_weight": 0.8},
            },
        }

        jobs = build_experiment_plan(
            "supplementary",
            datasets=("ml-1m",),
            backbones=("sasrec",),
            seeds=(42,),
            base_params=params,
        )
        fair_ncl = next(job for job in jobs if job.method == "fair_ncl")
        adv = next(job for job in jobs if job.method == "adv_debias")

        self.assertEqual(fair_ncl.params["utility_beta"], 0.4)
        self.assertEqual(adv.params["adv_debias_weight"], 0.8)
        self.assertEqual(fair_ncl.params["learning_rate"], BASE_PARAMS["learning_rate"])


class RevisionPlanArtifactTests(unittest.TestCase):
    def test_reference_artifact_uses_backbone_parameters_only(self):
        jobs = _load_jsonl("experiments/revision_baseline_reference.jsonl")
        backbone = _load_json("configs/backbone_best.json")["__dataset_backbones__"]
        protocol_keys = {
            "export_per_user_metrics",
            "report_intersectional_groups",
            "evaluate_test",
            "export_mechanism_analysis",
        }

        self.assertEqual(len(jobs), 24)
        for job in jobs:
            expected = backbone[job["dataset"]][job["backbone"]]
            self.assertEqual(set(job["params"]), set(expected) | protocol_keys)
            for key, value in expected.items():
                self.assertEqual(job["params"][key], value)
            self.assertFalse(job["params"]["evaluate_test"])

    def test_audit_artifact_uses_revision_centers_and_declared_grid(self):
        jobs = _load_jsonl("experiments/revision_validation_audit.jsonl")
        revision = _load_json("configs/revision_base_params.json")["__dataset_backbones__"]

        self.assertEqual(len(jobs), 1020)
        fair_jobs = [job for job in jobs if job["method"] == "fair_ncl"]
        for dataset in ("ml-1m", "taobao"):
            for backbone_name in ("sasrec", "bert4rec", "gru4rec", "caser"):
                group = [
                    job for job in fair_jobs
                    if job["dataset"] == dataset and job["backbone"] == backbone_name
                ]
                self.assertEqual(len(group), 15)
                centers = [job for job in group if job["params"]["audit_candidate"] == "c0"]
                self.assertEqual(len(centers), 3)
                for key, value in revision[dataset][backbone_name].items():
                    self.assertEqual(centers[0]["params"][key], value)
                self.assertEqual(
                    {round(float(job["params"]["epsilon"]), 10) for job in group},
                    DECLARED_EPSILON_GRID,
                )
                self.assertEqual(
                    {round(float(job["params"]["utility_beta"]), 10) for job in group},
                    DECLARED_BETA_GRID,
                )
        self.assertTrue(all(job["params"]["evaluate_test"] is False for job in jobs))

    def test_five_seed_template_uses_revision_base_parameters(self):
        jobs = _load_jsonl("experiments/revision_supplementary_old_params.jsonl")
        revision = _load_json("configs/revision_base_params.json")["__dataset_backbones__"]

        self.assertEqual(len(jobs), 540)
        self.assertEqual({job["seed"] for job in jobs}, set(FINAL_SEEDS))
        for dataset in ("ml-1m", "taobao"):
            for backbone_name in ("sasrec", "bert4rec", "gru4rec", "caser"):
                job = next(
                    job for job in jobs
                    if job["dataset"] == dataset
                    and job["backbone"] == backbone_name
                    and job["method"] == "fair_ncl"
                    and job["seed"] == 42
                )
                for key, value in revision[dataset][backbone_name].items():
                    self.assertEqual(job["params"][key], value)
                self.assertTrue(job["params"]["evaluate_test"])


if __name__ == "__main__":
    unittest.main()
