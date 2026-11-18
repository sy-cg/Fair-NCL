"""Export the reported comparison, ablation, and robustness tables as TeX."""

import csv
from pathlib import Path

from rebuild_five_seed_paper_assets import ablation_table, main_table, rq3_summary, rq3_table


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tables/supplementary/reproduced_paper_tables"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for dataset in ("ml-1m", "taobao"):
        for cutoff in (10, 20):
            name = f"comparison_{dataset}_{cutoff}.tex"
            (OUTPUT / name).write_text(main_table(cutoff, dataset) + "\n", encoding="utf-8")
    (OUTPUT / "ablation_sasrec_10.tex").write_text(ablation_table() + "\n", encoding="utf-8")
    rows = rq3_summary()
    (OUTPUT / "rq3_robustness_summary.tex").write_text(rq3_table(rows) + "\n", encoding="utf-8")
    with (OUTPUT / "rq3_robustness_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote six TeX tables and one CSV to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
