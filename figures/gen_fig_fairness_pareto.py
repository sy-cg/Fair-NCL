"""Generate the reviewer-facing NDCG@10 utility--fairness Pareto figure.

The input is prepared by ``analysis/prepare_reviewer_fairness_data.py``.
This script only visualizes that frozen table; it does not recompute results.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pypdfium2 as pdfium
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "tables" / "supplementary" / "reviewer_fairness"
    / "pareto_main_methods_ndcg10.csv"
)
PDF_PATH = ROOT / "figures" / "fairness_pareto_main.pdf"
PNG_PATH = ROOT / "figures" / "fairness_pareto_main.png"

DATASETS = (("ml-1m", "ML-1M"), ("taobao", "Taobao"))
BACKBONES = (
    ("sasrec", "SASRec"),
    ("bert4rec", "BERT4Rec"),
    ("gru4rec", "GRU4Rec"),
    ("caser", "Caser"),
)

BLUE = HexColor("#0072B2")
ORANGE = HexColor("#D55E00")
GREEN = HexColor("#009E73")
GRAY = HexColor("#8A9298")
DARK = HexColor("#24292D")
GRID = HexColor("#E2E7EA")


def read_points() -> list[dict[str, object]]:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 66:
        raise ValueError(f"Expected 66 Pareto rows, found {len(rows)}")
    points: list[dict[str, object]] = []
    for row in rows:
        points.append({
            "dataset": row["dataset"],
            "backbone": row["backbone"],
            "method": row["method"],
            "ndcg": float(row["ndcg_10"]) * 100.0,
            "gap": float(row["mean_ndcg_gap_10"]) * 1000.0,
            "is_pareto": row["is_pareto"].lower() == "true",
        })
    return points


def draw_star(pdf: canvas.Canvas, x: float, y: float, radius: float = 5.0) -> None:
    path = pdf.beginPath()
    for index in range(10):
        angle = math.pi / 2 + index * math.pi / 5
        distance = radius if index % 2 == 0 else radius * 0.43
        xx = x + distance * math.cos(angle)
        yy = y + distance * math.sin(angle)
        (path.moveTo if index == 0 else path.lineTo)(xx, yy)
    path.close()
    pdf.setFillColor(ORANGE)
    pdf.setStrokeColor(HexColor("#8F3D00"))
    pdf.drawPath(path, fill=1, stroke=1)


def draw_diamond(pdf: canvas.Canvas, x: float, y: float) -> None:
    path = pdf.beginPath()
    path.moveTo(x, y + 3.5)
    for xx, yy in ((x + 3.5, y), (x, y - 3.5), (x - 3.5, y)):
        path.lineTo(xx, yy)
    path.close()
    pdf.setFillColor(GREEN)
    pdf.drawPath(path, fill=1, stroke=0)


def draw_panel(
    pdf: canvas.Canvas,
    block: list[dict[str, object]],
    title: str,
    left: float,
    bottom: float,
    width: float,
    height: float,
    off_scale: bool,
) -> None:
    visible = [
        point for point in block
        if not (off_scale and point["method"] == "sm_pcfr")
    ]
    x_values = [float(point["ndcg"]) for point in visible]
    y_values = [float(point["gap"]) for point in visible]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max((x_max - x_min) * 0.09, 0.02)
    y_pad = max((y_max - y_min) * 0.10, 0.08)
    x_min -= x_pad
    x_max += x_pad
    y_min = max(0.0, y_min - y_pad)
    y_max += y_pad

    plot_left = left + 27
    plot_bottom = bottom + 23
    plot_width = width - 31
    plot_height = height - 43

    def position(point: dict[str, object]) -> tuple[float, float]:
        x = plot_left + (float(point["ndcg"]) - x_min) / (x_max - x_min) * plot_width
        y = plot_bottom + (float(point["gap"]) - y_min) / (y_max - y_min) * plot_height
        return x, y

    fair = next(point for point in block if point["method"] == "fair_ncl")
    pdf.setFillColor(DARK)
    pdf.setFont("Times-Bold", 7.2)
    pdf.drawString(left + 1, bottom + height - 10, title)
    pdf.setFont("Times-Roman", 6.2)
    status = "Front" if fair["is_pareto"] else "Off front"
    pdf.drawRightString(left + width - 1, bottom + height - 10, status)

    pdf.setStrokeColor(GRID)
    pdf.setLineWidth(0.45)
    pdf.setFont("Times-Roman", 5.8)
    for index in range(4):
        fraction = index / 3
        xx = plot_left + fraction * plot_width
        yy = plot_bottom + fraction * plot_height
        pdf.line(xx, plot_bottom, xx, plot_bottom + plot_height)
        pdf.line(plot_left, yy, plot_left + plot_width, yy)
        pdf.setFillColor(HexColor("#525B61"))
        pdf.drawCentredString(
            xx, plot_bottom - 10,
            f"{x_min + fraction * (x_max - x_min):.2f}",
        )
        pdf.drawRightString(
            plot_left - 4, yy - 2,
            f"{y_min + fraction * (y_max - y_min):.2f}",
        )
    pdf.setStrokeColor(HexColor("#4D555B"))
    pdf.setLineWidth(0.65)
    pdf.rect(plot_left, plot_bottom, plot_width, plot_height, fill=0, stroke=1)

    frontier = sorted(
        (point for point in visible if point["is_pareto"]),
        key=lambda point: float(point["ndcg"]),
    )
    pdf.setStrokeColor(BLUE)
    pdf.setLineWidth(1.35)
    for first, second in zip(frontier, frontier[1:]):
        pdf.line(*position(first), *position(second))

    for point in visible:
        x, y = position(point)
        if point["method"] == "fair_ncl":
            draw_star(pdf, x, y)
        elif point["method"] == "baseline":
            pdf.setFillColor(DARK)
            pdf.rect(x - 3.0, y - 3.0, 6.0, 6.0, fill=1, stroke=0)
        elif point["method"] == "coserec":
            draw_diamond(pdf, x, y)
        else:
            pdf.setFillColor(GRAY)
            pdf.circle(x, y, 2.6, fill=1, stroke=0)

    if off_scale:
        pdf.setFont("Times-Roman", 5.5)
        pdf.setFillColor(HexColor("#3D464D"))
        pdf.drawRightString(
            plot_left + plot_width - 2,
            plot_bottom + plot_height - 8,
            "PCFR-SM: 46.36 (off scale)",
        )


def draw_figure(points: list[dict[str, object]]) -> None:
    page_width, page_height = 7.15 * 72, 4.55 * 72
    pdf = canvas.Canvas(str(PDF_PATH), pagesize=(page_width, page_height))
    left_margin, right_margin = 29, 5
    top_margin, bottom_margin = 15, 37
    column_gap, row_gap = 6, 8
    panel_width = (
        page_width - left_margin - right_margin - 3 * column_gap
    ) / 4
    panel_height = (page_height - top_margin - bottom_margin - row_gap) / 2

    for dataset_index, (dataset, dataset_name) in enumerate(DATASETS):
        for backbone_index, (backbone, backbone_name) in enumerate(BACKBONES):
            panel_index = dataset_index * len(BACKBONES) + backbone_index
            column, row = panel_index % 4, panel_index // 4
            left = left_margin + column * (panel_width + column_gap)
            bottom = page_height - top_margin - (row + 1) * panel_height - row * row_gap
            block = [
                point for point in points
                if (point["dataset"], point["backbone"]) == (dataset, backbone)
            ]
            expected = 9 if backbone == "sasrec" else 8
            if len(block) != expected:
                raise ValueError(f"Expected {expected} methods for {dataset}/{backbone}")
            draw_panel(
                pdf,
                block,
                f"{dataset_name} / {backbone_name}",
                left,
                bottom,
                panel_width,
                panel_height,
                dataset == "ml-1m" and backbone == "bert4rec",
            )

    pdf.setFillColor(DARK)
    pdf.setFont("Times-Roman", 7.4)
    pdf.drawCentredString(page_width / 2, 26, "NDCG@10 (x 10^-2; higher is better)")
    pdf.saveState()
    pdf.translate(10, page_height / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Mean absolute gender/age NDCG gap (x 10^-3; lower is better)")
    pdf.restoreState()

    legend_y = 9
    pdf.setFillColor(GRAY)
    pdf.circle(45, legend_y + 2, 2.6, fill=1, stroke=0)
    pdf.setFillColor(DARK)
    pdf.rect(140, legend_y - 1, 6, 6, fill=1, stroke=0)
    draw_diamond(pdf, 224, legend_y + 2)
    draw_star(pdf, 304, legend_y + 2, 4.6)
    pdf.setStrokeColor(BLUE)
    pdf.setLineWidth(1.35)
    pdf.line(398, legend_y + 2, 412, legend_y + 2)
    pdf.setFillColor(DARK)
    pdf.setFont("Times-Roman", 7.0)
    for x, label in (
        (53, "Other method"),
        (150, "Backbone"),
        (232, "CoSeRec"),
        (314, "Fair-NCL"),
        (418, "Pareto front"),
    ):
        pdf.drawString(x, legend_y, label)
    pdf.save()

    document = pdfium.PdfDocument(str(PDF_PATH))
    document[0].render(scale=300 / 72).to_pil().save(PNG_PATH, dpi=(300, 300))


def main() -> None:
    points = read_points()
    draw_figure(points)
    front_count = sum(
        point["method"] == "fair_ncl" and point["is_pareto"]
        for point in points
    )
    print(f"Generated {PDF_PATH.name} and {PNG_PATH.name}")
    print(f"Fair-NCL lies on {front_count}/8 descriptive fronts")


if __name__ == "__main__":
    main()
