
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Polygon, Ellipse, Arc

# ----------------------------
# Global style
# ----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
})

# Soft academic color palette
COL = {
    "bg": "#ffffff",
    "panel": "#f7f8fb",
    "separator": "#d5d8df",
    "text": "#101010",
    "muted": "#5c6170",
    "blue": "#d9ecfb",
    "blue_edge": "#7aa6c7",
    "green": "#e8f5e8",
    "green_edge": "#8fbd92",
    "orange": "#f6c37a",
    "orange_light": "#fff0d7",
    "beige": "#f7ecd9",
    "beige_edge": "#c9b38a",
    "purple": "#e5e6fb",
    "purple_edge": "#8990c9",
    "gray": "#e9ecef",
    "gray_edge": "#8f949e",
    "dash": "#9aa3ad",
}

# Canvas in arbitrary diagram units
W, H = 180, 80
fig, ax = plt.subplots(figsize=(18, 8), dpi=300)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(COL["bg"])

# ----------------------------
# Helper functions
# ----------------------------
def rounded_box(ax, x, y, w, h, fc="white", ec="#333333", lw=1.1, r=1.2,
                alpha=1.0, ls="-", z=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=ls, alpha=alpha, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def rect(ax, x, y, w, h, fc="white", ec="#333", lw=1.0, alpha=1.0, ls="-", z=1):
    patch = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                      linewidth=lw, alpha=alpha, linestyle=ls, zorder=z)
    ax.add_patch(patch)
    return patch


def label(ax, x, y, s, size=10, weight="normal", ha="center", va="center",
          color=COL["text"], style="normal", z=5, rotation=0):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
                   color=color, style=style, zorder=z, rotation=rotation)


def arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.15, ls="-",
          mutation=10, rad=0.0, z=4, alpha=1.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mutation,
                        linewidth=lw, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z, alpha=alpha)
    ax.add_patch(a)
    return a


def line(ax, x1, y1, x2, y2, color="#555", lw=1.0, ls="-", z=2, alpha=1.0):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=z, alpha=alpha)


def item_card(ax, x, y, w=5.0, h=4.6, color="#d9ecfb", label_text="", skew=False, z=3):
    if skew:
        ax.add_patch(Ellipse((x + w/2, y + h/2), w*1.45, h*1.45,
                             facecolor="#f6a23a", edgecolor="none", alpha=0.20, zorder=z-1))
        ax.add_patch(Ellipse((x + w/2, y + h/2), w*1.15, h*1.15,
                             facecolor="#f6a23a", edgecolor="none", alpha=0.10, zorder=z-1))
    rounded_box(ax, x, y, w, h, fc="#fbfbfb", ec="#c4b7a5", lw=1.0, r=0.8, z=z)
    # simple abstract icon
    ax.add_patch(Circle((x+w*0.43, y+h*0.58), h*0.18, facecolor=color, edgecolor="#7b7f88", lw=0.7, zorder=z+1))
    ax.add_patch(Rectangle((x+w*0.48, y+h*0.35), w*0.25, h*0.33,
                           facecolor=color, edgecolor="#7b7f88", lw=0.7, zorder=z+1))
    if label_text:
        label(ax, x+w/2, y-1.4, label_text, size=8, z=z+2)


def draw_user(ax, x, y, gender="F", age="18-24", fill="#ffd9c4"):
    ax.add_patch(Circle((x, y), 2.8, facecolor=fill, edgecolor="#d0b1a1", lw=1.0, zorder=3))
    ax.add_patch(Circle((x, y+0.5), 0.75, facecolor="#f6cfb1", edgecolor="#8a6b5b", lw=0.6, zorder=4))
    ax.add_patch(Arc((x, y+0.8), 2.0, 1.7, theta1=190, theta2=350, color="#5d4037", lw=1.5, zorder=5))
    ax.add_patch(Polygon([[x-1.35, y-2.0], [x+1.35, y-2.0], [x+0.6, y-0.4], [x-0.6, y-0.4]],
                         closed=True, facecolor="#9bbbd7", edgecolor="#657c94", lw=0.6, zorder=4))
    sym = "F" if gender == "F" else "M"
    label(ax, x-1.6, y-4.0, sym, size=9, color="#d06a6a" if gender=="F" else "#5f8fd8", weight="bold")
    label(ax, x+0.3, y-4.0, "Female" if gender=="F" else "Male", size=7.2, ha="left")
    label(ax, x-1.6, y-6.0, "Age", size=6.5, color="#d8893d" if gender=="F" else "#5085b6")
    label(ax, x+0.3, y-6.0, age, size=7.2, ha="left")


def draw_sequence(ax, y, gender, age, colors, skew_indices=(), target_color="#d9ecfb"):
    draw_user(ax, 5.2, y+1.5, gender, age, fill="#ffe4d4" if gender=="F" else "#d8ebff")
    xs = [13, 20, 27, 37]
    labs = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_T$"]
    for idx, xx in enumerate(xs):
        item_card(ax, xx, y, color=colors[idx % len(colors)], label_text=labs[idx], skew=idx in skew_indices)
        if idx < len(xs)-1:
            if idx == 2:
                label(ax, (xx+xs[idx+1])/2+1.0, y+2.3, r"$\rightarrow\rightarrow$", size=9)
            else:
                arrow(ax, xx+5.2, y+2.3, xs[idx+1]-0.2, y+2.3, lw=1.0, mutation=8)
    # target
    arrow(ax, xs[-1]+5.2, y+2.3, 44.0, y+2.3, lw=1.0, mutation=8)
    item_card(ax, 44.2, y, color=target_color, label_text=r"$y$", skew=False)


def mini_barchart(ax, x, y, w=5.2, h=5.5, vals=(0.7, 0.4, 0.2, 0.55), color="#b5b5b5"):
    line(ax, x, y, x+w, y, color="#7f858c", lw=0.6)
    for i, v in enumerate(vals):
        bw = w / (len(vals)*1.7)
        bx = x + 0.5 + i * (w/len(vals))
        rect(ax, bx, y, bw, h*v, fc=color, ec="none", lw=0, z=5)


def draw_cooccurrence_graph(ax, cx, cy, radius=7.0):
    nodes = []
    for k in range(8):
        ang = 2 * math.pi * k / 8 + 0.2
        nodes.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
    ax.add_patch(Circle((cx, cy), 1.7, facecolor="#ffe2b8", edgecolor="#d49b5a", lw=1.0, zorder=5))
    label(ax, cx, cy-0.1, "i", size=9, weight="bold")
    for n, (nx, ny) in enumerate(nodes):
        ax.add_patch(FancyBboxPatch((nx-1.25, ny-1.25), 2.5, 2.5, boxstyle="round,pad=0.02,rounding_size=0.6",
                                    facecolor=["#d8efd6", "#d9ecfb", "#f2f2e8"][n % 3],
                                    edgecolor="#83a89b", lw=0.8, zorder=5))
        arrow(ax, cx+1.6*(nx-cx)/radius, cy+1.6*(ny-cy)/radius,
              nx-1.4*(nx-cx)/radius, ny-1.4*(ny-cy)/radius,
              color="#8f9aa3", lw=0.75, mutation=6, alpha=0.75, z=4)
    label(ax, cx, cy-3.3, "source\nitem i", size=8.2)


def scatter_cloud(ax, x, y, w, h, n=180, kind="align", seed=0):
    rng = np.random.default_rng(seed)
    if kind == "align":
        pts1 = rng.normal([x+w*0.35, y+h*0.52], [w*0.09, h*0.12], (n//2, 2))
        pts2 = rng.normal([x+w*0.67, y+h*0.53], [w*0.09, h*0.12], (n//2, 2))
        ax.scatter(pts1[:,0], pts1[:,1], s=2.5, c="#8db6d9", alpha=0.35, edgecolors="none", zorder=5)
        ax.scatter(pts2[:,0], pts2[:,1], s=2.5, c="#dfa96e", alpha=0.35, edgecolors="none", zorder=5)
        arrow(ax, x+w*0.47, y+h*0.53, x+w*0.56, y+h*0.53, lw=0.8, mutation=7)
    elif kind == "var":
        pts = rng.normal([x+w*0.5, y+h*0.48], [w*0.18, h*0.16], (n, 2))
        ax.scatter(pts[:,0], pts[:,1], s=4, c=np.where(np.arange(n)%2, "#8db6d9", "#dfa96e"), alpha=0.55, edgecolors="none", zorder=5)
    elif kind == "cov":
        a = rng.normal(0, 1, n)
        pts1 = np.column_stack([x+w*0.34 + a*w*0.08, y+h*0.50 + a*h*0.13 + rng.normal(0, h*0.04, n)])
        pts2 = np.column_stack([x+w*0.66 + a*w*0.08, y+h*0.48 + a*h*0.13 + rng.normal(0, h*0.04, n)])
        ax.scatter(pts1[:,0], pts1[:,1], s=3, c="#8db6d9", alpha=0.45, edgecolors="none", zorder=5)
        ax.scatter(pts2[:,0], pts2[:,1], s=3, c="#dfa96e", alpha=0.45, edgecolors="none", zorder=5)
        # small axes
        line(ax, x+w*0.5, y+h*0.18, x+w*0.5, y+h*0.78, lw=0.6, color="#333", z=6)
        line(ax, x+w*0.22, y+h*0.45, x+w*0.78, y+h*0.45, lw=0.6, color="#333", z=6)
        label(ax, x+w*0.53, y+h*0.78, "y", size=6)
        label(ax, x+w*0.79, y+h*0.41, "x", size=6)

# ----------------------------
# Panel backgrounds and separators
# ----------------------------
panels = [(0, 50), (50, 40), (90, 52), (142, 38)]
for px, pw in panels:
    rect(ax, px, 0, pw, H, fc=COL["panel"], ec="none", alpha=0.70, z=0)
for x in [50, 90, 142]:
    line(ax, x, 0, x, H, color=COL["separator"], lw=1.4, z=1)

# ----------------------------
# Panel 1: Input Sequential Data
# ----------------------------
label(ax, 25, 76, "1. Input Sequential Data", size=14, weight="bold")
label(ax, 25, 71.5, r"$x_u=[x_1,x_2,\ldots,x_T]\rightarrow y$", size=13, weight="bold")
label(ax, 5, 66.5, "User", size=8, weight="bold")
label(ax, 23, 66.5, "Item Interaction Sequence (chronological)", size=7.4, weight="bold")
label(ax, 46.5, 66.5, "Target\n$y$", size=8, weight="bold")

draw_sequence(ax, 57.5, "F", "18-24", ["#df805f", "#f2cf63", "#9cabb4", "#6a7f93"], skew_indices=(0,1))
line(ax, 3, 52.4, 47, 52.4, color="#c7c7c7", lw=0.8, ls="--")
draw_sequence(ax, 41.8, "M", "25-34", ["#5c82b5", "#7f8a95", "#626d77", "#9bb4d8"], skew_indices=(0,))
line(ax, 3, 36.8, 47, 36.8, color="#c7c7c7", lw=0.8, ls="--")
draw_sequence(ax, 26.0, "F", "35+", ["#6e94c4", "#d36b72", "#e7b35a", "#cb5b4d"], skew_indices=(1,))

# legend: demographic skew
ax.add_patch(Ellipse((16.7, 17.7), 2.6, 2.6, facecolor="#f6a23a", edgecolor="none", alpha=0.23, zorder=3))
ax.add_patch(Ellipse((16.7, 17.7), 1.35, 1.35, facecolor="#f6a23a", edgecolor="none", alpha=0.18, zorder=3))
label(ax, 24.6, 17.7, "demographic skew", size=8.5, color="#c96f2a", style="italic")
# sensitive note
rounded_box(ax, 3, 4.2, 44, 8.7, fc="#fbfbfb", ec="#aeb4bb", lw=1.0, r=1.0, ls="--", z=2)
# simple lock
rounded_box(ax, 5.0, 6.5, 3.2, 3.1, fc="#a5a9ad", ec="none", r=0.5, z=4)
ax.add_patch(Arc((6.6, 9.5), 2.2, 2.6, theta1=0, theta2=180, color="#a5a9ad", lw=2.0, zorder=4))
label(ax, 23.6, 8.6, "Sensitive attributes used only for train-time statistics\nand fairness evaluation, not for inference.",
      size=7.5, style="italic", color="#5c5c5c")

# ----------------------------
# Panel 2: Controlled Augmentation
# ----------------------------
label(ax, 70, 76, "Fairness-Aware\nControlled Augmentation", size=14, weight="bold")
rounded_box(ax, 52.0, 10.3, 36.0, 61.5, fc=COL["green"], ec=COL["green_edge"], lw=1.2, r=1.4, alpha=1, z=1)
# submodules
rounded_box(ax, 53.2, 55.2, 33.6, 15.2, fc="#eef9ef", ec=COL["green_edge"], lw=1.0, r=1.0, z=2)
label(ax, 70, 68.2, "Item Co-occurrence Neighborhood", size=10.5, weight="bold")
draw_cooccurrence_graph(ax, 70, 62.2, radius=6.2)
label(ax, 70, 55.8, r"$\mathcal{N}(i)=\mathrm{TopK}_{j\ne i} C_{ij}$", size=10)

rounded_box(ax, 53.2, 36.1, 33.6, 17.0, fc="#eef9ef", ec=COL["green_edge"], lw=1.0, r=1.0, z=2)
label(ax, 70, 51.2, "Train-Only Item Demographic Skew", size=10.4, weight="bold")
mini_barchart(ax, 56.6, 42.4, vals=(0.80,0.48,0.30,0.15))
mini_barchart(ax, 65.5, 42.4, vals=(0.25,0.70,0.38,0.22))
label(ax, 74.4, 44.4, r"$\cdots$", size=12)
mini_barchart(ax, 77.0, 42.4, vals=(0.62,0.83,0.45,0.26))
rounded_box(ax, 80.7, 43.3, 5.0, 5.2, fc="#f0f0f0", ec="#8f949e", lw=0.9, r=0.8, z=5)
label(ax, 83.2, 46.4, "skew", size=8)
label(ax, 83.2, 44.5, "score", size=8)
label(ax, 70, 38.9, r"$b(i)=\mathrm{mean}_{A}[\max_g p(g|i,A)-\min_g p(g|i,A)]$", size=8.3)

rounded_box(ax, 53.2, 13.0, 33.6, 21.0, fc="#eef9ef", ec=COL["green_edge"], lw=1.0, r=1.0, z=2)
label(ax, 70, 31.9, "Exponential Controlled Sampling", size=10.5, weight="bold")
# high skew -> lower skew
rounded_box(ax, 59.0, 22.0, 5.5, 5.5, fc="#fff7ef", ec="#e2a25c", lw=1.1, r=0.8, z=4)
# graph-like icon in high skew item
for (a,b) in [(60.5,24.0),(62.5,25.1),(61.4,26.2),(63.2,23.4)]:
    ax.add_patch(Circle((a,b), 0.35, fc="#f4c58d", ec="#bd7a36", lw=0.7, zorder=6))
line(ax, 60.5,24.0,62.5,25.1,color="#bd7a36", lw=0.8, z=5)
line(ax, 62.5,25.1,61.4,26.2,color="#bd7a36", lw=0.8, z=5)
line(ax, 62.5,25.1,63.2,23.4,color="#bd7a36", lw=0.8, z=5)
label(ax, 61.75, 20.2, "high-skew\nitem", size=8.2)
# gradient-ish arrow as thick line + head
line(ax, 67.1, 24.7, 73.8, 24.7, color="#d7c6aa", lw=4.5, alpha=0.5, z=4)
arrow(ax, 67.2, 24.7, 75.0, 24.7, color="#74a7cb", lw=1.7, mutation=16, z=5)
rounded_box(ax, 76.3, 22.0, 5.5, 5.5, fc="#eef6ff", ec="#79a8d0", lw=1.1, r=0.8, z=4)
rect(ax, 77.6, 23.3, 2.8, 2.1, fc="#cfe4f6", ec="#6f93ad", lw=0.7, z=5)
arrow(ax, 77.9, 24.4, 79.7, 24.4, color="#6f93ad", lw=0.7, mutation=5, z=6)
label(ax, 79.0, 20.2, "lower-skew\nneighbor", size=8.2)
label(ax, 70, 17.4, r"$P(j|i)\propto \exp(\epsilon\,U(i\rightarrow j))$", size=9.6)
label(ax, 70, 15.2, r"$U(i\rightarrow j)=\alpha s(i,j)+\beta\Delta_b$", size=8.2)
label(ax, 70, 13.6, r"$\Delta_b=\max(0,b(i)-b(j))$", size=7.7)
rounded_box(ax, 53.2, 5.5, 33.6, 3.8, fc="white", ec="#aeb4bb", lw=0.9, r=0.7, ls="--", z=2)
label(ax, 70, 7.4, "local replacement, preference-preserving,\nlower demographic skew", size=7.7, weight="bold")

# x and x' arrows to panel 3
label(ax, 93.0, 55.8, r"$x$", size=14, weight="bold")
arrow(ax, 88.2, 55.5, 99.0, 55.5, lw=1.1, mutation=10)
label(ax, 92.2, 30.2, r"$x'$", size=14, weight="bold")
label(ax, 91.8, 26.9, r"$x'=\mathcal{A}_{\mathrm{fair}}(x)$", size=8.2)
arrow(ax, 88.2, 30.0, 99.0, 30.0, lw=1.1, mutation=10)

# ----------------------------
# Panel 3: Shared encoder and non-contrastive learning
# ----------------------------
label(ax, 116, 76, "Shared Sequential Encoder\n& Non-Contrastive Learning", size=14, weight="bold")
line(ax, 109, 6, 109, 72, color="#bdbdbd", lw=0.8, ls=":", z=1)
# tied weights curly brace approximation
label(ax, 102.0, 42.8, "{", size=42, rotation=0)
label(ax, 104.0, 42.8, "tied-weights", size=8.5, ha="left")
# z boxes
rounded_box(ax, 99.2, 52.6, 11.0, 5.6, fc=COL["blue"], ec=COL["blue_edge"], lw=1.0, r=0.8, z=3)
label(ax, 104.7, 55.3, r"$z=f(x)$", size=12, weight="bold")
rounded_box(ax, 99.2, 26.8, 11.0, 5.6, fc=COL["blue"], ec=COL["blue_edge"], lw=1.0, r=0.8, z=3)
label(ax, 104.7, 29.5, r"$z'=f(x')$", size=12, weight="bold")
# encoder
rounded_box(ax, 106.6, 22.4, 15.8, 36.0, fc=COL["purple"], ec=COL["purple_edge"], lw=1.2, r=1.1, z=2)
label(ax, 114.5, 48.5, "Shared\nSequential\nEncoder:", size=10.5, weight="bold")
label(ax, 114.5, 38.2, "SASRec /\nGRU4Rec /\nBERT4Rec /\nCaser", size=12, weight="bold")
arrow(ax, 110.2, 55.5, 106.6, 55.5, lw=0.9, mutation=8)
arrow(ax, 110.2, 29.8, 106.6, 29.8, lw=0.9, mutation=8)
# outputs to regularizers and head
arrow(ax, 122.4, 55.5, 125.6, 55.5, lw=1.1, mutation=9)
arrow(ax, 122.4, 42.0, 125.6, 42.0, lw=1.1, mutation=9)
arrow(ax, 122.4, 30.0, 125.6, 30.0, lw=1.1, mutation=9)
# regularizers
rounded_box(ax, 125.5, 49.0, 14.8, 17.2, fc=COL["beige"], ec=COL["beige_edge"], lw=1.0, r=1.0, z=2)
label(ax, 132.9, 63.0, r"Alignment", size=10.2, weight="bold")
label(ax, 132.9, 60.8, r"$(\mathcal{L}_{align}=||\hat z-\hat z'||^2)$", size=7.8)
scatter_cloud(ax, 126.0, 49.7, 13.8, 10.8, kind="align", seed=1)

rounded_box(ax, 125.5, 29.5, 14.8, 16.8, fc=COL["beige"], ec=COL["beige_edge"], lw=1.0, r=1.0, z=2)
label(ax, 132.9, 43.5, r"Variance $(\mathcal{L}_{var})$", size=10.0, weight="bold")
scatter_cloud(ax, 126.0, 30.2, 13.8, 10.7, kind="var", seed=2)

rounded_box(ax, 125.5, 10.0, 14.8, 17.0, fc=COL["beige"], ec=COL["beige_edge"], lw=1.0, r=1.0, z=2)
label(ax, 132.9, 24.1, r"Covariance $(\mathcal{L}_{cov})$", size=9.7, weight="bold")
scatter_cloud(ax, 126.0, 10.4, 13.8, 11.7, kind="cov", seed=3)
rounded_box(ax, 121.6, 4.0, 17.4, 4.7, fc="white", ec="#aeb4bb", lw=0.9, r=0.7, ls="--", z=2)
label(ax, 130.3, 6.3, "negative-sampling-free\nrepresentation regularization", size=7.2, weight="bold")

# ----------------------------
# Panel 4: Optimization & inference
# ----------------------------
label(ax, 161, 76, "Model Optimization\n& Inference", size=14, weight="bold")
rounded_box(ax, 147.6, 66.5, 29.8, 8.0, fc=COL["beige"], ec=COL["beige_edge"], lw=1.1, r=0.9, z=2)
label(ax, 162.5, 70.6, r"$\mathcal{L}=\mathcal{L}_{rec}+\eta\mathcal{L}_{aug}+\lambda_{align}\mathcal{L}_{align}$", size=10.5)
label(ax, 162.5, 68.3, r"$+\lambda_{var}\mathcal{L}_{var}+\lambda_{cov}\mathcal{L}_{cov}$", size=10.5)
# arrows from regularizer area to total objective
arrow(ax, 140.4, 57.0, 147.3, 69.0, lw=0.9, mutation=8)
arrow(ax, 140.4, 37.8, 147.3, 69.0, lw=0.9, mutation=8)
arrow(ax, 140.4, 18.0, 147.3, 68.2, lw=0.9, mutation=8)

# recommendation head
rounded_box(ax, 148.4, 18.0, 18.8, 44.5, fc=COL["blue"], ec="#9eb8d0", lw=1.1, r=1.0, z=2)
label(ax, 157.8, 58.9, "recommendation\nhead", size=10)
rounded_box(ax, 151.2, 43.7, 13.4, 8.6, fc=COL["beige"], ec=COL["beige_edge"], lw=1.0, r=0.8, z=3)
label(ax, 157.9, 48.2, r"$\mathcal{L}_{rec}$" + "\n(from original\nview)", size=8.8)
label(ax, 157.9, 36.6, r"$g(\cdot)$", size=18, weight="bold")
rounded_box(ax, 151.2, 22.4, 13.4, 9.2, fc=COL["beige"], ec=COL["beige_edge"], lw=1.0, r=0.8, z=3)
label(ax, 157.9, 27.1, r"$\mathcal{L}_{aug}$" + "\n(from augmentation\nview)", size=8.7)
# arrows into recommendation head
arrow(ax, 140.5, 55.6, 148.4, 49.2, lw=1.0, mutation=8)
arrow(ax, 140.5, 30.0, 148.4, 26.8, lw=1.0, mutation=8)
# dashed inference divider
line(ax, 167.8, 11, 167.8, 63.4, color="#bbbbbb", lw=0.9, ls=":")
# balanced group exposure small dots
label(ax, 173.7, 59.5, "balanced\ngroup exposure", size=8.5)
for dx, dy, c in [(-3,0,"#a7c7e7"),(-1,0,"#e8a4a4"),(1,0,"#d6b278"),(3,0,"#8bc78d"),(-2,-1.8,"#b0b7d8"),(2,-1.8,"#bddbbd")]:
    ax.add_patch(Circle((173.7+dx, 55.8+dy), 0.8, fc=c, ec="white", lw=0.7, zorder=4))
label(ax, 173.7, 52.0, "utility preserved\ngroup gaps reduced", size=8.6)
label(ax, 174.0, 47.0, "Top-K", size=11, weight="bold")
# Top-K list
rounded_box(ax, 170.5, 26.2, 7.0, 18.4, fc="#e8f5e8", ec=COL["green_edge"], lw=1.1, r=0.9, z=3)
for i, c in enumerate(["#d9ecfb", "#e8f5e8", "#d9ecfb", "#d9ecfb"]):
    yy = 40.1 - i*4.1
    rounded_box(ax, 171.4, yy, 5.2, 3.3, fc=c, ec="#95b4a6", lw=0.8, r=0.5, z=4)
    # simple top-k icons
    if i == 0:
        for a,b in [(172.3,yy+1.0),(174.0,yy+2.1),(175.3,yy+1.1)]:
            ax.add_patch(Circle((a,b), 0.25, fc="#8db6d9", ec="#6e9bbd", lw=0.4, zorder=5))
        line(ax, 172.3,yy+1.0,174.0,yy+2.1,color="#6e9bbd",lw=0.5,z=5)
        line(ax, 174.0,yy+2.1,175.3,yy+1.1,color="#6e9bbd",lw=0.5,z=5)
    elif i == 1:
        ax.add_patch(Polygon([[173.0,yy+0.6],[175.0,yy+0.6],[175.0,yy+2.2],[174.0,yy+2.7],[173.0,yy+2.2]],
                             fc="#8bc78d", ec="#709d70", lw=0.5, zorder=5))
    else:
        rect(ax, 173.1, yy+0.6, 2.0, 2.0, fc="#a7c7e7", ec="#6e9bbd", lw=0.5, z=5)
        line(ax, 173.5, yy+2.1, 174.6, yy+2.1, color="#6e9bbd", lw=0.45, z=6)
        line(ax, 173.5, yy+1.6, 174.6, yy+1.6, color="#6e9bbd", lw=0.45, z=6)
# arrow g to top-k
arrow(ax, 167.2, 37.0, 170.3, 37.0, lw=1.2, mutation=10)
# latent space at bottom
rng = np.random.default_rng(10)
for mean, c in [((171.3, 13.2), "#d393b3"), ((176.0, 15.0), "#8bc78d")]:
    pts = rng.normal(mean, [1.7, 1.3], size=(70, 2))
    ax.scatter(pts[:,0], pts[:,1], s=7, c=c, alpha=0.45, edgecolors="white", linewidths=0.15, zorder=5)
arrow(ax, 174.0, 26.2, 174.0, 19.5, color="#555", lw=1.0, mutation=9)

# Save outputs
os.makedirs('figures', exist_ok=True)
os.makedirs('els-cas-templates', exist_ok=True)
plt.savefig('figures/fair_ncl_mechanism.png', dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.savefig('figures/fair_ncl_mechanism.svg', bbox_inches='tight', pad_inches=0.08)
plt.savefig('els-cas-templates/Fair-NCL.png', dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.close(fig)
print('Saved figures/fair_ncl_mechanism.png, figures/fair_ncl_mechanism.svg, and els-cas-templates/Fair-NCL.png')
