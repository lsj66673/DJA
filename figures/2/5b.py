import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Data
# ============================================================

methods = [
    "Full DJF",
    "w/o\nDecomposition",
    "w/o Sequential\nInteraction",
    "w/o Final\nRefinement"
]

unified = [80.83, 4.17, 48.33, 92.50]
human   = [78.33, 3.33, 29.17, 42.50]

x = np.arange(len(methods))
width = 0.34


# ============================================================
# 2. Font
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. Figure
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 5.8))


# ============================================================
# 4. Bars
# ============================================================

bars1 = ax.bar(
    x - width / 2,
    unified,
    width,
    label="Unified Judge"
)

bars2 = ax.bar(
    x + width / 2,
    human,
    width,
    label="Human Review"
)


# ============================================================
# 5. Value labels
# ============================================================

for bars in [bars1, bars2]:
    for bar in bars:
        value = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10.5
        )


# ============================================================
# 6. Axes
# ============================================================

ax.set_ylim(0, 110)

ax.set_yticks(
    np.arange(0, 101, 20)
)

ax.set_ylabel(
    "ASR@10 (%)",
    fontsize=13
)

ax.set_xticks(x)

ax.set_xticklabels(
    methods,
    fontsize=10.5
)

ax.tick_params(
    axis="y",
    labelsize=11
)


# ============================================================
# 7. Grid
# ============================================================

ax.grid(
    axis="y",
    linestyle="-",
    linewidth=0.7,
    alpha=0.20
)

ax.set_axisbelow(True)


# ============================================================
# 8. Spines
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)


# ============================================================
# 9. Legend
# ============================================================

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    frameon=False,
    fontsize=10.5
)


# ============================================================
# 10. Subfigure title BELOW the plot
# ============================================================

ax.text(
    0.5,
    -0.31,
    "(b) Gemini",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=15,
    fontweight="bold"
)


# ============================================================
# 11. Layout
# ============================================================

plt.subplots_adjust(
    left=0.13,
    right=0.98,
    top=0.97,
    bottom=0.34
)


# ============================================================
# 12. Save
# ============================================================

plt.savefig(
    "Fig5b_Gemini_Ablation_EN.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "Fig5b_Gemini_Ablation_EN.svg",
    bbox_inches="tight"
)

plt.savefig(
    "Fig5b_Gemini_Ablation_EN.pdf",
    bbox_inches="tight"
)

plt.show()