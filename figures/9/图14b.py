import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. Data
# ============================================================

categories = [
    "Harassment /\nDiscrimination",
    "Malware /\nHacking",
    "Physical\nHarm",
    "Economic\nHarm",
    "Fraud /\nMisinformation",
    "Sexual / Adult\nContent",
    "Privacy\nViolations",
    "Professional\nAdvice",
    "Government\nDecision-Making"
]

counts = [6, 6, 6, 6, 6, 6, 6, 6, 6]

x = np.arange(len(categories))


# ============================================================
# 2. Font settings
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))


# ============================================================
# 4. Bar chart
# ============================================================

bars = ax.bar(
    x,
    counts,
    width=0.66,
    edgecolor="black",
    linewidth=0.8
)


# ============================================================
# 5. Y-axis
# ============================================================

ax.set_ylim(0, 8)

ax.set_yticks([0, 2, 4, 6, 8])

ax.set_ylabel(
    "Number of Samples",
    fontsize=16,
    fontfamily="Times New Roman"
)

ax.tick_params(
    axis="y",
    labelsize=13,
    direction="out",
    length=5,
    width=1
)


# ============================================================
# 6. X-axis
# ============================================================

ax.set_xticks(x)

ax.set_xticklabels(
    categories,
    fontsize=10.5,
    fontfamily="Times New Roman"
)

ax.tick_params(
    axis="x",
    direction="out",
    length=5,
    width=1
)


# ============================================================
# 7. Horizontal grid lines
# ============================================================

ax.grid(
    axis="y",
    linestyle="-",
    linewidth=0.7,
    alpha=0.25
)

ax.set_axisbelow(True)


# ============================================================
# 8. Axis style
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)


# ============================================================
# 9. Values above bars
# ============================================================

for bar, value in zip(bars, counts):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.14,
        str(value),
        ha="center",
        va="bottom",
        fontsize=14,
        fontfamily="Times New Roman",
        fontweight="bold"
    )


# ============================================================
# 10. Subfigure title below the plot
# ============================================================

ax.text(
    0.5,
    -0.20,
    "(b) JailbreakBench ($N = 54$)",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=16,
    fontfamily="Times New Roman",
    fontweight="bold"
)


# ============================================================
# 11. Layout
# ============================================================

plt.subplots_adjust(
    left=0.11,
    right=0.985,
    top=0.97,
    bottom=0.27
)


# ============================================================
# 12. Save
# ============================================================

plt.savefig(
    "Fig14b_JailbreakBench_category_distribution_EN.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "Fig14b_JailbreakBench_category_distribution_EN.svg",
    bbox_inches="tight"
)

plt.savefig(
    "Fig14b_JailbreakBench_category_distribution_EN.pdf",
    bbox_inches="tight"
)

plt.show()