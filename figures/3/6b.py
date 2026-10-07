import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Data
# ============================================================

models = [
    "DeepSeek",
    "Doubao",
    "Gemini",
    "GPT",
    "Kimi",
    "Qwen"
]

# Columns:
# Both Successful
# Unified Judge Only
# Human Review Only
# Both Failed

counts = np.array([
    [41, 4, 46, 4],   # DeepSeek -- corrected
    [79, 7, 11, 4],   # Doubao
    [80, 3, 14, 0],   # Gemini
    [12, 2, 25, 2],   # GPT
    [55, 2, 38, 8],   # Kimi
    [15, 3, 29, 4]    # Qwen
])

categories = [
    "Both Successful",
    "Unified Judge Only",
    "Human Review Only",
    "Both Failed"
]

# Number of paired/self-judge-positive samples
n = counts.sum(axis=1)

# Convert counts to percentages
percentages = counts / n[:, None] * 100


# ============================================================
# 2. Font settings
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(13.5, 5.3))

y = np.arange(len(models))
left = np.zeros(len(models))


# ============================================================
# 4. Draw 100% stacked horizontal bars
# ============================================================

for i, category in enumerate(categories):

    ax.barh(
        y,
        percentages[:, i],
        left=left,
        height=0.60,
        label=category
    )

    left += percentages[:, i]


# ============================================================
# 5. Add count + percentage labels
# ============================================================

left = np.zeros(len(models))

for i in range(len(categories)):

    for j in range(len(models)):

        width = percentages[j, i]
        count = counts[j, i]

        # Do not label zero-width segments
        if count == 0:
            left[j] += width
            continue

        center = left[j] + width / 2

        # Large segments:
        # display count and percentage
        if width >= 8:

            ax.text(
                center,
                j,
                f"{count}\n({width:.1f}%)",
                ha="center",
                va="center",
                fontsize=10.5,
                fontfamily="Times New Roman"
            )

        # Small segments:
        # display count only
        elif width >= 2.5:

            ax.text(
                center,
                j,
                f"{count}",
                ha="center",
                va="center",
                fontsize=9.5,
                fontfamily="Times New Roman"
            )

        left[j] += width


# ============================================================
# 6. Add sample size on the right
# ============================================================

for i, total in enumerate(n):

    ax.text(
        101.2,
        i,
        f"n={total}",
        ha="left",
        va="center",
        fontsize=11,
        fontfamily="Times New Roman"
    )


# ============================================================
# 7. Y-axis
# ============================================================

ax.set_yticks(y)

ax.set_yticklabels(
    models,
    fontsize=12,
    fontfamily="Times New Roman"
)

ax.invert_yaxis()


# ============================================================
# 8. X-axis
# ============================================================

ax.set_xlim(0, 105)

ax.set_xticks(
    np.arange(0, 101, 20)
)

ax.set_xlabel(
    "Proportion of Self-Judge Positive Samples (%)",
    fontsize=13,
    fontfamily="Times New Roman",
    labelpad=8
)

ax.tick_params(
    axis="x",
    labelsize=11,
    direction="out",
    length=4,
    width=1
)

ax.tick_params(
    axis="y",
    direction="out",
    length=4,
    width=1
)


# ============================================================
# 9. Grid
# ============================================================

ax.grid(
    axis="x",
    linestyle="-",
    linewidth=0.7,
    alpha=0.20
)

ax.set_axisbelow(True)


# ============================================================
# 10. Axis style
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)


# ============================================================
# 11. Legend
# ============================================================

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=4,
    frameon=False,
    fontsize=10.5,
    handlelength=1.8,
    columnspacing=1.8
)


# ============================================================
# 12. Layout
# ============================================================

plt.subplots_adjust(
    left=0.11,
    right=0.94,
    top=0.97,
    bottom=0.28
)


# ============================================================
# 13. Save
# ============================================================

plt.savefig(
    "unified_judge_vs_human_review_EN.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "unified_judge_vs_human_review_EN.svg",
    bbox_inches="tight"
)

plt.savefig(
    "unified_judge_vs_human_review_EN.pdf",
    bbox_inches="tight"
)

plt.show()