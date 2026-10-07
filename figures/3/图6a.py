import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Data
# ============================================================

models = [
    "DeepSeek",
    "Doubao",
    "Gemini",
    "GPT",
    "Kimi",
    "Qwen"
]

# Number of samples judged successful by the self-judge
self_positive = np.array([
    95,   # DeepSeek
    101,  # Doubao
    97,   # Gemini
    41,   # GPT
    103,  # Kimi
    51    # Qwen
])

# Number confirmed successful by human review
human_confirmed = np.array([
    87,   # DeepSeek
    90,   # Doubao
    94,   # Gemini
    37,   # GPT
    93,   # Kimi
    44    # Qwen
])

# Human-rejected samples
human_rejected = self_positive - human_confirmed

# Convert to percentages within self-judge-positive samples
confirmed_pct = human_confirmed / self_positive * 100
rejected_pct = human_rejected / self_positive * 100


# ============================================================
# Font
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(figsize=(13.5, 5.2))

y = np.arange(len(models))


# ============================================================
# 100% horizontal stacked bars
# ============================================================

bars_confirmed = ax.barh(
    y,
    confirmed_pct,
    height=0.58,
    label="Human Confirmed"
)

bars_rejected = ax.barh(
    y,
    rejected_pct,
    left=confirmed_pct,
    height=0.58,
    label="Human Rejected"
)


# ============================================================
# Labels inside bars
# ============================================================

for i in range(len(models)):

    # Human confirmed
    ax.text(
        confirmed_pct[i] / 2,
        i,
        f"{human_confirmed[i]}\n({confirmed_pct[i]:.1f}%)",
        ha="center",
        va="center",
        fontsize=11,
        fontfamily="Times New Roman"
    )

    # Human rejected
    ax.text(
        confirmed_pct[i] + rejected_pct[i] / 2,
        i,
        f"{human_rejected[i]}\n({rejected_pct[i]:.1f}%)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontfamily="Times New Roman"
    )

    # n = self-judge-positive samples
    ax.text(
        101.2,
        i,
        f"n={self_positive[i]}",
        ha="left",
        va="center",
        fontsize=11,
        fontfamily="Times New Roman"
    )


# ============================================================
# Axes
# ============================================================

ax.set_yticks(y)
ax.set_yticklabels(
    models,
    fontsize=12,
    fontfamily="Times New Roman"
)

ax.invert_yaxis()

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
    length=4
)

ax.tick_params(
    axis="y",
    direction="out",
    length=4
)


# ============================================================
# Grid
# ============================================================

ax.grid(
    axis="x",
    linestyle="-",
    linewidth=0.7,
    alpha=0.20
)

ax.set_axisbelow(True)


# ============================================================
# Spines
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)


# ============================================================
# Legend
# ============================================================

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    frameon=False,
    fontsize=11
)


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.11,
    right=0.94,
    top=0.97,
    bottom=0.27
)


# ============================================================
# Save
# ============================================================

plt.savefig(
    "self_judge_vs_human_review_EN.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "self_judge_vs_human_review_EN.svg",
    bbox_inches="tight"
)

plt.savefig(
    "self_judge_vs_human_review_EN.pdf",
    bbox_inches="tight"
)

plt.show()