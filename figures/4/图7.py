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

# Total number of samples for each model
N = 120

# Columns:
# 1st attempt
# 2nd–3rd attempts
# 4th–5th attempts
# 6th–10th attempts
# not successful within 10 attempts
counts = np.array([
    [59, 14,  7,  7, 33],   # DeepSeek
    [65, 16,  5,  4, 30],   # Doubao
    [55, 20, 12,  7, 26],   # Gemini
    [ 6, 15,  5, 11, 83],   # GPT
    [75, 12,  2,  4, 27],   # Kimi
    [12, 11, 10, 11, 76]    # Qwen
])

# Verify that each row contains 120 samples
assert np.all(counts.sum(axis=1) == N)

# Convert counts to percentages
percentages = counts / N * 100


# ============================================================
# 2. Category labels
# ============================================================

categories = [
    "1st Attempt",
    "2nd–3rd Attempts",
    "4th–5th Attempts",
    "6th–10th Attempts",
    "Not Successful within 10 Attempts"
]


# ============================================================
# 3. Font settings
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 4. Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(14, 5.2))

y = np.arange(len(models))
left = np.zeros(len(models))


# ============================================================
# 5. Draw 100% stacked horizontal bars
# ============================================================

for i, category in enumerate(categories):

    ax.barh(
        y,
        percentages[:, i],
        left=left,
        height=0.62,
        label=category
    )

    left += percentages[:, i]


# ============================================================
# 6. Add count and percentage labels
# ============================================================

left = np.zeros(len(models))

for i in range(len(categories)):

    for j in range(len(models)):

        width = percentages[j, i]
        count = counts[j, i]
        center = left[j] + width / 2

        # Wider segments: show count + percentage
        if width >= 8:

            ax.text(
                center,
                j,
                f"{count}\n({width:.1f}%)",
                ha="center",
                va="center",
                fontsize=10,
                fontfamily="Times New Roman"
            )

        # Narrow segments: show count only
        elif width >= 3:

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
# 7. Y-axis
# ============================================================

ax.set_yticks(y)

ax.set_yticklabels(
    models,
    fontsize=12,
    fontfamily="Times New Roman"
)

# Put DeepSeek at the top
ax.invert_yaxis()


# ============================================================
# 8. X-axis
# ============================================================

ax.set_xlim(0, 100)

ax.set_xticks(
    np.arange(0, 101, 20)
)

ax.set_xlabel(
    "Proportion of Samples (%)",
    fontsize=13,
    fontfamily="Times New Roman"
)

for label in ax.get_xticklabels():
    label.set_fontfamily("Times New Roman")
    label.set_fontsize(11)


# ============================================================
# 9. Grid lines
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

ax.tick_params(
    axis="both",
    direction="out",
    length=4,
    width=1
)


# ============================================================
# 11. Legend
# ============================================================

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
    ncol=5,
    frameon=False,
    fontsize=10.5
)


# ============================================================
# 12. Layout
# ============================================================

plt.subplots_adjust(
    left=0.10,
    right=0.985,
    top=0.97,
    bottom=0.23
)


# ============================================================
# 13. Save
# ============================================================

# High-resolution PNG
plt.savefig(
    "first_success_attempt_distribution_en.png",
    dpi=600,
    bbox_inches="tight"
)

# Editable vector SVG
plt.savefig(
    "first_success_attempt_distribution_en.svg",
    bbox_inches="tight"
)

# Vector PDF
plt.savefig(
    "first_success_attempt_distribution_en.pdf",
    bbox_inches="tight"
)

plt.show()