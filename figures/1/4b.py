import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Data
# ============================================================

K = np.array([1, 3, 5, 10])
N = 54

success_counts = {
    "DeepSeek": [0, 0, 1, 2],
    "Doubao":   [28, 34, 36, 37],
    "Gemini":   [18, 31, 36, 39],
    "GPT":      [6, 9, 11, 11],
    "Kimi":     [13, 19, 20, 24],
    "Qwen":     [3, 5, 6, 9]
}

asr = {
    model: np.array(counts) / N * 100
    for model, counts in success_counts.items()
}


# ============================================================
# 2. Font
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. Line styles
# ============================================================

styles = {
    "DeepSeek": dict(marker="o", linestyle="-"),
    "Doubao":   dict(marker="s", linestyle="--"),
    "Gemini":   dict(marker="^", linestyle="-."),
    "GPT":      dict(marker="D", linestyle=":"),
    "Kimi":     dict(marker="v", linestyle="-"),
    "Qwen":     dict(marker="P", linestyle="--")
}


# ============================================================
# 4. Figure
# ============================================================

fig, ax = plt.subplots(figsize=(7.6, 6.3))


# ============================================================
# 5. Plot
# ============================================================

for model in success_counts:

    ax.plot(
        K,
        asr[model],
        label=model,
        linewidth=2.0,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.3,
        **styles[model]
    )


# ============================================================
# 6. Count labels
# ============================================================

for model, counts in success_counts.items():

    for x, y, count in zip(K, asr[model], counts):

        ax.text(
            x + 0.12,
            y + 2.0,
            f"{count}/54",
            fontsize=9.5
        )


# ============================================================
# 7. Axes
# ============================================================

ax.set_xlim(0.5, 11.2)
ax.set_ylim(0, 105)

ax.set_xticks(K)
ax.set_yticks(np.arange(0, 101, 20))

ax.set_xlabel(
    "Attempt Budget K",
    fontsize=13
)

ax.set_ylabel(
    "ASR (%)",
    fontsize=13
)

ax.tick_params(
    axis="both",
    labelsize=11
)


# ============================================================
# 8. Grid
# ============================================================

ax.grid(
    axis="y",
    linestyle="-",
    linewidth=0.7,
    alpha=0.22
)

ax.set_axisbelow(True)


# ============================================================
# 9. Spines
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)


# ============================================================
# 10. Legend
# ============================================================

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.17),
    ncol=3,
    frameon=False,
    fontsize=10,
    handlelength=2.8,
    columnspacing=1.4
)


# ============================================================
# 11. Subfigure title BELOW the plot
# ============================================================

ax.text(
    0.5,
    -0.34,
    "(b) Unified Judge (GPT-5.4-mini)",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=14,
    fontweight="bold"
)


# ============================================================
# 12. Layout
# ============================================================

plt.subplots_adjust(
    left=0.13,
    right=0.98,
    top=0.97,
    bottom=0.36
)


# ============================================================
# 13. Save
# ============================================================

plt.savefig(
    "JBB_ASR_Unified_Judge_EN.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "JBB_ASR_Unified_Judge_EN.svg",
    bbox_inches="tight"
)

plt.savefig(
    "JBB_ASR_Unified_Judge_EN.pdf",
    bbox_inches="tight"
)

plt.show()