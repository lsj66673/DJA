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

categories = [
    "Harassment / Discrimination",
    "Malware / Hacking",
    "Physical Harm",
    "Economic Harm",
    "Fraud / Misinformation",
    "Sexual / Adult Content",
    "Privacy Violations",
    "Professional Advice",
    "Government Decision-Making"
]

# Human-reviewed ASR@10 (%)
data = np.array([
    [33.33,  83.33,  50.00, 16.67,  66.67, 66.67],
    [ 0.00,  83.33, 100.00, 66.67,  83.33, 66.67],
    [33.33,  83.33,  66.67, 50.00,  66.67, 50.00],
    [ 0.00, 100.00, 100.00, 66.67, 100.00, 83.33],
    [16.67, 100.00,  83.33, 83.33,  83.33, 66.67],
    [16.67,  33.33,  83.33, 16.67,  83.33, 33.33],
    [33.33, 100.00, 100.00, 83.33, 100.00, 66.67],
    [16.67, 100.00,  66.67, 33.33,  33.33, 50.00],
    [ 0.00,  83.33, 100.00, 66.67,   0.00, 33.33]
], dtype=float)


# ============================================================
# 2. Font settings
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(10.5, 8.0))


# ============================================================
# 4. Heatmap
# ============================================================

im = ax.imshow(
    data,
    cmap="Blues",
    vmin=0,
    vmax=100,
    aspect="auto"
)


# ============================================================
# 5. Axes
# ============================================================

ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(categories)))

ax.set_xticklabels(
    models,
    fontsize=12
)

ax.set_yticklabels(
    categories,
    fontsize=11.5
)

ax.set_xlabel(
    "Target Model",
    fontsize=13,
    labelpad=8
)


# ============================================================
# 6. Cell boundaries
# ============================================================

ax.set_xticks(
    np.arange(-0.5, len(models), 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, len(categories), 1),
    minor=True
)

ax.grid(
    which="minor",
    color="white",
    linewidth=0.8,
    alpha=0.75
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)


# ============================================================
# 7. Numerical annotations
# ============================================================

for i in range(data.shape[0]):
    for j in range(data.shape[1]):

        value = data[i, j]

        text_color = "white" if value >= 75 else "black"

        # Integer values -> no decimals
        # Other values -> two decimal places
        if np.isclose(value, round(value)):
            value_text = f"{value:.0f}"
        else:
            value_text = f"{value:.2f}"

        ax.text(
            j,
            i,
            value_text,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold" if value >= 80 else "normal",
            color=text_color
        )


# ============================================================
# 8. Colorbar
# ============================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.025
)

cbar.set_ticks([0, 20, 40, 60, 80, 100])

cbar.set_ticklabels([
    "0", "20", "40", "60", "80", "100"
])

cbar.set_label(
    "ASR@10 (%)",
    fontsize=12,
    labelpad=10
)

cbar.ax.tick_params(
    labelsize=10.5
)


# ============================================================
# 9. Remove outer spines
# ============================================================

for spine in ax.spines.values():
    spine.set_visible(False)


# ============================================================
# 10. Subfigure title BELOW the plot
# ============================================================

ax.text(
    0.5,
    -0.14,
    "(b) JBB-Behaviors",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=15,
    fontweight="bold",
    fontfamily="Times New Roman"
)


# ============================================================
# 11. Layout
# ============================================================

plt.subplots_adjust(
    left=0.27,
    right=0.92,
    top=0.97,
    bottom=0.18
)


# ============================================================
# 12. Save
# ============================================================

plt.savefig(
    "Fig8b_JBB_Category_ASR_EN.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "Fig8b_JBB_Category_ASR_EN.svg",
    bbox_inches="tight"
)

plt.savefig(
    "Fig8b_JBB_Category_ASR_EN.pdf",
    bbox_inches="tight"
)

plt.show()