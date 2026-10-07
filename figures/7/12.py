import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Data
# ============================================================

models = [
    "DeepSeek-V4-Flash",
    "Doubao Web",
    "K2.6",
    "Qwen3.6-Flash"
]

categories = [
    "Illegal & Criminal\nActivities",
    "Violence &\nPhysical Harm",
    "Hate, Discrimination\n& Harassment",
    "Self-Harm &\nSuicide",
    "Misinformation &\nManipulation",
    "Cybersecurity\nAttacks",
    "Overall"
]

# Category-level ASR@1 (%)
data = np.array([
    [85.00, 90.00, 95.00, 45.00, 85.00, 100.00, 83.33],
    [95.00, 95.00, 100.00, 65.00, 90.00, 100.00, 90.83],
    [75.00, 80.00, 95.00, 40.00, 85.00, 100.00, 79.17],
    [70.00, 75.00, 85.00, 40.00, 75.00, 100.00, 74.17]
])


# ============================================================
# 2. Font settings
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(15.5, 6.2))


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
# 5. Axis labels
# ============================================================

ax.set_xticks(np.arange(len(categories)))
ax.set_yticks(np.arange(len(models)))

ax.set_xticklabels(
    categories,
    fontsize=11.5,
    fontfamily="Times New Roman"
)

ax.set_yticklabels(
    models,
    fontsize=13,
    fontfamily="Times New Roman"
)

ax.tick_params(
    axis="x",
    bottom=False,
    top=False,
    pad=10
)

ax.tick_params(
    axis="y",
    left=False,
    right=False,
    pad=8
)


# ============================================================
# 6. Cell boundaries
# ============================================================

ax.set_xticks(
    np.arange(-0.5, len(categories), 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, len(models), 1),
    minor=True
)

ax.grid(
    which="minor",
    linewidth=0.8,
    alpha=0.65
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

        # White text on dark cells,
        # dark text on light cells
        text_color = "white" if value >= 72 else "black"

        ax.text(
            j,
            i,
            f"{value:.2f}%",
            ha="center",
            va="center",
            fontsize=12,
            fontfamily="Times New Roman",
            color=text_color
        )


# ============================================================
# 8. Emphasize the Overall column
# ============================================================

# Vertical separator before Overall
ax.axvline(
    x=5.5,
    linewidth=1.4,
    color="black",
    alpha=0.65
)


# ============================================================
# 9. Colorbar
# ============================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.030,
    pad=0.025
)

cbar.set_ticks(
    [0, 20, 40, 60, 80, 100]
)

cbar.set_ticklabels(
    ["0%", "20%", "40%", "60%", "80%", "100%"]
)

cbar.ax.tick_params(
    labelsize=11
)

for label in cbar.ax.get_yticklabels():
    label.set_fontfamily("Times New Roman")

cbar.outline.set_linewidth(0.9)


# ============================================================
# 10. Remove outer spines
# ============================================================

for spine in ax.spines.values():
    spine.set_visible(False)


# ============================================================
# 11. Layout
# ============================================================

plt.subplots_adjust(
    left=0.15,
    right=0.94,
    top=0.97,
    bottom=0.22
)


# ============================================================
# 12. Save
# ============================================================

plt.savefig(
    "Fig12_Web_Category_ASR_Heatmap_EN.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "Fig12_Web_Category_ASR_Heatmap_EN.svg",
    bbox_inches="tight"
)

plt.savefig(
    "Fig12_Web_Category_ASR_Heatmap_EN.pdf",
    bbox_inches="tight"
)

plt.show()