import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ============================================================
# 1. Basic settings
# ============================================================

models = [
    "DeepSeek",
    "Doubao",
    "Gemini",
    "GPT",
    "Kimi",
    "Qwen"
]

# Difference matrix:
# row model - column model
diff = np.array([
    [  0.00, -68.52, -66.67, -37.04, -51.85, -40.74],
    [ 68.52,   0.00,   1.85,  31.48,  16.67,  27.78],
    [ 66.67,  -1.85,   0.00,  29.63,  14.81,  25.93],
    [ 37.04, -31.48, -29.63,   0.00, -14.81,  -3.70],
    [ 51.85, -16.67, -14.81,  14.81,   0.00,  11.11],
    [ 40.74, -27.78, -25.93,   3.70, -11.11,   0.00]
])

# Adjusted p-values as displayed in the original figure
p_text = np.array([
    ["—",      "<.001", "<.001", "<.001", "<.001", "<.001"],
    ["<.001",  "—",     "1.000", "0.002", "0.211", "0.013"],
    ["<.001",  "1.000", "—",     "0.013", "0.481", "0.088"],
    ["<.001",  "0.002", "0.013", "—",     "0.606", "1.000"],
    ["<.001",  "0.211", "0.481", "0.606", "—",     "0.790"],
    ["<.001",  "0.013", "0.088", "1.000", "0.790", "—"]
])

# Significance:
# -1 = significantly lower
#  0 = not significant
# +1 = significantly higher

sig = np.zeros((6, 6), dtype=int)

# DeepSeek row
sig[0, 1] = -1
sig[0, 2] = -1
sig[0, 3] = -1
sig[0, 4] = -1
sig[0, 5] = -1

# Doubao row
sig[1, 0] = 1
sig[1, 3] = 1
sig[1, 5] = 1

# Gemini row
sig[2, 0] = 1
sig[2, 3] = 1

# GPT row
sig[3, 0] = 1
sig[3, 1] = -1
sig[3, 2] = -1

# Kimi row
sig[4, 0] = 1

# Qwen row
sig[5, 0] = 1
sig[5, 1] = -1


# ============================================================
# 2. Font
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. Colors
# ============================================================

cmap = ListedColormap([
    "#2C7FB8",   # significantly lower
    "#F2F2F2",   # not significant
    "#FF7F0E"    # significantly higher
])

color_matrix = sig + 1


# ============================================================
# 4. Figure
# ============================================================

fig, ax = plt.subplots(figsize=(8.3, 8.0))

ax.imshow(
    color_matrix,
    cmap=cmap,
    vmin=0,
    vmax=2,
    aspect="equal"
)


# ============================================================
# 5. Axes
# ============================================================

ax.set_xticks(np.arange(6))
ax.set_yticks(np.arange(6))

ax.set_xticklabels(
    models,
    rotation=40,
    ha="right",
    fontsize=12
)

ax.set_yticklabels(
    models,
    fontsize=12
)

ax.set_xlabel(
    "Comparison Model",
    fontsize=13,
    labelpad=8
)

ax.set_ylabel(
    "Reference Model",
    fontsize=13,
    labelpad=10
)


# ============================================================
# 6. Cell boundaries
# ============================================================

ax.set_xticks(
    np.arange(-0.5, 6, 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, 6, 1),
    minor=True
)

ax.grid(
    which="minor",
    linewidth=0.8,
    color="white",
    alpha=0.75
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)


# ============================================================
# 7. Cell annotations
# ============================================================

for i in range(6):
    for j in range(6):

        if i == j:
            ax.text(
                j,
                i,
                "—",
                ha="center",
                va="center",
                fontsize=14
            )
            continue

        value = diff[i, j]
        p = p_text[i, j]

        star = "*" if sig[i, j] != 0 else ""

        value_text = f"{value:+.2f}{star}"

        text_color = "white" if sig[i, j] != 0 else "black"

        ax.text(
            j,
            i,
            f"{value_text}\n({p})",
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=text_color,
            linespacing=1.15
        )


# ============================================================
# 8. Legend
# ============================================================

legend_handles = [
    Patch(
        facecolor="#2C7FB8",
        label="Significantly Lower"
    ),
    Patch(
        facecolor="#F2F2F2",
        label="Not Significant"
    ),
    Patch(
        facecolor="#FF7F0E",
        label="Significantly Higher"
    )
]

ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.17),
    ncol=3,
    frameon=False,
    fontsize=11,
    columnspacing=1.7,
    handlelength=1.8
)


# ============================================================
# 9. Subfigure title at bottom
# ============================================================

ax.text(
    0.5,
    -0.29,
    "(b) JBB-Behaviors",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=15,
    fontweight="bold",
    fontfamily="Times New Roman"
)


# ============================================================
# 10. Layout
# ============================================================

plt.subplots_adjust(
    left=0.17,
    right=0.98,
    top=0.98,
    bottom=0.31
)


# ============================================================
# 11. Save
# ============================================================

plt.savefig(
    "Fig13b_JBB_significance_matrix_EN.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "Fig13b_JBB_significance_matrix_EN.svg",
    bbox_inches="tight"
)

plt.savefig(
    "Fig13b_JBB_significance_matrix_EN.pdf",
    bbox_inches="tight"
)

plt.show()