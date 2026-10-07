import numpy as np
import matplotlib.pyplot as plt

# Attempt budgets
K = np.array([1, 3, 5, 10])

# DeepSeek — Human Review
# AdvBench, 20 samples per category
cybersecurity = [70, 85, 90, 100]
self_harm     = [20, 30, 35, 35]
hate_harass   = [60, 70, 70, 85]

plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots(figsize=(6.6, 5.5))

# Cybersecurity Attacks
ax.plot(
    K, cybersecurity,
    marker="P",
    linestyle="--",
    linewidth=2.2,
    markersize=7,
    label="Cybersecurity Attacks"
)

# Self-Harm & Suicide
ax.plot(
    K, self_harm,
    marker="D",
    linestyle=":",
    linewidth=2.2,
    markersize=7,
    label="Self-Harm & Suicide"
)

# Hate, Discrimination & Harassment
ax.plot(
    K, hate_harass,
    marker="^",
    linestyle="-.",
    linewidth=2.2,
    markersize=7,
    label="Hate, Discrimination & Harassment"
)

# Axes
ax.set_xlim(0.6, 10.4)
ax.set_ylim(0, 105)

ax.set_xticks(K)
ax.set_yticks(np.arange(0, 101, 20))

ax.set_xlabel("Attempt Budget K", fontsize=13)
ax.set_ylabel("ASR (%)", fontsize=13)

ax.tick_params(axis="both", labelsize=11)

# Grid
ax.grid(
    axis="both",
    alpha=0.20,
    linewidth=0.7
)
ax.set_axisbelow(True)

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=1,
    frameon=False,
    fontsize=10
)

# Subfigure title at the bottom
ax.text(
    0.5, -0.43,
    "(a) DeepSeek",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=14,
    fontweight="bold"
)

plt.subplots_adjust(
    left=0.14,
    right=0.98,
    top=0.97,
    bottom=0.43
)

# High-resolution outputs
plt.savefig(
    "Fig9a_DeepSeek.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "Fig9a_DeepSeek.svg",
    bbox_inches="tight"
)

plt.savefig(
    "Fig9a_DeepSeek.pdf",
    bbox_inches="tight"
)

plt.show()