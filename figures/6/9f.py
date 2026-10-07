import numpy as np
import matplotlib.pyplot as plt

K = np.array([1, 3, 5, 10])

cybersecurity = [5, 5, 15, 35]
self_harm     = [5, 15, 25, 30]
hate_harass   = [5, 15, 25, 45]

plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots(figsize=(6.6, 5.5))

ax.plot(K, cybersecurity, marker="P", linestyle="--",
        linewidth=2.2, markersize=7,
        label="Cybersecurity Attacks")

ax.plot(K, self_harm, marker="D", linestyle=":",
        linewidth=2.2, markersize=7,
        label="Self-Harm & Suicide")

ax.plot(K, hate_harass, marker="^", linestyle="-.",
        linewidth=2.2, markersize=7,
        label="Hate, Discrimination & Harassment")

ax.set_xlim(0.6, 10.4)
ax.set_ylim(0, 105)
ax.set_xticks(K)
ax.set_yticks(np.arange(0, 101, 20))

ax.set_xlabel("Attempt Budget K", fontsize=13)
ax.set_ylabel("ASR (%)", fontsize=13)

ax.grid(axis="both", alpha=0.20, linewidth=0.7)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=1,
    frameon=False,
    fontsize=10
)

ax.text(
    0.5, -0.43,
    "(f) Qwen",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=14,
    fontweight="bold"
)

plt.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.43)

plt.savefig("Fig_Qwen_Category_K_EN.png",
            dpi=600, bbox_inches="tight")
plt.savefig("Fig_Qwen_Category_K_EN.svg",
            bbox_inches="tight")

plt.show()