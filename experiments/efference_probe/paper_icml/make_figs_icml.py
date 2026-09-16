"""ICML two-column figure set (vector PDFs). Same verified data pipeline as
make_figs*.py: stats re-read from the export, E3 recomputed and asserted
against pipeline values before plotting. Sizes: full-width ~6.8in, column
~3.3in, fonts 7-8.5pt at final size."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent / "extracted" / "data"
OUT = Path(__file__).resolve().parent

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": BASE,
        "axes.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
    }
)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def best_cells(run, adir):
    pr = pd.read_csv(ROOT / run / adir / "probe_results.csv")
    g = pr.loc[pr.groupby("name")["balanced_acc_mean"].idxmax()]
    return {
        r["name"]: (r["balanced_acc_mean"], r["balanced_acc_std"])
        for _, r in g.iterrows()
    }


ARMS = [
    ("main01", "analysis", "swap\n10 tasks"),
    ("main01", "analysis_E_matched", "swap\n5 tasks"),
    ("main01_mirror", "analysis", "mirror\n5 tasks"),
    ("main01_freeze", "analysis", "freeze\n5 tasks"),
]
EXPECT = {
    ("main01", "analysis"): (0.010005, -1.115168),
    ("main01", "analysis_E_matched"): (0.011924, -1.384574),
    ("main01_mirror", "analysis"): (0.003811, -1.656544),
    ("main01_freeze", "analysis"): (-0.003228, 0.267024),
}


def e3_stats():
    rows = []
    for run, adir, lab in ARMS:
        ps = pd.read_csv(ROOT / run / adir / "probe_samples.csv")
        calls = pd.read_parquet(ROOT / run / "calls.parquet")[
            ["episode_id", "call_idx", "entropy_mean", "logprob_sum"]
        ]
        m = ps.merge(
            calls,
            left_on=["episode_id", "cur_call"],
            right_on=["episode_id", "call_idx"],
            how="left",
        )
        piv = m.pivot_table(
            index=["pair_id", "episode_id"],
            columns="y",
            values=["entropy_mean", "logprob_sum"],
        )
        d_ent = piv[("entropy_mean", 1)] - piv[("entropy_mean", 0)]
        d_lp = piv[("logprob_sum", 1)] - piv[("logprob_sum", 0)]
        exp = EXPECT[(run, adir)]
        assert abs(d_ent.mean() - exp[0]) < 2e-4 and abs(d_lp.mean() - exp[1]) < 2e-2
        ep = (
            pd.DataFrame({"d_ent": d_ent, "d_lp": d_lp})
            .groupby(level="episode_id")
            .mean()
        )
        rows.append(
            dict(
                lab=lab.split("\n")[0],
                n=len(ep),
                ent=d_ent.mean(),
                ent_se=ep["d_ent"].std() / np.sqrt(len(ep)),
                lp=d_lp.mean(),
                lp_se=ep["d_lp"].std() / np.sqrt(len(ep)),
                p_ent=wilcoxon(ep["d_ent"]).pvalue,
                p_lp=wilcoxon(ep["d_lp"]).pvalue,
            )
        )
    return pd.DataFrame(rows)


def fmt_p(p):
    return f"p={p:.1e}" if p < 0.01 else f"p={p:.2f}"


# ===================================================== Fig 1 (headline, full width)
e3 = e3_stats()
fig, (a, b) = plt.subplots(
    1, 2, figsize=(6.8, 2.55), gridspec_kw={"wspace": 0.34, "width_ratios": [1.15, 1]}
)

arm_x = np.arange(4)
p2r, p2r_e, hid, hid_e, flo, flo_e, ns = [], [], [], [], [], [], []
for run, adir, _ in ARMS:
    c = best_cells(run, adir)
    p2r.append(c["P2r"][0])
    p2r_e.append(c["P2r"][1])
    hb = max(["P1", "P3", "P4"], key=lambda k: c[k][0])
    hid.append(c[hb][0])
    hid_e.append(c[hb][1])
    flo.append(c["P0"][0])
    flo_e.append(c["P0"][1])
a.errorbar(
    arm_x,
    p2r,
    yerr=p2r_e,
    fmt="o-",
    color=ORANGE,
    ms=5,
    lw=1.5,
    elinewidth=1,
    capsize=2,
    zorder=4,
)
a.errorbar(
    arm_x + 0.1,
    hid,
    yerr=hid_e,
    fmt="o",
    color=BLUE,
    ms=4.5,
    elinewidth=1,
    capsize=2,
    zorder=3,
)
a.errorbar(
    arm_x - 0.1,
    flo,
    yerr=flo_e,
    fmt="s",
    color=MUTED,
    ms=4,
    elinewidth=0.9,
    capsize=2,
    zorder=2,
)
for i, v in enumerate(p2r):
    a.annotate(
        f"{v:.2f}" if i < 3 else f"{v:.3f}",
        (arm_x[i] - 0.06, v + p2r_e[i]),
        xytext=(-2, 3),
        textcoords="offset points",
        ha="center",
        fontsize=6.5,
        color=INK,
    )
a.text(
    3.30,
    p2r[-1] - 0.012,
    "P2r\n(mechanics\noracle)",
    fontsize=6.3,
    color=INK2,
    va="center",
    linespacing=1.1,
)
a.text(
    3.38,
    hid[-1] + 0.004,
    "best hidden\nprobe",
    fontsize=6.3,
    color=INK2,
    va="center",
    linespacing=1.1,
)
a.text(
    3.38,
    flo[-1] - 0.062,
    "shuffled-\nlabel floor",
    fontsize=6.3,
    color=INK2,
    va="center",
    linespacing=1.1,
)
a.axhline(0.5, ls=":", lw=0.8, color=MUTED, zorder=1)
a.set_xticks(arm_x)
a.set_xticklabels([t for _, _, t in ARMS], fontsize=6.5)
a.set_xlim(-0.45, 4.45)
a.set_ylim(0.42, 0.96)
a.set_ylabel("balanced accuracy (best of 27 cells)", fontsize=7)
a.set_title(
    "(a) Decodable from mechanics, never from the policy",
    loc="left",
    fontsize=7.4,
    color=INK,
)
style(a)

b.axhline(0, lw=0.8, color=BASE, zorder=1)
b.errorbar(
    np.arange(4),
    e3["ent"],
    yerr=e3["ent_se"],
    fmt="o",
    color=BLUE,
    ms=5,
    elinewidth=1.1,
    capsize=2,
    zorder=3,
)
for i, r in e3.iterrows():
    sig = r["p_ent"] < 0.05
    y = r["ent"] + (r["ent_se"] if r["ent"] >= 0 else -r["ent_se"])
    va, dy = ("bottom", 3.5) if r["ent"] >= 0 else ("top", -3.5)
    b.annotate(
        fmt_p(r["p_ent"]),
        (i, y),
        xytext=(0, dy),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=6.3,
        color=INK if sig else MUTED,
        fontweight="bold" if sig else "normal",
    )
b.set_xticks(np.arange(4))
b.set_xticklabels([f"{r.lab}\nn={r.n} ep" for r in e3.itertuples()], fontsize=6.5)
b.set_title("(b) Reaction gated by visual novelty", loc="left", fontsize=7.4, color=INK)
b.set_ylabel(r"$\Delta$ entropy after hijack (nats)", fontsize=7)
style(b)
b.set_xlim(-0.6, 3.6)
lo = min(e3["ent"] - e3["ent_se"])
hi = max(e3["ent"] + e3["ent_se"])
pad = 0.55 * (hi - lo)
b.set_ylim(lo - pad, hi + pad)
fig.savefig(OUT / "fig_headline.pdf", bbox_inches="tight")
plt.close(fig)

# ===================================================== Fig 2 (setup, full width)
fig, ax = plt.subplots(figsize=(6.8, 2.15))
ax.set_xlim(0, 104)
ax.set_ylim(0, 100)
ax.axis("off")
ax.grid(False)


def box(x, y, w, h, text, fc="#f4f4f2", ec=BASE, fs=7.2):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=1.1", fc=fc, ec=ec, lw=0.9
        )
    )
    ax.text(
        x, y, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25
    )


def arrow(x0, x1, y=64):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=9,
            color=INK2,
            lw=1.1,
            shrinkA=2,
            shrinkB=2,
        )
    )


Y = 64
box(7, Y, 10, 24, "image $t$\n+ instruction", fs=6.8)
arrow(13, 19)
box(26, Y, 12, 24, "VLA policy\n(frozen, 7B)", fc="#e8f0fb", ec=BLUE, fs=6.8)
arrow(33, 39)
box(45, Y, 10, 24, "chunk $a_t$\n(8×7)", fs=6.8)
arrow(51, 57)
box(
    64,
    Y,
    15,
    32,
    "executed chunk\n75%: $a_t$ itself\n25%: transform",
    fc="#fdeee7",
    ec=ORANGE,
    fs=6.5,
)
ax.text(64, 37, "swap · mirror · freeze", ha="center", fontsize=6.2, color=ORANGE)
arrow(72, 77)
box(83, Y, 9, 24, "env\n8 steps", fs=6.8)
arrow(88, 94)
box(96.5, Y, 6, 24, "image\n$t\\!+\\!1$", fs=6.4)
ax.plot([26, 26], [48, 30], color=BLUE, lw=0.9, ls=":")
ax.text(
    26,
    23,
    "$h(t)$ captured\n9 layers × 3 pools",
    ha="center",
    fontsize=6.4,
    color=BLUE,
    linespacing=1.2,
)
ax.plot([96.5, 96.5], [48, 32], color=BLUE, lw=0.9, ls=":")
ax.text(96.5, 26, "→ next call", ha="center", fontsize=6.2, color=BLUE)
ax.text(
    37,
    8,
    "next call: $h(t\\!+\\!1)$ captured → probes: was that transition self-caused?\n"
    "next chunk $a_{t+1}$ → behavior: $\\Delta$entropy, $\\Delta$log-prob (hijack vs. phase-matched self)",
    ha="left",
    fontsize=6.6,
    color=INK,
    linespacing=1.55,
)
fig.savefig(OUT / "fig_setup.pdf", bbox_inches="tight")
plt.close(fig)

# ===================================================== Fig 3 (ladder, column)
cells = best_cells("main01", "analysis")
floor_mu = cells["P0"][0]
probes = ["P1", "P3", "P4", "C_cmd", "C_dstates", "C_phase", "P2", "P2r"]
labels = ["P1", "P3", "P4", "cmd", "Δs", "phase", "P2", "P2r"]
colors = [BLUE] * 3 + [MUTED] * 3 + [ORANGE] * 2
fig, a = plt.subplots(figsize=(3.3, 2.5))
xs = np.array([0, 1, 2, 3.6, 4.7, 5.8, 7.2, 8.2])
for i, p in enumerate(probes):
    mu, sd = cells[p]
    filled = p != "P2"
    a.errorbar(
        xs[i],
        mu,
        yerr=sd,
        fmt="o",
        ms=4.5,
        color=colors[i],
        mfc=colors[i] if filled else SURFACE,
        mec=colors[i],
        mew=1.1,
        ecolor=colors[i],
        elinewidth=0.9,
        capsize=1.8,
        zorder=3,
    )
a.axhline(0.5, ls=":", lw=0.8, color=MUTED, zorder=1)
a.axhline(floor_mu, ls="--", lw=1, color=INK2, zorder=1)
a.text(
    9.7,
    floor_mu + 0.006,
    "selection-aware\nfloor 0.536",
    ha="right",
    va="bottom",
    fontsize=6,
    color=INK2,
    linespacing=1.1,
)
a.text(9.7, 0.497, "chance", fontsize=6, color=MUTED, va="top", ha="right")
mu, sd = cells["P2r"]
a.annotate(
    f"{mu:.2f}",
    (xs[-1], mu + sd),
    xytext=(0, 3),
    textcoords="offset points",
    ha="center",
    fontsize=7,
    color=INK,
    fontweight="bold",
)
a.set_xticks(xs)
a.set_xticklabels(labels, fontsize=6.5)
a.set_xlim(-0.7, 9.9)
a.set_ylim(0.42, 0.78)
a.set_ylabel("balanced accuracy", fontsize=7)
style(a)
for x0, x1, name in [
    (-0.4, 2.4, "hidden state"),
    (3.2, 6.2, "controls"),
    (6.8, 8.6, "mechanics"),
]:
    a.text(
        (x0 + x1) / 2, 0.43, name, ha="center", fontsize=6, color=MUTED, style="italic"
    )
fig.savefig(OUT / "fig_ladder.pdf", bbox_inches="tight")
plt.close(fig)

# ===================================================== Fig 4 (E1, column)
e1 = pd.read_csv(ROOT / "main01" / "analysis" / "e1_readout.csv")
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for pool, color, lab in [
    ("ctx_mean", BLUE, "context mean"),
    ("ctx_last", ORANGE, "last pre-action token"),
    ("act_mean", AQUA, "action-window mean"),
]:
    d = e1[e1["pool"] == pool].sort_values("layer")
    ax.plot(
        d["layer"], d["r2_mean"], "-o", color=color, lw=1.4, ms=3.5, label=lab, zorder=3
    )
best = e1.loc[e1["r2_mean"].idxmax()]
ax.annotate(
    f"$R^2$ = {best['r2_mean']:.2f}",
    (best["layer"], best["r2_mean"]),
    xytext=(0, 5),
    textcoords="offset points",
    ha="center",
    fontsize=7,
    color=INK,
    fontweight="bold",
)
ax.set_xticks(sorted(e1["layer"].unique()))
ax.set_xlabel("layer", fontsize=7)
ax.set_ylabel("$R^2$ (readout of commanded chunk)", fontsize=7)
ax.set_ylim(-0.05, 0.78)
ax.legend(loc="lower right", fontsize=6.2, handlelength=1.6)
style(ax)
fig.savefig(OUT / "fig_e1.pdf", bbox_inches="tight")
plt.close(fig)

# ===================================================== Fig 5 (E3 log-prob, column)
fig, b = plt.subplots(figsize=(3.3, 2.3))
b.axhline(0, lw=0.8, color=BASE, zorder=1)
b.errorbar(
    np.arange(4),
    e3["lp"],
    yerr=e3["lp_se"],
    fmt="o",
    color=BLUE,
    ms=5,
    elinewidth=1.1,
    capsize=2,
    zorder=3,
)
for i, r in e3.iterrows():
    sig = r["p_lp"] < 0.05
    y = r["lp"] + (r["lp_se"] if r["lp"] >= 0 else -r["lp_se"])
    va, dy = ("bottom", 3.5) if r["lp"] >= 0 else ("top", -3.5)
    b.annotate(
        fmt_p(r["p_lp"]),
        (i, y),
        xytext=(0, dy),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=6.3,
        color=INK if sig else MUTED,
        fontweight="bold" if sig else "normal",
    )
b.set_xticks(np.arange(4))
b.set_xticklabels([f"{r.lab}\nn={r.n} ep" for r in e3.itertuples()], fontsize=6.5)
b.set_ylabel(r"$\Delta$ log-prob of chosen chunk", fontsize=7)
style(b)
b.set_xlim(-0.6, 3.6)
lo = min(e3["lp"] - e3["lp_se"])
hi = max(e3["lp"] + e3["lp_se"])
pad = 0.55 * (hi - lo)
b.set_ylim(lo - pad, hi + pad)
fig.savefig(OUT / "fig_e3_logprob.pdf", bbox_inches="tight")
plt.close(fig)

# ===================================================== Fig 6 (frames, appendix column)
F = ROOT / "pilotC_task3" / "frames"
panels = [
    ("ep00002/call0005_main.jpg", "Ep. 2, call 5 (hijacked)"),
    ("ep00002/call0006_main.jpg", "call 6 (next view)"),
    ("ep00001/call0037_main.jpg", "Ep. 1, call 37 (hijacked)"),
    ("ep00001/call0038_main.jpg", "call 38 (next view)"),
]
fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.55))
for ax, (rel, title) in zip(axes.flat, panels):
    ax.imshow(mpimg.imread(F / rel))
    ax.set_title(title, fontsize=6.5, color=INK, loc="left", pad=2)
    ax.axis("off")
fig.subplots_adjust(hspace=0.14, wspace=0.04)
fig.savefig(OUT / "fig_frames.pdf", bbox_inches="tight", dpi=200)
plt.close(fig)

print("ICML figures done:", sorted(p.name for p in OUT.glob("fig_*.pdf")))
