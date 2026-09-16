"""Wave-2 ICML figure refresh: regenerates fig_headline.pdf (7 runs),
fig_ladder.pdf (replication run, both probe families), fig_e3_logprob.pdf
(7 runs). fig_setup/fig_e1/fig_frames are unchanged from make_figs_icml.py.
All stats re-read from the exports; every E3 mean asserted against the
pipeline values from reports/p0_wave2_report.md before plotting."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
# Wave-1 / wave-2 run exports (each: <run>/{calls.parquet,analysis*/...}).
# Override when the exports live elsewhere than the authoring layout.
W1 = Path(os.environ.get("EFP_WAVE1_DATA", HERE.parent / "extracted" / "data"))
W2 = Path(os.environ.get("EFP_WAVE2_DATA", HERE.parent / "wave2" / "data"))
OUT = HERE

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


def best_cells(root, run, adir):
    pr = pd.read_csv(root / run / adir / "probe_results.csv")
    g = pr.loc[pr.groupby("name")["balanced_acc_mean"].idxmax()]
    return {
        r["name"]: (r["balanced_acc_mean"], r["balanced_acc_std"])
        for _, r in g.iterrows()
    }


# (label, root, run, is_visible_outcome_arm)
RUNS = [
    ("goal\ndisc.", W1, "main01", False),
    ("goal\nrepl.", W2, "main02", False),
    ("spatial", W2, "suite_spatial02", False),
    ("object", W2, "suite_object01", False),
    ("long", W2, "suite_long01", False),
    ("mirror", W2, "main02_mirror", True),
    ("freeze", W2, "main02_freeze", True),
]

# pair-level E3 means from the verified pipeline outputs (report §4);
# recomputed here from raw exports and asserted before plotting.
EXPECT = {
    "main01": (0.010005, -1.115168),
    "main02": (0.00125, 0.1448),
    "suite_spatial02": (0.00444, -0.1894),
    "suite_object01": (0.01280, -1.6188),
    "suite_long01": (0.00197, -0.4730),
    "main02_mirror": (0.01402, -1.9390),
    "main02_freeze": (-0.00361, 0.7134),
}


def e3_stats():
    rows = []
    for lab, root, run, _vis in RUNS:
        ps = pd.read_csv(root / run / "analysis" / "probe_samples.csv")
        calls = pd.read_parquet(root / run / "calls.parquet")[
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
        exp = EXPECT[run]
        assert abs(d_ent.mean() - exp[0]) < 2e-4, (run, d_ent.mean())
        assert abs(d_lp.mean() - exp[1]) < 2e-2, (run, d_lp.mean())
        ep = (
            pd.DataFrame({"d_ent": d_ent, "d_lp": d_lp})
            .groupby(level="episode_id")
            .mean()
        )
        rows.append(
            dict(
                lab=lab,
                run=run,
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


e3 = e3_stats()
print(e3[["run", "ent", "p_ent", "lp", "p_lp", "n"]].to_string(index=False))

# ===================================================== Fig 1 (headline, full width)
fig, (a, b) = plt.subplots(
    1, 2, figsize=(6.9, 2.6), gridspec_kw={"wspace": 0.30, "width_ratios": [1.25, 1]}
)

x = np.arange(len(RUNS))
p2r, p2r_e, hid, hid_e, flo, flo_e, cds = [], [], [], [], [], [], []
for lab, root, run, vis in RUNS:
    c = best_cells(root, run, "analysis")
    p2r.append(c["P2r"][0])
    p2r_e.append(c["P2r"][1])
    hb = max(["P1", "P3", "P4"], key=lambda k: c[k][0])
    hid.append(c[hb][0])
    hid_e.append(c[hb][1])
    flo.append(c["P0"][0])
    flo_e.append(c["P0"][1])
    cds.append(c["C_dstates"][0] if vis else np.nan)

a.errorbar(
    x,
    p2r,
    yerr=p2r_e,
    fmt="o-",
    color=ORANGE,
    ms=4.5,
    lw=1.4,
    elinewidth=1,
    capsize=2,
    zorder=4,
)
a.errorbar(
    x + 0.12,
    hid,
    yerr=hid_e,
    fmt="o",
    color=BLUE,
    ms=4,
    elinewidth=1,
    capsize=2,
    zorder=3,
)
a.errorbar(
    x - 0.12,
    flo,
    yerr=flo_e,
    fmt="s",
    color=MUTED,
    ms=3.5,
    elinewidth=0.9,
    capsize=2,
    zorder=2,
)
a.scatter(x, cds, marker="_", s=150, color=AQUA, linewidths=1.8, zorder=5)
for i, v in enumerate(p2r):
    a.annotate(
        f"{v:.2f}",
        (x[i] - 0.05, v + p2r_e[i]),
        xytext=(-2, 3),
        textcoords="offset points",
        ha="center",
        fontsize=6.2,
        color=INK,
    )
a.text(
    6.42,
    p2r[-1] - 0.005,
    "P2r\n(mechanics\noracle)",
    fontsize=6.2,
    color=ORANGE,
    va="center",
    linespacing=1.1,
)
a.text(
    6.42,
    cds[-1] + 0.012,
    "C$_{\\Delta s}$\n(outcome\nonly)",
    fontsize=6.2,
    color=AQUA,
    va="center",
    linespacing=1.1,
)
a.text(
    6.50,
    hid[-1] - 0.030,
    "best hidden\nprobe",
    fontsize=6.2,
    color=BLUE,
    va="center",
    linespacing=1.1,
)
a.text(
    6.50,
    flo[-1] - 0.062,
    "shuffled-\nlabel floor",
    fontsize=6.2,
    color=MUTED,
    va="center",
    linespacing=1.1,
)
a.axhline(0.5, ls=":", lw=0.8, color=MUTED, zorder=1)
a.axvline(4.55, ls=":", lw=0.7, color=BASE, zorder=1)
a.text(1.7, 0.945, "swap: outcome plausible", fontsize=6.2, color=INK2, ha="center")
a.text(6.05, 0.955, "outcome visible", fontsize=6.2, color=INK2, ha="center")
a.set_xticks(x)
a.set_xticklabels([r[0] for r in RUNS], fontsize=6.2)
a.set_xlim(-0.5, 7.7)
a.set_ylim(0.42, 0.97)
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
    np.arange(len(e3)),
    e3["ent"],
    yerr=e3["ent_se"],
    fmt="o",
    color=BLUE,
    ms=4.5,
    elinewidth=1.0,
    capsize=2,
    zorder=3,
)
for i, r in e3.iterrows():
    sig = r["p_ent"] < 0.05
    y = r["ent"] + (r["ent_se"] if r["ent"] >= 0 else -r["ent_se"])
    va, dy = ("bottom", 3.0) if r["ent"] >= 0 else ("top", -3.0)
    b.annotate(
        fmt_p(r["p_ent"]),
        (i, y),
        xytext=(0, dy),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=5.8,
        color=INK if sig else MUTED,
        fontweight="bold" if sig else "normal",
        rotation=0,
    )
b.set_xticks(np.arange(len(e3)))
b.set_xticklabels([r[0] for r in RUNS], fontsize=5.9)
b.set_title("(b) Reaction gated by visual novelty", loc="left", fontsize=7.4, color=INK)
b.set_ylabel(r"$\Delta$ entropy after hijack (nats)", fontsize=7)
style(b)
b.set_xlim(-0.6, len(e3) - 0.4)
lo = float(min(e3["ent"] - e3["ent_se"]))
hi = float(max(e3["ent"] + e3["ent_se"]))
pad = 0.55 * (hi - lo)
b.set_ylim(lo - pad, hi + pad)
fig.savefig(OUT / "fig_headline.pdf", bbox_inches="tight")
plt.close(fig)

# ===================================================== Fig ladder (replication run, both families)
lin = best_cells(W2, "main02", "analysis")
mlp = best_cells(W2, "main02", "analysis_mlp")
probes = ["P0", "P1", "P3", "P4", "P2", "P2r", "C_cmd", "C_dstates", "C_phase"]
labels = [
    "P0\n(floor)",
    "P1",
    "P3",
    "P4",
    "P2\n$a{\\oplus}\\Delta s$",
    "P2r",
    "C$_{cmd}$",
    "C$_{\\Delta s}$",
    "C$_{ph}$",
]
xx = np.arange(len(probes))

fig, ax = plt.subplots(figsize=(3.35, 2.65))
ax.axhline(lin["P0"][0], ls=(0, (4, 3)), lw=0.8, color=BLUE, alpha=0.45, zorder=1)
ax.axhline(mlp["P0"][0], ls=(0, (1, 2)), lw=0.9, color=ORANGE, alpha=0.5, zorder=1)
ax.errorbar(
    xx - 0.10,
    [lin[p][0] for p in probes],
    yerr=[lin[p][1] for p in probes],
    fmt="o",
    color=BLUE,
    ms=3.8,
    elinewidth=0.9,
    capsize=1.8,
    zorder=3,
    label="linear",
)
ax.errorbar(
    xx + 0.10,
    [mlp[p][0] for p in probes],
    yerr=[mlp[p][1] for p in probes],
    fmt="^",
    color=ORANGE,
    ms=4.0,
    elinewidth=0.9,
    capsize=1.8,
    zorder=4,
    label="MLP (256, 1 layer)",
)
for p, dx in (("P2", 0.10), ("P2r", 0.10)):
    i = probes.index(p)
    ax.annotate(
        f"{mlp[p][0]:.2f}",
        (i + dx, mlp[p][0] + mlp[p][1]),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        fontsize=6.2,
        color=ORANGE,
    )
ax.annotate(
    "same family,\nsame $a$ block:\nswap $\\Delta s$ for $h(t{+}1)$\nand it collapses",
    xy=(3 + 0.10, mlp["P3"][0] + 0.012),
    xytext=(1.15, 0.655),
    fontsize=6.0,
    color=INK2,
    linespacing=1.15,
    arrowprops=dict(arrowstyle="-", lw=0.7, color=INK2, shrinkB=2),
)
ax.axhline(0.5, ls=":", lw=0.8, color=MUTED, zorder=1)
ax.set_xticks(xx)
ax.set_xticklabels(labels, fontsize=6.1)
ax.set_ylim(0.40, 0.85)
ax.set_ylabel("balanced accuracy", fontsize=7)
ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
style(ax)
fig.savefig(OUT / "fig_ladder.pdf", bbox_inches="tight")
plt.close(fig)

# ===================================================== Fig E3 log-prob (7 runs)
fig, ax = plt.subplots(figsize=(3.35, 2.35))
ax.axhline(0, lw=0.8, color=BASE, zorder=1)
ax.errorbar(
    np.arange(len(e3)),
    e3["lp"],
    yerr=e3["lp_se"],
    fmt="o",
    color=BLUE,
    ms=4.2,
    elinewidth=1.0,
    capsize=2,
    zorder=3,
)
for i, r in e3.iterrows():
    sig = r["p_lp"] < 0.05
    y = r["lp"] + (r["lp_se"] if r["lp"] >= 0 else -r["lp_se"])
    va, dy = ("bottom", 3.0) if r["lp"] >= 0 else ("top", -3.0)
    ax.annotate(
        fmt_p(r["p_lp"]),
        (i, y),
        xytext=(0, dy),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=5.8,
        color=INK if sig else MUTED,
        fontweight="bold" if sig else "normal",
    )
ax.set_xticks(np.arange(len(e3)))
ax.set_xticklabels([r[0] for r in RUNS], fontsize=5.9)
ax.set_ylabel(r"$\Delta$ log-prob of next chunk", fontsize=7)
style(ax)
lo = float(min(e3["lp"] - e3["lp_se"]))
hi = float(max(e3["lp"] + e3["lp_se"]))
pad = 0.5 * (hi - lo)
ax.set_ylim(lo - pad, hi + pad)
fig.savefig(OUT / "fig_e3_logprob.pdf", bbox_inches="tight")
plt.close(fig)

print("wrote fig_headline.pdf fig_ladder.pdf fig_e3_logprob.pdf")
