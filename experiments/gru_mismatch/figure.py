#!/usr/bin/env python
"""E1 figure: attribution in the GRU, per efference interface and transform.

One panel per hijack transform; per interface, the best hidden-state probe
(dots = seeds, bar = mean ± sd) against its selection-aware floor, the
mechanical ceiling P2r and the outcome-only control C_out.  Palette and
styling follow ``efference_probe/paper_icml/make_figs_icml_w2.py`` so the
figure drops into the same paper.

Example::

    python experiments/gru_mismatch/figure.py experiments/gru_mismatch/results/e1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gru_mismatch.aggregate import MIN_EFFECT, _load  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
INTERFACES = ("none", "trace_abs", "trace_signed", "command")
LABELS = {
    "none": "none\n(exp3)",
    "trace_abs": "|a| trace\n(exp4)",
    "trace_signed": "signed\ntrace",
    "command": "command\na_t",
}
TRANSFORMS = ("swap", "mirror", "freeze")

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="run_gru.py --out directory")
    parser.add_argument("--out", default=None, help="figure path stem")
    args = parser.parse_args(argv)
    root = Path(args.results)
    arms, probes = _load(root)
    arms = arms[arms["transform"] != ""]
    if arms.empty:
        raise SystemExit(f"no phase-3 arms under {root}")
    interfaces = [i for i in INTERFACES if i in set(arms["interface"])]

    fig, axes = plt.subplots(1, len(TRANSFORMS), figsize=(7.0, 2.5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, transform in zip(axes, TRANSFORMS):
        for x, interface in enumerate(interfaces):
            group = arms[
                (arms["interface"] == interface) & (arms["transform"] == transform)
            ]
            if group.empty:
                continue
            subset = probes[
                (probes["interface"] == interface) & (probes["transform"] == transform)
            ]
            floor = group["selection_floor"].mean()
            ax.hlines(
                floor, x - 0.3, x + 0.3, colors=MUTED, linestyles="--", lw=1.0, zorder=2
            )
            hidden = group["best_hidden"].to_numpy()
            jitter = rng.uniform(-0.07, 0.07, size=len(hidden))
            ax.scatter(
                x - 0.12 + jitter, hidden, s=9, color=BLUE, alpha=0.45, lw=0, zorder=3
            )
            ax.errorbar(
                x - 0.12,
                hidden.mean(),
                yerr=hidden.std(ddof=1) if len(hidden) > 1 else 0,
                fmt="o",
                ms=4.5,
                color=BLUE,
                elinewidth=1.2,
                capsize=2,
                zorder=4,
            )
            for name, color, marker, dx in (
                ("P2r", ORANGE, "D", 0.12),
                ("C_out", MUTED, "s", 0.24),
            ):
                values = subset[subset["name"] == name]["balanced_acc_mean"]
                if not values.empty:
                    ax.scatter(
                        x + dx,
                        values.mean(),
                        s=18,
                        marker=marker,
                        facecolor=SURFACE if name == "C_out" else color,
                        edgecolor=color,
                        lw=1.1,
                        zorder=4,
                    )
        ax.set_title(transform.upper(), color=INK)
        ax.set_xticks(range(len(interfaces)))
        ax.set_xticklabels([LABELS[i] for i in interfaces])
        ax.axhline(0.5, ls=":", lw=0.8, color=MUTED, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("balanced accuracy")

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color=BLUE,
            lw=0,
            ms=4.5,
            label="best hidden probe (P1/P1g; dots = seeds)",
        ),
        plt.Line2D([], [], color=MUTED, ls="--", lw=1.0, label="selection-aware floor"),
        plt.Line2D(
            [], [], marker="D", color=ORANGE, lw=0, ms=4, label="P2r mechanics oracle"
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            markerfacecolor=SURFACE,
            markeredgecolor=MUTED,
            color=MUTED,
            lw=0,
            ms=4,
            label="C_out outcome only",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.text(
        0.99,
        0.01,
        f"readable = ≥ floor + {MIN_EFFECT} in ≥ 4/5 seeds",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    stem = Path(args.out) if args.out else root / "fig_e1_gru"
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=200, bbox_inches="tight")
    print(f"wrote {stem.with_suffix('.png')} and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
