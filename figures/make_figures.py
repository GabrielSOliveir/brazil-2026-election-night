"""Regenerates every figure in figures/ from data/derived (and data/work for the section-level panels).
Run from the repo root:  python figures/make_figures.py"""
import json, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "harness")
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
FINAL = 100 * 64256232 / (64256232 + 59826041)  # official 2nd-round result, Lula share of valid votes

# ---------------------------------------------------------------- 1. election night
d = pd.read_csv("data/derived/live_projection_log.csv")
d["lo"] = d.ic90.apply(lambda s: json.loads(s)[0]); d["hi"] = d.ic90.apply(lambda s: json.loads(s)[1])
raw, tse_pct = d.bruto_lula, d.pct_secoes   # raw Lula share and % of sections in the TSE feed at each reading
swing = raw.max() - raw.min()
fig, ax = plt.subplots(figsize=(11, 5.4)); x = np.arange(len(d))
ax.fill_between(x, d.lo, d.hi, color=BLUE, alpha=.14, lw=0, label="Model 90% interval")
ax.plot(x, d.lula, color=BLUE, lw=2.4, marker="o", ms=6, label="Model projection")
ax.plot(x, d.checagem_so_completas, color=AQUA, lw=1.8, marker="o", ms=4.5, label="Cross-check (completed zones only)")
ax.plot(x, raw, color=GRAY, lw=2, ls="--", marker="o", ms=4.5, label="Raw count in the TSE feed")
ax.axhline(FINAL, color=INK, lw=1.1); ax.text(len(d) - .55, FINAL + .08, f"Final: Lula {FINAL:.2f}%", ha="right", color=INK)
ax.axhline(50, color="#e34948", lw=1, ls=":"); ax.text(4.0, 49.15, "raw count shows Flávio ahead", color="#c23b3a", fontsize=9, ha="center")
ax.set_xticks(x); ax.set_xticklabels([f"{h}\n{p:.0f}%" for h, p in zip(d.hora, tse_pct)], fontsize=9)
ax.set_xlabel("Time of reading (BRT) and share of polling sections counted by TSE"); ax.set_ylabel("Lula, % of valid votes")
ax.set_ylim(49.0, 54.2); ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.set_title(f"Election night, Oct 25 2026: the model stayed within 51.2–52.3% while the raw count swung {swing:.1f} points", loc="left", fontsize=12, color=INK)
fig.tight_layout(); fig.savefig("figures/01_election_night.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- 2. research harness: backtests
M = [("Uniform swing\n(baseline)", .77, .211), ("Frequentist\nrandom effects", 1.639, .141), ("Hierarchical\nBayes (r1)", .46, .135),
     ("ML stack\n(XGB+MLP)", .44, .135), ("Hierarchical\nBayes (final)", .406, .120)]
P = [("Treat partial zones\nas complete", 4.32), ("Completed zones\nonly", .313), ("Partial-aware\nBayes (deployed)", .209)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
xx = np.arange(len(M)); w = .38
b1 = a1.bar(xx - w / 2, [m[1] for m in M], w, color=ORANGE, label="2018 (development set)")
b2 = a1.bar(xx + w / 2, [m[2] for m in M], w, color=BLUE, label="2022 (test set)")
for b in list(b1) + list(b2): a1.text(b.get_x() + b.get_width() / 2, b.get_height() + .03, f"{b.get_height():.2f}", ha="center", fontsize=8.5, color=INK2)
a1.set_xticks(xx); a1.set_xticklabels([m[0] for m in M], fontsize=9); a1.set_ylabel("National error, RMSE (pp)")
a1.set_title("Replaying past election nights, zone by zone", loc="left", color=INK); a1.legend(frameon=False)
b = a2.bar(range(3), [p[1] for p in P], color=[GRAY, AQUA, BLUE]); 
for r in b: a2.text(r.get_x() + r.get_width() / 2, r.get_height() + .08, f"{r.get_height():.2f}", ha="center", fontsize=9, color=INK2)
a2.set_xticks(range(3)); a2.set_xticklabels([p[0] for p in P], fontsize=9); a2.set_ylabel("National error, RMSE (pp)")
a2.set_title("The live feed also ships half-counted zones\n(2022 replay with realistic partial counts)", loc="left", color=INK)
fig.tight_layout(); fig.savefig("figures/02_backtests.png", dpi=160); plt.close(fig)
print("figures written")
