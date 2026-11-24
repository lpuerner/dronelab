"""Tables and figures from the rollout evaluation -> results/report.md + results/figures/.

    python projects/surrogate/scripts/analyze.py

Unit of every statistic is the *flight*: the errors of all rollouts of a flight are averaged
first, then medians / IQRs / Wilcoxon tests are taken over flights. Networks: the per-flight
errors are averaged over the training seeds; the seed spread is reported separately.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from dronelab.surrogate.datasets import T2_MAX_SPEED  # noqa: E402
from dronelab.surrogate.evaluate import HORIZONS  # noqa: E402

KINDS = ("phys", "phys_mlp", "mlp")
LABEL = {"phys": "phys", "phys_mlp": "phys+mlp", "mlp": "mlp"}
COLOR = {"phys": "#7f7f7f", "phys_mlp": "#1f77b4", "mlp": "#d62728"}
SET_DESC = {"T1": "in distribution: new flights like training",
            "T2": "speed extrapolation: T1 flights > 1.5 m/s",
            "T3": "benchmark courses, incl. unseen DSL-PID",
            "T4": "disturbances: wind, gusts, mass, motor lag"}
H_MAIN = HORIZONS.index(20)  # 0.42 s, the MPC horizon
H_TXT = {h: f"{h / 48:.2f} s" if h < 48 else f"{h / 48:.0f} s" for h in HORIZONS}
H_TXT[1] = "1 step"
TRUE_PARAMS = {"kf": 3.16e-10, "km": 7.94e-12, "kd_xy": 9.1785e-7, "kd_z": 1.0311e-6,
               "tau": 0.02}
SPEED_BINS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5]
Z_BINS = [0, 0.3, 0.5, 1.0, 1.5, 3.0]


def _parse(stem: str):
    """'phys_mlp_full_s1__T3' -> (kind, variant, seed, set)."""
    model, s = stem.split("__")
    m = re.match(r"(phys_mlp|phys|mlp)_(full|slow)(?:_s(\d+))?$", model)
    return m.group(1), m.group(2), int(m.group(3) or 0), s


def load(eval_dir: Path):
    """res[(kind, variant, set)] = list over seeds of rollout dicts."""
    res = defaultdict(list)
    for f in sorted(eval_dir.glob("*__*.npz")):
        kind, var, seed, s = _parse(f.stem)
        with np.load(f) as d:
            res[(kind, var, s)].append({k: d[k] for k in d.files})
    flights = {s: json.loads((eval_dir / f"flights_{s}.json").read_text())
               for s in ("T1", "T3", "T4") if (eval_dir / f"flights_{s}.json").exists()}
    return res, flights


def per_flight(r: dict, key: str, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """(flight ids, mean error per flight x horizon), optionally on a rollout mask."""
    fl, e = r["flight"], r[key]
    if mask is not None:
        fl, e = fl[mask], e[mask]
    ids = np.unique(fl)
    return ids, np.stack([e[fl == i].mean(0) for i in ids]) if len(ids) else np.zeros((0, 5))


def flight_errors(runs: list[dict], key: str, keep=None):
    """Per-flight errors averaged over seeds, and the per-seed medians (seed spread)."""
    per_seed, ids = [], None
    for r in runs:
        i, e = per_flight(r, key)
        if keep is not None:
            sel = np.isin(i, keep)
            i, e = i[sel], e[sel]
        ids = i
        per_seed.append(e)
    stack = np.stack(per_seed)
    return ids, stack.mean(0), np.median(stack, 1)  # (flights,), (flights, H), (seeds, H)


def subsets(flights: dict) -> dict[str, tuple[str, np.ndarray | None]]:
    """Evaluation sets -> (source set, flight indices or None for all)."""
    t1 = flights.get("T1", [])
    fast = np.array([i for i, f in enumerate(t1) if f["max_speed"] > T2_MAX_SPEED])
    return {"T1": ("T1", None), "T2": ("T1", fast), "T3": ("T3", None), "T4": ("T4", None)}


def fmt(med, q1, q3, scale=1.0, d=1):
    return f"{med * scale:.{d}f} [{q1 * scale:.{d}f}–{q3 * scale:.{d}f}]"


def training_curves(models_dir: Path, figs: Path, kind: str) -> None:
    """Train / validation loss per epoch of the final models of one kind (seeds, variants)."""
    runs = sorted(models_dir.glob(f"{kind}_*_s*.json"))
    if not runs:
        return
    variants = sorted({json.loads(p.read_text())["variant"] for p in runs})
    fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 3.8),
                             sharey=True, squeeze=False)
    for ax, var in zip(axes[0], variants, strict=True):
        ramp = None
        for p in runs:
            d = json.loads(p.read_text())
            if d["variant"] != var:
                continue
            h, s = d["history"], d["seed"]
            ramp = d["config"]["ramp_epochs"]
            c = plt.cm.Reds(0.45 + 0.25 * s)
            ax.plot(h["epoch"], h["train"], "--", color=c, lw=1.2,
                    label="train" if s == 0 else None)
            ax.plot(h["epoch"], h["val"], "-o", color=c, lw=1.5, ms=3, label=f"val, seed {s}")
            best = int(np.argmin(h["val"][ramp:])) + ramp
            ax.plot(h["epoch"][best], h["val"][best], "*", color="k", ms=9,
                    label="restored (best val)" if s == 0 else None)
        if ramp:
            ax.axvspan(-0.5, ramp - 0.5, color="0.9", zorder=0)
            ax.text(ramp / 2 - 0.5, 0.97, "horizon\nramp 5→20", ha="center", va="top",
                    fontsize=7, transform=ax.get_xaxis_transform())
        ax.set_yscale("log")
        ax.set_xlabel("epoch (150 batches of 512)")
        ax.set_title(f"{LABEL[kind]} – {var} training data")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("scaled rollout loss (20 steps)")
    fig.tight_layout()
    fig.savefig(figs / f"training_{kind}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="projects/surrogate/results")
    args = ap.parse_args()
    root = Path(args.results)
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    res, flights = load(root / "eval")
    sets = subsets(flights)
    md: list[str] = ["# Surrogate – results\n",
                     "Generated by `projects/surrogate/scripts/analyze.py`. Errors are open-loop "
                     "rollouts with the recorded commands; per flight averaged over all start "
                     "points (every 0.25 s), then median [IQR] over flights. Networks: mean of "
                     "3 training seeds per flight; ± = spread (SD) of the median over seeds.\n"]

    def get(kind, var, s, key="e_pos"):
        src, keep = sets[s]
        runs = res.get((kind, var, src))
        if not runs:
            return None
        return flight_errors(runs, key, keep)

    # ---------------------------------------------------------------- main table
    md.append("## Main result (full training data)\n")
    for key, unit, sc, d in (("e_pos", "position error [cm]", 100, 1),
                             ("e_att", "attitude error [°]", 1, 1),
                             ("e_vel", "velocity error [m/s]", 1, 3)):
        md.append(f"\n**{unit}**\n")
        md.append("| set | model | n | " + " | ".join(H_TXT[h] for h in HORIZONS) + " |")
        md.append("|---|---|---|" + "---|" * len(HORIZONS))
        for s in sets:
            for k in KINDS:
                g = get(k, "full", s, key)
                if g is None:
                    continue
                ids, e, seed_med = g
                cells = []
                for j in range(len(HORIZONS)):
                    q1, med, q3 = np.percentile(e[:, j], [25, 50, 75])
                    c = fmt(med, q1, q3, sc, d)
                    if len(seed_med) > 1:
                        c += f" ±{np.std(seed_med[:, j]) * sc:.{d}f}"
                    cells.append(c)
                md.append(f"| {s} | {LABEL[k]} | {len(ids)} | " + " | ".join(cells) + " |")

    # ---------------------------------------------------------------- ablation
    md.append(f"\n## Ablation: full data vs. slow flights only (≤ {T2_MAX_SPEED} m/s)\n")
    md.append(f"Median position error [cm] at {H_TXT[20]} (MPC horizon). T2 = T1 flights "
              f"faster than {T2_MAX_SPEED} m/s.\n")
    md.append("| model | training data | " + " | ".join(sets) + " |")
    md.append("|---|---|" + "---|" * len(sets))
    for k in KINDS:
        for var in ("full", "slow"):
            row = []
            for s in sets:
                g = get(k, var, s)
                row.append("–" if g is None else f"{np.median(g[1][:, H_MAIN]) * 100:.2f}")
            md.append(f"| {LABEL[k]} | {var} | " + " | ".join(row) + " |")

    # ---------------------------------------------------------------- wilcoxon
    md.append(f"\n## Paired comparison (Wilcoxon signed-rank, per flight, position error at "
              f"{H_TXT[20]})\n")
    md.append("| set | comparison | data | flights where A is better | median ratio A/B | p |")
    md.append("|---|---|---|---|---|---|")
    for s in sets:
        for a, b in (("phys_mlp", "mlp"), ("phys_mlp", "phys"), ("mlp", "phys")):
            for var in ("full", "slow"):
                ga, gb = get(a, var, s), get(b, var, s)
                if ga is None or gb is None or len(ga[0]) < 5:
                    continue
                ea, eb = ga[1][:, H_MAIN], gb[1][:, H_MAIN]
                p = wilcoxon(ea, eb).pvalue
                md.append(f"| {s} | {LABEL[a]} vs {LABEL[b]} | {var} | "
                          f"{np.mean(ea < eb):.0%} | {np.median(ea / eb):.2f} | {p:.1e} |")

    # ---------------------------------------------------------------- T4 per scenario
    if "T4" in flights:
        md.append(f"\n## T4 per scenario (position error [cm] at {H_TXT[20]}, median)\n")
        scen = np.array([f["scenario"] for f in flights["T4"]])
        md.append("| scenario | flights | " + " | ".join(LABEL[k] for k in KINDS) + " |")
        md.append("|---|---|" + "---|" * len(KINDS))
        cols = {k: get(k, "full", "T4") for k in KINDS}
        for sc_name in dict.fromkeys(scen):
            row = []
            n = 0
            for k in KINDS:
                g = cols[k]
                if g is None:
                    row.append("–")
                    continue
                sel = scen[g[0]] == sc_name
                n = int(sel.sum())
                row.append(f"{np.median(g[1][sel, H_MAIN]) * 100:.2f}")
            md.append(f"| {sc_name} | {n} | " + " | ".join(row) + " |")

    # ---------------------------------------------------------------- T3 per controller
    if "T3" in flights:
        md.append(f"\n## T3 per controller (position error [cm] at {H_TXT[20]}, median)\n")
        md.append("DSL-PID never flew in the training data.\n")
        ctrl = np.array([f["controller"] for f in flights["T3"]])
        md.append("| controller | flights | " + " | ".join(LABEL[k] for k in KINDS) + " |")
        md.append("|---|---|" + "---|" * len(KINDS))
        for c in dict.fromkeys(ctrl):
            row, n = [], 0
            for k in KINDS:
                g = get(k, "full", "T3")
                if g is None:
                    row.append("–")
                    continue
                sel = ctrl[g[0]] == c
                n = int(sel.sum())
                row.append(f"{np.median(g[1][sel, H_MAIN]) * 100:.2f}")
            md.append(f"| {c} | {n} | " + " | ".join(row) + " |")

    # ---------------------------------------------------------------- physics params
    md.append("\n## Identified physics parameters\n")
    md.append("True values of the simulator for comparison. The fitted values are *effective* "
              "parameters: thrust also absorbs the average inflow and ground effect, the "
              "rotor drag also the body drag – so they are not expected to match exactly.\n")
    md.append("| parameter | PyBullet value | " + " | ".join(
        f"{m}" for m in ("phys full", "phys slow", "phys+mlp full (seed 0)")) + " |")
    md.append("|---|---|---|---|---|")
    models_dir = root / "models"
    ident = {}
    for name in ("phys_full", "phys_slow", "phys_mlp_full_s0"):
        p = models_dir / f"{name}.json"
        ident[name] = json.loads(p.read_text())["identified"] if p.exists() else {}
    for k, v in TRUE_PARAMS.items():
        md.append(f"| {k} | {v:.3e} | " + " | ".join(
            f"{ident[n][k]:.3e}" if k in ident[n] else "–" for n in ident) + " |")

    # ---------------------------------------------------------------- training / compute
    timing_p = root / "eval" / "timing.json"
    if timing_p.exists():
        timing = json.loads(timing_p.read_text())
        md.append("\n## Compute\n")
        md.append("Rollout of 512 command sequences × 20 steps (incl. 8 warm-up steps), one "
                  "CPU thread (i5-7300U), torch eager mode. Training time: mean over seeds.\n")
        md.append("| model | parameters | rollout 512 × 20 [ms] | training [min] | epochs |")
        md.append("|---|---|---|---|---|")
        for k in KINDS:
            t = timing.get(LABEL[k])
            infos = [json.loads(p.read_text()) for p in models_dir.glob(f"{k}_full*.json")]
            if k == "phys":
                infos = [i for i in infos if i["kind"] == "phys"]
            if not t or not infos:
                continue
            tt = np.mean([i["train_time_s"] for i in infos]) / 60
            ep = np.mean([len(i["history"]["epoch"]) for i in infos])
            md.append(f"| {LABEL[k]} | {t['params']} | {t['ms_512x20']:.0f} | {tt:.0f} | "
                      f"{ep:.0f} |")
        search_p = models_dir / "search" / "search.json"
        if search_p.exists():
            sr = json.loads(search_p.read_text())
            md.append("\n**Window search** (validation loss, seed 0, shorter training): "
                      + ", ".join(f"`{k}` {v}" for k, v in sr["val_loss"].items())
                      + f" → chosen: {sr['best_window']}\n")

    # ---------------------------------------------------------------- figures
    # 1. error over horizon
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), sharey=True)
    t = np.array(HORIZONS) / 48
    for ax, s in zip(axes, sets, strict=True):
        for k in KINDS:
            for var, ls in (("full", "-"), ("slow", ":")):
                g = get(k, var, s)
                if g is None:
                    continue
                q1, med, q3 = np.percentile(g[1], [25, 50, 75], axis=0) * 100
                ax.plot(t, med, ls, marker="o", ms=3, color=COLOR[k],
                        label=f"{LABEL[k]} ({var})")
                if var == "full":
                    ax.fill_between(t, q1, q3, color=COLOR[k], alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{s}\n{SET_DESC.get(s, '')}", fontsize=10)
        ax.set_xlabel("prediction horizon [s]")
        ax.axvline(20 / 48, color="k", lw=0.5, ls="--")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("position error [cm]")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Open-loop position error over the horizon (median; band = IQR of the "
                 "full-data models; dotted = trained on slow flights only; dashed line = "
                 "0.42 s MPC horizon)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "error_over_horizon.png", dpi=120)
    plt.close(fig)

    # 2. error vs momentary speed and height (T1 + slow variant to show extrapolation)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, cond, bins, xl in ((axes[0], "speed", SPEED_BINS, "speed at start [m/s]"),
                               (axes[1], "z", Z_BINS, "height at start [m]")):
        centers = 0.5 * (np.array(bins[1:]) + np.array(bins[:-1]))
        for k in KINDS:
            for var, ls in (("full", "-"), ("slow", ":")):
                runs = res.get((k, var, "T1"))
                if not runs:
                    continue
                med = []
                for lo, hi in zip(bins[:-1], bins[1:], strict=True):
                    vals = np.concatenate([r["e_pos"][(r[cond] >= lo) & (r[cond] < hi),
                                                      H_MAIN] for r in runs])
                    med.append(np.median(vals) * 100 if len(vals) > 20 else np.nan)
                ax.plot(centers, med, ls, marker="o", color=COLOR[k], label=f"{LABEL[k]} ({var})")
        if cond == "speed":
            ax.axvline(T2_MAX_SPEED, color="k", lw=0.5, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(f"position error at {H_TXT[20]} [cm]")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    fig.suptitle("T1 rollouts grouped by the momentary condition (median over rollouts)")
    fig.tight_layout()
    fig.savefig(figs / "error_vs_condition.png", dpi=120)
    plt.close(fig)

    # 3. accuracy vs compute
    if timing_p.exists():
        fig, ax = plt.subplots(figsize=(5, 3.8))
        for k in KINDS:
            g = get(k, "full", "T1")
            if g is None or LABEL[k] not in timing:
                continue
            ax.scatter(timing[LABEL[k]]["ms_512x20"], np.median(g[1][:, H_MAIN]) * 100,
                       color=COLOR[k], s=60, label=LABEL[k])
        ax.set_xlabel("rollout 512 × 20 steps [ms], 1 thread")
        ax.set_ylabel(f"T1 position error at {H_TXT[20]} [cm]")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs / "accuracy_vs_compute.png", dpi=120)
        plt.close(fig)

    # 4. example rollouts
    ex_p = root / "eval" / "examples.npz"
    if ex_p.exists():
        with np.load(ex_p) as ex:
            names = [n for n in ("phys_full", "phys_mlp_full_s0", "mlp_full_s0") if n in ex]
            fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2),
                                     squeeze=False)
            pos = ex["pos"]
            for ax, n in zip(axes[0], names, strict=True):
                k = n.split("_full")[0]
                ax.plot(pos[:, 0], pos[:, 1], color="k", lw=1, label="true path")
                for i, snip in enumerate(ex[n]):
                    ax.plot(snip[:, 0], snip[:, 1], color=COLOR[k], lw=1.4,
                            label=f"{LABEL[k]}: {float(ex['horizon_s']):.0f} s predictions"
                            if i == 0 else None)
                    ax.plot(snip[0, 0], snip[0, 1], ".", color=COLOR[k], ms=4)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(pos[:, 0].min() - 0.5, pos[:, 0].max() + 0.5)
                ax.set_ylim(pos[:, 1].min() - 0.5, pos[:, 1].max() + 0.5)
                ax.set_title(LABEL[k])
                ax.set_xlabel("x [m]")
                ax.legend(fontsize=7, loc="upper right")
            axes[0][0].set_ylabel("y [m]")
            fig.suptitle(f"Open-loop predictions every 0.5 s along a T3 flight "
                         f"({ex['name']}), seed-0 models")
            fig.tight_layout()
            fig.savefig(figs / "example_rollouts.png", dpi=120)
            plt.close(fig)

    training_curves(root / "models", figs, "mlp")

    md.append("\n## Figures\n")
    for f in ("error_over_horizon", "error_vs_condition", "accuracy_vs_compute",
              "example_rollouts", "training_mlp"):
        if (figs / f"{f}.png").exists():
            md.append(f"![{f}](figures/{f}.png)\n")
    (root / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {root / 'report.md'}")


if __name__ == "__main__":
    main()
