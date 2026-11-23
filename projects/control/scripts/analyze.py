"""Turn a benchmark CSV into tables, plots and the README results block.

    python projects/control/scripts/analyze.py --csv projects/control/results/benchmark/stage1.csv

Success = course feasible, no crash. RMSE statistics use successful episodes only and are
reported next to the success rate. Paired comparisons use the Wilcoxon signed-rank test on
episodes that both controllers finished (same course, start state and disturbance).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, NullFormatter  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from dronelab.core.eval.scenarios import SCENARIOS  # noqa: E402

# fixed categorical order (colour follows the controller, never its rank)
COLORS = {"dsl_pid": "#2a78d6", "pid": "#eb6834", "lqr": "#1baf7a", "lqi": "#eda100"}
MARKERS = {"dsl_pid": "o", "pid": "s", "lqr": "^", "lqi": "D"}
LABELS = {"dsl_pid": "DSL-PID (untuned anchor)", "pid": "PID (tuned)",
          "lqr": "LQR (tuned)", "lqi": "LQI (tuned)"}
SEQ = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"
NUM = ("speed", "level", "rmse_pos", "max_err", "final_err", "effort", "smoothness",
       "ctrl_ms_per_step",
       "gates_passed", "gates_total", "course_max_acc", "course_max_tilt_deg")
MARK = ("<!-- STAGE1-RESULTS:START -->", "<!-- STAGE1-RESULTS:END -->")


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in NUM:
                r[k] = float(r[k]) if r.get(k) not in (None, "") else np.nan
            r["feasible"] = r["feasible"] == "True"
            r["ok"] = r["feasible"] and r.get("crashed") in ("0.0", "0", "False")
            r["all_gates"] = r.get("all_gates") in ("True", "1", "1.0")
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def ctrl_order(rows):
    return [c for c in COLORS if any(r["controller"] == c for r in rows)]


def stats(rs: list[dict]) -> dict:
    flown = [r for r in rs if r["feasible"]]
    ok = [r["rmse_pos"] for r in flown if r["ok"]]
    return {
        "n": len(flown),
        "infeasible": len(rs) - len(flown),
        "success": len(ok) / len(flown) if flown else np.nan,
        "rmse_med": float(np.median(ok)) if ok else np.nan,
        "rmse_q1": float(np.percentile(ok, 25)) if ok else np.nan,
        "rmse_q3": float(np.percentile(ok, 75)) if ok else np.nan,
    }


def cm(x: float) -> str:
    return "–" if np.isnan(x) else f"{100 * x:.1f}"


def pct(x: float) -> str:
    return "–" if np.isnan(x) else f"{100 * x:.0f} %"


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def line(ax, c, x, y, lo=None, hi=None, dx=0.0):
    """Median line; optional interquartile range as thin error bars (slightly offset in x)."""
    x = np.asarray(x, dtype=float) + dx
    if lo is not None:
        y_, lo_, hi_ = (np.asarray(v, dtype=float) for v in (y, lo, hi))
        ax.errorbar(x, y_, yerr=[y_ - lo_, hi_ - y_], fmt="none", ecolor=COLORS[c],
                    elinewidth=0.8, alpha=0.5, capsize=0)
    ax.plot(x, y, color=COLORS[c], lw=1.2, marker=MARKERS[c], ms=4.5, label=LABELS[c],
            markeredgecolor="white", markeredgewidth=0.6)


def _series(rs_by_x):
    """Median and quartiles (in cm) for a list of row-lists."""
    st = [stats(rs) for rs in rs_by_x]
    return [100 * np.array([s[k] for s in st]) for k in ("rmse_med", "rmse_q1", "rmse_q3")]


# ---------------------------------------------------------------- part A
def part_a(rows, ctrls, out: Path) -> list[str]:
    a = [r for r in rows if r["part"] == "A" and r["family"] != "hover"]
    if not a:
        return []
    speeds = sorted({r["speed"] for r in a})
    fams = sorted({r["family"] for r in a}, key=["figure8", "helix", "waypoints", "gates"].index)
    md = ["### Part A – speed sweep (nominal)", "",
          "Median position RMSE [cm] of successful flights / success rate, over "
          f"{len(fams)} families × {len({r['seed'] for r in a})} seeds. "
          "Speeds above 1.5 m/s are outside the tuning range.", "",
          "| Controller | " + " | ".join(f"{v:g} m/s" for v in speeds) + " |",
          "|---|" + "---|" * len(speeds)]
    for c in ctrls:
        cells = []
        for v in speeds:
            s = stats([r for r in a if r["controller"] == c and r["speed"] == v])
            cells.append(f"{cm(s['rmse_med'])} / {pct(s['success'])}")
        md.append(f"| {LABELS[c]} | " + " | ".join(cells) + " |")
    infeas = {v: stats([r for r in a if r["controller"] == ctrls[0] and r["speed"] == v])
              for v in speeds}
    skipped = {v: s["infeasible"] for v, s in infeas.items() if s["infeasible"]}
    if skipped:
        md += ["", "Infeasible courses (rejected by the thrust/tilt check, not flown): "
               + ", ".join(f"{v:g} m/s: {n}" for v, n in skipped.items()) + "."]

    g = [r for r in a if r["family"] == "gates" and r["ok"]]
    if g:
        md += ["", "Gate course – all gates passed (successful flights): "
               + "; ".join(f"{LABELS[c].split(' (')[0]} "
                           f"{pct(np.mean([r['all_gates'] for r in g if r['controller'] == c]))}"
                           for c in ctrls) + "."]

    fig, axes = plt.subplots(1, len(fams), figsize=(3.4 * len(fams), 3.3), sharey=True)
    for ax, fam in zip(np.atleast_1d(axes), fams, strict=True):
        for k, c in enumerate(ctrls):
            med, q1, q3 = _series([[r for r in a if r["controller"] == c and r["family"] == fam
                                    and r["speed"] == v] for v in speeds])
            line(ax, c, speeds, med, q1, q3, dx=0.03 * (k - 1.5))
        ax.axvspan(1.5, max(speeds) + 0.2, color="#f0efec", zorder=0)
        ax.set_title(fam, color=INK, fontsize=11, loc="left")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        style(ax, "path speed [m/s]", "RMSE [cm], median + IQR" if fam == fams[0] else "")
    np.atleast_1d(axes)[-1].text(max(speeds) + 0.1, ax.get_ylim()[0], "extrapolation",
                                 ha="right", va="bottom", color=MUTED, fontsize=8)
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="upper center",
               ncol=len(ctrls), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "speed_sweep.png", dpi=130)
    plt.close(fig)
    return md + ["", f"![speed sweep]({link(out / 'speed_sweep.png')})", ""]


# ---------------------------------------------------------------- part B
B_FAMS = ("hover", "figure8", "gates")
KINDS = ["mass", "wind", "gust", "motor_lag", "sensor_noise"]
XLAB = {"mass": "mass scale", "wind": "wind force [N]", "gust": "gust std [N]",
        "motor_lag": "motor time constant [s]", "sensor_noise": "noise level (1-3)"}


def part_b(rows, ctrls, out: Path) -> list[str]:
    b = [r for r in rows if r["part"] == "B"]
    if not b:
        return []
    present = {r["scenario"] for r in b}
    scen = [n for n in SCENARIOS if n in present]  # fixed protocol order
    fams = [f for f in B_FAMS if any(r["family"] == f for r in b)]
    short = [LABELS[c].split(" (")[0] for c in ctrls]
    md = ["### Part B – disturbances (1 m/s), per course family", "",
          "Success rate and median RMSE [cm] of successful flights; in brackets the RMSE "
          "relative to the nominal scenario of the same family (1.0 = no degradation). Hover "
          "includes the take-off from 0.1 m, so its RMSE is an order of magnitude larger.", ""]
    ratio = np.full((len(fams), len(ctrls), len(scen)), np.nan)
    succ = np.full_like(ratio, np.nan)
    for fi, fam in enumerate(fams):
        bf = [r for r in b if r["family"] == fam]
        md += [f"<details><summary><b>{fam}</b></summary>", "",
               "| Scenario | " + " | ".join(short) + " |", "|---|" + "---|" * len(ctrls)]
        for j, s_name in enumerate(scen):
            cells = []
            for i, c in enumerate(ctrls):
                nom = stats([r for r in bf if r["controller"] == c and r["scenario"] == "nominal"])
                st = stats([r for r in bf if r["controller"] == c and r["scenario"] == s_name])
                ratio[fi, i, j] = st["rmse_med"] / nom["rmse_med"]
                succ[fi, i, j] = st["success"]
                rel = "" if np.isnan(ratio[fi, i, j]) else f" ({ratio[fi, i, j]:.2f})"
                cells.append(f"{pct(st['success'])}, {cm(st['rmse_med'])}{rel}")
            md.append(f"| {s_name} | " + " | ".join(cells) + " |")
        md += ["", "</details>", ""]

    # heatmaps: one row per family, shared colour scale (log2 of the RMSE ratio)
    vmax = max(2.0, float(np.nanmax(np.log2(ratio))))
    fig, axes = plt.subplots(len(fams), 1, figsize=(0.55 * len(scen) + 2.4,
                                                    (0.42 * len(ctrls) + 0.5) * len(fams) + 1.4),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for fi, (ax, fam) in enumerate(zip(axes, fams, strict=True)):
        im = ax.imshow(np.log2(ratio[fi]), cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
        for i, j in itertools.product(range(len(ctrls)), range(len(scen))):
            v = ratio[fi, i, j]
            dark = bool(np.isfinite(v) and np.log2(v) > 0.6 * vmax)
            txt = "–" if not np.isfinite(v) else f"{v:.1f}"
            if succ[fi, i, j] < 1:
                txt += f"\n{100 * succ[fi, i, j]:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if dark else INK)
        ax.set_yticks(range(len(ctrls)), short, color=INK, fontsize=8)
        ax.set_title(fam, color=INK, fontsize=10, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[-1].set_xticks(range(len(scen)), scen, rotation=60, ha="right", fontsize=8,
                        color=MUTED)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("log2(RMSE / RMSE nominal)", color=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("Degradation per family; % = success rate (if < 100 %)", color=INK,
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(out / "disturbance_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # degradation curves: rows = families, columns = disturbance kinds, median + IQR
    fig, axes = plt.subplots(len(fams), len(KINDS), figsize=(3.0 * len(KINDS), 2.6 * len(fams)),
                             sharex="col", squeeze=False)
    for fi, fam in enumerate(fams):
        bf = [r for r in b if r["family"] == fam]
        for ki, kind in enumerate(KINDS):
            ax = axes[fi, ki]
            lv = sorted({r["level"] for r in bf if r["kind"] == kind})
            xs = [1.0 if kind == "mass" else 0.0, *lv]
            order = np.argsort(xs)
            step = 0.012 * (max(xs) - min(xs))
            for k, c in enumerate(ctrls):
                groups = [[r for r in bf if r["controller"] == c and r["scenario"] == "nominal"]]
                groups += [[r for r in bf if r["controller"] == c and r["kind"] == kind
                            and r["level"] == v] for v in lv]
                med, q1, q3 = (v[order] for v in _series(groups))
                line(ax, c, np.array(xs)[order], med, q1, q3, dx=step * (k - 1.5))
            if fi == 0:
                ax.set_title(kind.replace("_", " "), color=INK, fontsize=11, loc="left")
            style(ax, XLAB[kind] if fi == len(fams) - 1 else "",
                  f"{fam}\nRMSE [cm]" if ki == 0 else "")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=len(ctrls),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "degradation.png", dpi=130)
    plt.close(fig)
    return md + [f"![heatmap]({link(out / 'disturbance_heatmap.png')})", "",
                 "Degradation curves (median, bars = interquartile range of the 5 seeds; "
                 "missing points = no successful flight):", "",
                 f"![degradation]({link(out / 'degradation.png')})", ""]


# ---------------------------------------------------------------- paired tests
def paired(rows, ctrls) -> list[str]:
    key = lambda r: (r["part"], r["scenario"], r["family"], r["speed"], r["seed"])  # noqa: E731
    by = defaultdict(dict)
    for r in rows:
        if r["ok"]:
            by[key(r)][r["controller"]] = r["rmse_pos"]
    md = ["### Paired comparison (same course, seed and disturbance; both controllers "
          "finished)", "",
          "Hover and path courses are tested separately because their RMSE scales differ.", "",
          "| Courses | A vs B | n | median(RMSE_A − RMSE_B) [cm] | A better in | Wilcoxon p |",
          "|---|---|---|---|---|---|"]
    groups = {"hover": lambda k: k[2] == "hover", "path courses": lambda k: k[2] != "hover"}
    for gname, sel in groups.items():
        for x, y in itertools.combinations(ctrls, 2):
            d = np.array([v[x] - v[y] for k, v in by.items() if sel(k) and x in v and y in v])
            if len(d) < 10 or not np.any(d):
                continue
            p = wilcoxon(d).pvalue
            md.append(f"| {gname} | {x} vs {y} | {len(d)} | {100 * np.median(d):+.2f} | "
                      f"{100 * np.mean(d < 0):.0f} % | {p:.1e} |")
    return md + [""]


def tuning_table(tdir: Path, ctrls) -> list[str]:
    rows = []
    for c in ctrls:
        f = tdir / c / "summary.json"
        if f.exists():
            s = json.loads(f.read_text())
            d, b = s["validation_defaults"], s["validation_selected"]
            rows.append(f"| {LABELS[c].split(' (')[0]} | {s['n_trials']} | {d['J']:.4f} | "
                        f"{b['J']:.4f} | {cm(d['rmse'])} → {cm(b['rmse'])} | "
                        f"{s['cpu_hours']:.2f} |")
    if not rows:
        return []
    return ["### Tuning (validation set, 40 episodes, seeds 100–139)", "",
            "| Controller | Trials | J default | J tuned | mean RMSE [cm] | CPU-h |",
            "|---|---|---|---|---|---|", *rows, ""]


# ---------------------------------------------------------------- tuning convergence
def tuning_convergence(tdir: Path, ctrls, out: Path) -> list[str]:
    """Best-so-far objective over the trial number: is the tuning budget sufficient?"""
    data = {}
    for c in ctrls:
        f = tdir / c / "trials.csv"
        if f.exists():
            with f.open() as fh:
                tr = sorted((int(r["trial"]), float(r["J"])) for r in csv.DictReader(fh))
            data[c] = np.array(tr)
    if not data:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    md = []
    for c, tr in data.items():
        best = np.minimum.accumulate(tr[:, 1])
        ax.scatter(tr[:, 0], tr[:, 1], s=5, color=COLORS[c], alpha=0.25, linewidths=0)
        ax.step(tr[:, 0], best, where="post", color=COLORS[c], lw=1.2, label=LABELS[c])
        n = len(best)
        half = best[n // 2 - 1]
        md.append(f"{LABELS[c].split(' (')[0]}: best J {best[-1]:.3f}, "
                  f"{100 * (half - best[-1]) / half:.1f} % gained in the second half")
    hi = max(np.percentile(tr[:, 1], 90) for tr in data.values())
    lo = min(tr[:, 1].min() for tr in data.values())
    ax.set_ylim(lo * 0.95, hi)
    style(ax, "trial", "objective J (lower is better)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Tuning convergence (line = best so far, dots = single trials)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "tuning_convergence.png", dpi=130)
    plt.close(fig)
    return ["Tuning convergence – " + "; ".join(md) + ".", "",
            f"![tuning convergence]({link(out / 'tuning_convergence.png')})", ""]


# ---------------------------------------------------------------- accuracy vs smoothness
def pareto(rows, ctrls, out: Path) -> list[str]:
    a = [r for r in rows if r["part"] == "A" and r["family"] != "hover" and r["ok"]]
    if not a:
        return []
    order = ("figure8", "helix", "waypoints", "gates")
    fams = [f for f in order if any(r["family"] == f for r in a)]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.4 * len(fams), 3.3), sharex=True,
                             sharey=True)
    for ax, fam in zip(np.atleast_1d(axes), fams, strict=True):
        for c in ctrls:
            rs = [r for r in a if r["controller"] == c and r["family"] == fam]
            ax.scatter([100 * r["rmse_pos"] for r in rs], [r["smoothness"] for r in rs], s=14,
                       color=COLORS[c], marker=MARKERS[c], edgecolors="white", linewidths=0.4,
                       label=LABELS[c], alpha=0.85)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FormatStrFormatter("%g"))
            axis.set_minor_formatter(NullFormatter())
        ax.set_title(fam, color=INK, fontsize=11, loc="left")
        style(ax, "RMSE [cm]", "motor command change per step\n(fraction of max rpm)"
              if fam == fams[0] else "")
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="upper center",
               ncol=len(ctrls), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "accuracy_vs_smoothness.png", dpi=130)
    plt.close(fig)
    return ["### Accuracy vs. smoothness (Part A, one dot per flight)", "",
            "Lower left is better. A controller can buy accuracy with more aggressive motor "
            "commands; the tuning objective weighs this trade-off (w_s = 0.3).", "",
            f"![accuracy vs smoothness]({link(out / 'accuracy_vs_smoothness.png')})", ""]


# ---------------------------------------------------------------- motor saturation
MAX_RPM = 21702.6  # CF2X (gym-pybullet-drones)
DSL_MIN_RPM = 0.2685 * 20000 + 4070.3  # DSLPIDControl clips PWM at 20000


def _saturation(f: Path) -> tuple[dict, float]:
    z = np.load(f, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    rpm = z["rpm_cmd"]
    lo = DSL_MIN_RPM if meta["controller"] == "dsl_pid" else 0.0
    sat = np.any((rpm >= 0.995 * MAX_RPM) | (rpm <= lo + 0.005 * MAX_RPM), axis=1)
    return meta, float(sat.mean()) if len(sat) else np.nan


def saturation(flights: Path, ctrls, out: Path) -> list[str]:
    files = sorted(flights.glob("*.npz"))
    if not files:
        return []
    res = [_saturation(f) for f in files]
    md = ["### Motor saturation", "",
          "Share of control steps in which at least one motor command sits at its limit "
          "(max rpm, or the minimum: 0 rpm for own controllers, PWM floor ≈ 9 440 rpm for "
          "DSL-PID). Mean over flights; crashed flights included.", ""]
    a = [(m, s) for m, s in res if m["part"] == "A" and m["family"] != "hover"]
    speeds = sorted({m["speed"] for m, _ in a})
    md += ["| Controller | " + " | ".join(f"{v:g} m/s" for v in speeds) + " |",
           "|---|" + "---|" * len(speeds)]
    for c in ctrls:
        cells = [pct(np.mean([s for m, s in a if m["controller"] == c and m["speed"] == v]
                               or [np.nan])) for v in speeds]
        md.append(f"| {LABELS[c].split(' (')[0]} | " + " | ".join(cells) + " |")

    b = [(m, s) for m, s in res if m["part"] == "B" and m["family"] != "hover"]
    scen = [n for n in SCENARIOS if any(m["scenario"] == n for m, _ in b)]
    fig, ax = plt.subplots(figsize=(0.5 * len(scen) + 2, 3.2))
    w = 0.8 / len(ctrls)
    for k, c in enumerate(ctrls):
        y = [100 * np.mean([s for m, s in b if m["controller"] == c and m["scenario"] == n]
                             or [np.nan]) for n in scen]
        ax.bar(np.arange(len(scen)) + (k - (len(ctrls) - 1) / 2) * w, y, width=w * 0.9,
               color=COLORS[c], label=LABELS[c])
    ax.set_xticks(range(len(scen)), scen, rotation=60, ha="right", fontsize=8)
    style(ax, "", "steps saturated [%]")
    ax.legend(frameon=False, fontsize=8, ncol=len(ctrls), loc="upper left")
    ax.set_title("Motor saturation under disturbances (figure8 + gates, 1 m/s, mean)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "saturation.png", dpi=130)
    plt.close(fig)
    md += ["", "Nominal flights stay below the limits at every speed – the feasibility check "
           "admits only courses that need ≤ 80 % of the maximum thrust. Saturation appears "
           "under disturbances, above all with motor lag:"]
    return md + ["", f"![saturation]({link(out / 'saturation.png')})", ""]


# ---------------------------------------------------------------- error vs. required acceleration
ACC_BINS = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf])  # [m/s^2]


def _load_flight(f: Path) -> tuple[dict, dict]:
    z = np.load(f, allow_pickle=True)
    return json.loads(str(z["meta"])), {k: z[k] for k in ("t", "pos", "ref_pos", "ref_vel")}


def accel_bins(flights: Path, ctrls, out: Path) -> list[str]:
    """Tracking error per control step, grouped by the reference acceleration at that step."""
    err = defaultdict(list)  # (controller, bin) -> errors [m]
    for f in sorted(flights.glob("*.npz")):
        meta, d = _load_flight(f)
        if meta.get("part") != "A" or meta["family"] == "hover" or len(d["t"]) < 3:
            continue
        acc = np.linalg.norm(np.gradient(d["ref_vel"], d["t"], axis=0), axis=1)
        e = np.linalg.norm(d["pos"] - d["ref_pos"], axis=1)
        idx = np.digitize(acc, ACC_BINS) - 1
        for b in np.unique(idx):
            err[(meta["controller"], int(b))].append(e[idx == b])
    if not err:
        return []
    nb = len(ACC_BINS) - 1
    labels = [f"{ACC_BINS[i]:g}–{ACC_BINS[i + 1]:g}" if np.isfinite(ACC_BINS[i + 1])
              else f"> {ACC_BINS[i]:g}" for i in range(nb)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    w = 0.8 / len(ctrls)
    counts = np.zeros(nb, dtype=int)
    for k, c in enumerate(ctrls):
        med, lo, hi = np.full(nb, np.nan), np.full(nb, np.nan), np.full(nb, np.nan)
        for b in range(nb):
            if (c, b) in err:
                e = 100 * np.concatenate(err[(c, b)])
                med[b], lo[b], hi[b] = np.median(e), *np.percentile(e, [25, 75])
                counts[b] = max(counts[b], len(e))
        x = np.arange(nb) + (k - (len(ctrls) - 1) / 2) * w
        ax.bar(x, med, width=w * 0.9, color=COLORS[c], label=LABELS[c])
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none", ecolor=INK, elinewidth=0.7,
                    alpha=0.6, capsize=0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_xticks(range(nb), [f"{lab}\n({n / 1000:.0f}k steps)" if n >= 1000 else
                              f"{lab}\n({n} steps)" for lab, n in zip(labels, counts,
                                                                      strict=True)], fontsize=8)
    style(ax, "reference acceleration [m/s²]", "position error [cm]")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ax.set_title("Error per control step vs. required acceleration "
                 "(Part A, all path courses and speeds)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "error_vs_acceleration.png", dpi=130)
    plt.close(fig)
    return ["### Error vs. required acceleration (per control step)", "",
            "Every control step of the nominal speed sweep is sorted by the reference "
            "acceleration at that moment (straights ≈ 0, tight turns and braking > 4 m/s²). "
            "Bars: median error, whiskers: interquartile range.", "",
            f"![error vs acceleration]({link(out / 'error_vs_acceleration.png')})", ""]


LINK_BASE = Path(".")  # image links are written relative to the README that embeds them


def link(p: Path) -> str:
    return Path(os.path.relpath(p, LINK_BASE)).as_posix()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="projects/control/results/benchmark/stage1.csv")
    ap.add_argument("--tuning", default="projects/control/results/tuning")
    ap.add_argument("--readme", default="projects/control/README.md")
    ap.add_argument("--flights", default=None, help="flight logs (default data/flights/<name>)")
    args = ap.parse_args()
    global LINK_BASE
    LINK_BASE = Path(args.readme).parent
    path = Path(args.csv)
    rows = load(path)
    out = path.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    ctrls = ctrl_order(rows)
    ms = {c: np.nanmedian([r["ctrl_ms_per_step"] for r in rows if r["controller"] == c])
          for c in ctrls}
    flights = Path(args.flights or f"data/flights/{path.stem}")
    md = [*tuning_table(Path(args.tuning), ctrls),
          *tuning_convergence(Path(args.tuning), ctrls, out), *part_a(rows, ctrls, out),
          *pareto(rows, ctrls, out), *accel_bins(flights, ctrls, out),
          *part_b(rows, ctrls, out),
          *saturation(flights, ctrls, out), *paired(rows, ctrls),
          "Controller compute time per step (median): "
          + ", ".join(f"{c} {v:.3f} ms" for c, v in ms.items()) + ".",
          "", f"_{len(rows)} episodes, git {rows[0]['git']}, config {rows[0]['config_hash']}_"]
    text = "\n".join(md)
    # report.md sits next to the images: strip the README-relative prefix
    prefix = link(out) + "/"
    (out / "report.md").write_text(text.replace(f"]({prefix}", "]("), encoding="utf-8")
    print(text.encode("ascii", "replace").decode())

    readme = Path(args.readme)
    if readme.exists():
        r = readme.read_text(encoding="utf-8")
        if MARK[0] in r:
            r = re.sub(re.escape(MARK[0]) + ".*?" + re.escape(MARK[1]),
                       lambda _: f"{MARK[0]}\n{text}\n{MARK[1]}", r, flags=re.S)
            readme.write_text(r, encoding="utf-8")
            print("README block updated")


if __name__ == "__main__":
    main()
