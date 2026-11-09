"""Aggregate subpixel convergence JSON lines into Markdown tables, orders and plots."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# (case, metric key, group keys, label)
METRICS = {
    "planar": [("rel_error", "frequency error")],
    "pair": [("beta_error", "even-mode beta error vs continuum"), ("fdtd_minus_analog", "FDTD vs Yee analog")],
    "sphere": [("err_csca_flux", "C_sca (box flux)"), ("err_csca_farfield", "C_sca (far-field quadrature)"),
               ("err_pattern_E", "E-plane sigma(theta) rel. L2"), ("err_pattern_H", "H-plane sigma(theta) rel. L2"),
               ("err_back", "backscatter sigma")],
    "pec": [("err_csca_flux", "C_sca (box flux)"), ("err_pattern_E", "E-plane sigma(theta) rel. L2"),
            ("err_pattern_H", "H-plane sigma(theta) rel. L2"), ("err_back", "backscatter sigma")],
    "cylinder": [("err_width", "scattering width")],
}
GROUP_KEYS = {
    "planar": ("mode", "eps_r"),
    "pair": (),
    "sphere": ("eps_r", "size_parameter"),
    "pec": ("size_parameter",),
    "cylinder": ("polarization", "eps_r", "size_parameter"),
}
COLORS = {"staircase": "#7f7f7f", "arithmetic": "#1f77b4", "polarized": "#d62728", "conformal": "#2ca02c"}


def load(paths):
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def fit_order(resolutions, errors):
    """Least-squares slope of log|err| vs log(resolution) (positive = convergent)."""
    r = np.asarray(resolutions, dtype=float)
    e = np.abs(np.asarray(errors, dtype=float))
    keep = e > 0
    if keep.sum() < 2:
        return float("nan")
    slope = np.polyfit(np.log(r[keep]), np.log(e[keep]), 1)[0]
    return float(-slope)


def summarize(rows, case, metric):
    rows = [r for r in rows if not r.get("samples_study")]
    """{group: {(method, samples): {resolution: dict(mean_abs, signed, max_abs, n, wall, mem)}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    buckets = defaultdict(list)
    for row in rows:
        if row.get("case") != case or metric not in row:
            continue
        group = tuple(row.get(k) for k in GROUP_KEYS[case])
        buckets[(group, row["method"], row.get("samples"), row["resolution"])].append(row)
    for (group, method, samples, resolution), items in buckets.items():
        values = np.asarray([item[metric] for item in items], dtype=float)
        out[group][(method, samples)][resolution] = dict(
            mean_abs=float(np.mean(np.abs(values))), signed=float(np.mean(values)), max_abs=float(np.max(np.abs(values))),
            spread=float(np.ptp(values)) if values.size > 1 else 0.0, n=int(values.size),
            wall=float(np.mean([item.get("wall_s", float("nan")) for item in items])),
            mem=float(np.max([item.get("peak_gpu_mem_gb", float("nan")) for item in items])),
            cells=int(items[0].get("cells", 0)), steps=int(items[0].get("time_steps", 0)))
    return out


def markdown_table(summary, *, signed=False):
    lines = []
    for group, methods in sorted(summary.items(), key=lambda kv: str(kv[0])):
        resolutions = sorted({r for data in methods.values() for r in data})
        header = "| method | " + " | ".join(f"{r:g}" for r in resolutions) + " | order (all) | order (last 3) |"
        lines.append(f"\n{group}\n")
        lines.append(header)
        lines.append("|" + "---|" * (len(resolutions) + 3))
        for (method, samples), data in sorted(methods.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            cells = []
            for r in resolutions:
                if r in data:
                    value = data[r]["signed"] if signed else data[r]["mean_abs"]
                    cells.append(f"{value:+.2e}" if signed else f"{value:.2e}")
                else:
                    cells.append("")
            rs = sorted(data)
            order_all = fit_order(rs, [data[r]["mean_abs"] for r in rs])
            order_tail = fit_order(rs[-3:], [data[r]["mean_abs"] for r in rs[-3:]]) if len(rs) >= 3 else float("nan")
            label = method if method in ("staircase", "conformal") else f"{method} (s={samples})"
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {order_all:.2f} | {order_tail:.2f} |")
    return "\n".join(lines)


def cost_table(summary):
    lines = ["| group | method | resolution | cells | steps | wall s | peak GPU GB |", "|---|---|---|---|---|---|---|"]
    for group, methods in sorted(summary.items(), key=lambda kv: str(kv[0])):
        for (method, samples), data in sorted(methods.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            for r in sorted(data):
                d = data[r]
                label = method if method in ("staircase", "conformal") else f"{method} (s={samples})"
                lines.append(f"| {group} | {label} | {r:g} | {d['cells']} | {d['steps']} | {d['wall']:.1f} | {d['mem']:.3f} |")
    return "\n".join(lines)


def plot(summary, path, title, xlabel):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted(summary.items(), key=lambda kv: str(kv[0]))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.2), squeeze=False)
    for ax, (group, methods) in zip(axes[0], groups):
        for (method, samples), data in sorted(methods.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            rs = sorted(data)
            ys = [data[r]["mean_abs"] for r in rs]
            ax.loglog(rs, ys, "o-", color=COLORS.get(method, "k"), ms=4, lw=1.4,
                      label=f"{method} ({fit_order(rs, ys):.1f})")
        rs_all = sorted({r for data in methods.values() for r in data})
        if rs_all:
            ref_y = max(d[rs_all[0]]["mean_abs"] for d in methods.values() if rs_all[0] in d)
            for order, style in ((1, ":"), (2, "--")):
                ax.loglog(rs_all, [ref_y * (rs_all[0] / r) ** order for r in rs_all], style, color="#bbbbbb", lw=1,
                          label=f"O(h^{order})")
        ax.set_title(", ".join(str(g) for g in group) if group else title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("|relative error|", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=6, loc="lower left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


XLABEL = {"planar": "cells across the plate gap", "pair": "1 / transverse dx (1/mm)",
          "sphere": "cells per wavelength in the material", "pec": "cells per wavelength",
          "cylinder": "cells per wavelength in the material"}


def attach_pair_reference(rows, reference):
    """FDTD-equivalent even-mode beta error at the measured cavity frequency.

    The x cavity fixes kx = pi / L; the FDTD line therefore has beta = kx at its
    resonance f. The reference beta(f) is the lossless continuum Richardson
    value, interpolated (quadratic) over the reference frequencies. The Yee
    analog (same transverse nodes, external solver) is moved to f with the
    continuum dispersion slope and mapped through the FDTD x/t steps.
    """
    freqs = np.asarray(reference["frequencies_hz"], dtype=float)
    betas = np.asarray([item["value"] for item in reference["continuum"]["substrate"]], dtype=float)
    coeff = np.polyfit(freqs, betas, 2)
    f0 = freqs[len(freqs) // 2]
    slope = np.polyval(np.polyder(coeff), f0) * f0 / np.polyval(coeff, f0)  # d ln beta / d ln f
    for row in rows:
        if row.get("case") != "pair":
            continue
        f = row["f_even_hz"]
        beta_ref = float(np.polyval(coeff, f))
        row["beta_reference"] = beta_ref
        row["beta_error"] = row["kx_rad_per_m"] / beta_ref - 1.0
        analog = reference["analog"]["substrate"].get(str(row["resolution"]))
        if analog is not None:
            dl, dt = row["longitudinal_dl_m"], row["dt_s"]
            f_tilde = math.sin(math.pi * f * dt) / (math.pi * dt)
            beta2d = analog * (f_tilde / f0) ** slope
            kx_analog = 2.0 / dl * math.asin(0.5 * beta2d * dl)
            row["beta_analog_equivalent"] = kx_analog
            row["analog_error"] = kx_analog / beta_ref - 1.0
            row["fdtd_minus_analog"] = row["kx_rad_per_m"] / kx_analog - 1.0
        row["resolution_inverse_mm"] = 1.0 / row["resolution"]


def _samples_count(value):
    """Sub-samples per cell along the interface normal ("(1, 1, S)" -> S)."""
    text = str(value).strip("()")
    return int(text.split(",")[-1]) if text else 1


def plot_samples(rows, path, resolution=40, eps_r=2.25):
    """Planar error vs sub-samples per cell along the normal at one resolution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("case") != "planar" or row["resolution"] != resolution or row["eps_r"] != eps_r:
            continue
        if row["method"] == "staircase":
            continue
        data[(row["mode"], row["method"])][_samples_count(row["samples"])].append(abs(row["rel_error"]))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    styles = {"TM0": "-", "TE0": "--"}
    for (mode, method), by_s in sorted(data.items()):
        counts = sorted(by_s)
        ax.loglog(counts, [np.mean(by_s[c]) for c in counts], "o" + styles[mode], color=COLORS[method], ms=4,
                  label=f"{mode} {method}")
    ax.set_xlabel("sub-samples per cell along the normal", fontsize=8)
    ax.set_ylabel("mean |relative error|", fontsize=8)
    ax.set_title(f"planar, eps {eps_r:g}, {resolution} cells across the gap", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(inputs, plots):
    references = [p for p in inputs if str(p).endswith(".json")]
    rows = []
    for path in [p for p in inputs if not str(p).endswith(".json")]:
        loaded = load([path])
        if "samples" in Path(path).name or "controls" in Path(path).name:
            for row in loaded:
                row["samples_study"] = True
        rows.extend(loaded)
    for path in references:
        attach_pair_reference(rows, json.loads(Path(path).read_text()))
    for row in rows:
        if row.get("case") == "pair":
            row["resolution"] = row.pop("resolution_inverse_mm", 1.0 / row["resolution"])
    sections = []
    if plots:
        plot_samples(rows, Path(plots) / "planar_samples.png")
    for case, metrics in METRICS.items():
        for metric, label in metrics:
            summary = summarize(rows, case, metric)
            if not summary:
                continue
            sections.append(f"## {case}: {label} (`{metric}`), mean |error| over offsets\n")
            sections.append(markdown_table(summary))
            if metric in ("rel_error", "err_csca_flux", "err_width", "beta_error"):
                sections.append(f"\nSigned mean:\n")
                sections.append(markdown_table(summary, signed=True))
                sections.append("\nCost:\n")
                sections.append(cost_table(summary))
            if plots:
                plot(summary, Path(plots) / f"{case}_{metric}.png", f"{case}: {label}", XLABEL[case])
    text = "\n".join(sections)
    print(text)
    return text
