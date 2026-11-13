#!/usr/bin/env python3
"""Plot benchmark results from `./build/gemm all --csv`.

Usage:
    python scripts/plot_results.py results/rtx5080_laptop.csv --gpu "RTX 5080 Laptop"
    python scripts/plot_results.py results/rtx5080_laptop.csv --gpu "RTX 5080 Laptop" --update-readme

Writes, next to the CSV:
    plots/<name>_ladder_{light,dark}.png   every kernel at one size, against cuBLAS
    plots/<name>_sizes_{light,dark}.png    best kernel against cuBLAS across sizes
    <name>.md                              results table in markdown

Needs only matplotlib:  pip install matplotlib
"""
import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Short labels for the x axis. Unknown names fall back to the CSV name.
LABELS = {
    "naive": "naive",
    "coalesced": "coalesced",
    "tiling": "shared-mem\ntiling",
    "1d_thread": "1D thread\ntiling",
    "2d_thread": "2D thread\ntiling",
    "vectorized_smem_float4": "float4\nloads",
    "warp_tiling": "warp\ntiling",
}

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e",
                  grid="#e6e5e1", mine="#2a78d6", ref="#52514e", faint="#c9c8c2"),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7",
                 grid="#33332f", mine="#3987e5", ref="#c3c2b7", faint="#55554f"),
}


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("pass") != "1" or not r.get("tflops"):
                print(f"skipping failed row: kernel {r.get('id')} at {r.get('M')}", file=sys.stderr)
                continue
            rows.append(dict(
                id=int(r["id"]), name=r["name"],
                M=int(r["M"]), N=int(r["N"]), K=int(r["K"]),
                tflops=float(r["tflops"]), cublas=float(r["cublas_tflops"]),
            ))
    if not rows:
        sys.exit(f"no passing rows in {path}")
    return rows


def cube_sizes(rows):
    return sorted({r["M"] for r in rows if r["M"] == r["N"] == r["K"]})


def at_size(rows, size):
    """Kernels at one cubic size, ordered by id, plus the cuBLAS number for that size."""
    sel = sorted((r for r in rows if r["M"] == r["N"] == r["K"] == size), key=lambda r: r["id"])
    if not sel:
        sys.exit(f"no rows for size {size}; sizes in file: {cube_sizes(rows)}")
    refs = [r["cublas"] for r in sel]
    ref = statistics.median(refs)
    # In one clean run every row of a size carries the same cuBLAS number.
    if (max(refs) - min(refs)) / ref > 0.10:
        print(f"WARNING: cuBLAS at {size} ranges {min(refs):.1f}-{max(refs):.1f} TFLOP/s across rows. "
              "These rows were not measured in one run under the same conditions; "
              "re-run `./build/gemm all --csv` in a single session before publishing.", file=sys.stderr)
    return sel, ref


def setup_fonts():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Inter", "Helvetica Neue", "Arial", "DejaVu Sans"):
        if fam in names:
            plt.rcParams["font.family"] = fam
            return


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=9, colors=t["muted"])
    ax.set_ylabel("TFLOP/s", fontsize=9, color=t["muted"])


def titles(fig, t, title, subtitle, note):
    fig.text(0.06, 0.94, title, fontsize=15, fontweight="bold", color=t["text"], ha="left", va="top")
    fig.text(0.06, 0.875, subtitle, fontsize=9.5, color=t["muted"], ha="left", va="top")
    if note:
        fig.text(0.06, 0.025, note, fontsize=7.5, color=t["muted"], ha="left", va="bottom")


def plot_ladder(rows, size, gpu, t, out, note):
    sel, ref = at_size(rows, size)
    xs = list(range(len(sel)))
    ys = [r["tflops"] for r in sel]

    fig = plt.figure(figsize=(8, 4.5), dpi=200, facecolor=t["surface"])
    ax = fig.add_axes([0.08, 0.20, 0.87, 0.52])
    style_axes(ax, t)

    ax.axhline(ref, color=t["ref"], linewidth=1.4, linestyle=(0, (5, 4)),
               label=f"cuBLAS ({ref:.1f} TFLOP/s)")
    ax.plot(xs, ys, color=t["mine"], linewidth=2.2, marker="o", markersize=7,
            markeredgecolor=t["surface"], markeredgewidth=1.6, label="my kernels", zorder=3)

    top = max(ref, max(ys)) * 1.10
    for i, (x, y) in enumerate(zip(xs, ys)):
        txt = f"{y:.2f}" if y < 10 else f"{y:.1f}"
        # put the label under the point when the reference line is too close above it
        below = (ref - y) < top * 0.07 and y <= ref
        ax.annotate(txt, (x, y), xytext=(0, -13 if below else 9), textcoords="offset points",
                    ha="center", va="top" if below else "bottom", fontsize=9,
                    fontweight="bold" if i == len(xs) - 1 else "normal", color=t["text"])

    last = sel[-1]
    best = max(sel, key=lambda r: r["tflops"])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['id']:02d}\n{LABELS.get(r['name'], r['name'])}" for r in sel])
    ax.set_xlim(-0.45, len(xs) - 0.55)
    ax.set_ylim(0, top)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[::-1], labels[::-1], loc="lower left", frameon=False, fontsize=9,
                    handlelength=2.4, ncol=2, columnspacing=1.6, bbox_to_anchor=(-0.01, 1.02))
    for txt in leg.get_texts():
        txt.set_color(t["text"])

    titles(fig, t,
           f"Hand-written CUDA matrix multiply: {best['tflops'] / ref * 100:.0f}% of cuBLAS",
           f"FP32 SGEMM, {size} x {size}, {gpu}. "
           f"{last['tflops'] / sel[0]['tflops']:.0f}x faster than naive after {len(sel) - 1} steps.",
           note)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def plot_sizes(rows, size, gpu, t, out, note):
    sizes = cube_sizes(rows)
    sel, _ = at_size(rows, size)
    best = max(sel, key=lambda r: r["tflops"])
    by_kernel = {}
    for r in rows:
        if r["M"] == r["N"] == r["K"]:
            by_kernel.setdefault(r["id"], {})[r["M"]] = r["tflops"]
    ref = [at_size(rows, s)[1] for s in sizes]
    xs = list(range(len(sizes)))

    fig = plt.figure(figsize=(8, 4.5), dpi=200, facecolor=t["surface"])
    ax = fig.add_axes([0.08, 0.15, 0.87, 0.57])
    style_axes(ax, t)

    first = True
    for kid, vals in sorted(by_kernel.items()):
        if kid == best["id"]:
            continue
        ax.plot(xs, [vals.get(s) for s in sizes], color=t["faint"], linewidth=1.2,
                label="earlier kernels" if first else None, zorder=2)
        first = False
    ax.plot(xs, ref, color=t["ref"], linewidth=1.6, linestyle=(0, (5, 4)), marker="o",
            markersize=5, label="cuBLAS", zorder=3)
    mine = [by_kernel[best["id"]].get(s) for s in sizes]
    ax.plot(xs, mine, color=t["mine"], linewidth=2.4, marker="o", markersize=7,
            markeredgecolor=t["surface"], markeredgewidth=1.6,
            label=f"kernel {best['id']:02d} ({best['name'].replace('_', ' ')})", zorder=4)
    for x, y, c in zip(xs, mine, ref):
        if y is not None:
            ax.annotate(f"{y / c * 100:.0f}%", (x, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=t["text"])

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s} x {s}" for s in sizes])
    ax.set_xlim(-0.3, len(xs) - 0.7)
    ax.set_ylim(0, max(ref + [m for m in mine if m]) * 1.08)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (labels[i] == "earlier kernels", labels[i] == "cuBLAS"))
    leg = ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower left",
                    frameon=False, fontsize=9, handlelength=2.4, ncol=3, columnspacing=1.6,
                    bbox_to_anchor=(-0.01, 1.02))
    for txt in leg.get_texts():
        txt.set_color(t["text"])

    titles(fig, t, "Best kernel against cuBLAS across matrix sizes",
           f"FP32 SGEMM, square matrices, {gpu}. Percentages are my kernel as a share of cuBLAS.",
           note)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def markdown_table(rows, size, gpu):
    sizes = cube_sizes(rows)
    sel, ref_at = at_size(rows, size)
    by = {(r["id"], r["M"]): r["tflops"] for r in rows if r["M"] == r["N"] == r["K"]}
    naive = sel[0]["tflops"]
    head = ["Kernel"] + [f"{s}" for s in sizes] + [f"% of cuBLAS at {size}", "vs naive"]
    lines = [f"TFLOP/s on {gpu}, FP32, square matrices. Higher is better.", "",
             "| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    for r in sel:
        cells = [f"{by[(r['id'], s)]:.2f}" if (r["id"], s) in by else "-" for s in sizes]
        lines.append(f"| {r['id']:02d} {r['name'].replace('_', ' ')} | " + " | ".join(cells) +
                     f" | {r['tflops'] / ref_at * 100:.1f}% | {r['tflops'] / naive:.0f}x |")
    refs = [f"{at_size(rows, s)[1]:.2f}" for s in sizes]
    lines.append("| cuBLAS | " + " | ".join(refs) + f" | 100% | {ref_at / naive:.0f}x |")
    return "\n".join(lines) + "\n"


def update_readme(readme, block):
    text = readme.read_text()
    pat = re.compile(r"(<!-- results:start -->).*?(<!-- results:end -->)", re.S)
    if not pat.search(text):
        sys.exit(f"{readme} has no <!-- results:start --> / <!-- results:end --> markers")
    readme.write_text(pat.sub(lambda m: f"{m.group(1)}\n{block}\n{m.group(2)}", text))
    print(f"updated {readme}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--gpu", required=True, help='name shown on the plots, e.g. "RTX 5080 Laptop"')
    ap.add_argument("--size", type=int, default=4096, help="size for the headline plot (default 4096)")
    ap.add_argument("--note", default="", help="small footnote, e.g. CUDA version and date")
    ap.add_argument("--update-readme", action="store_true",
                    help="write plot + table between the results markers in README.md")
    args = ap.parse_args()

    rows = load(args.csv)
    setup_fonts()
    stem = args.csv.stem
    plots = args.csv.parent / "plots"
    plots.mkdir(exist_ok=True)

    for theme, t in THEMES.items():
        for kind, fn in (("ladder", plot_ladder), ("sizes", plot_sizes)):
            out = plots / f"{stem}_{kind}_{theme}.png"
            fn(rows, args.size, args.gpu, t, out, args.note)
            print(f"wrote {out}")

    table = markdown_table(rows, args.size, args.gpu)
    md = args.csv.with_suffix(".md")
    md.write_text(table)
    print(f"wrote {md}")

    if args.update_readme:
        root = Path(__file__).resolve().parent.parent
        rel = plots.resolve().relative_to(root).as_posix()

        def pic(kind, alt):
            return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{rel}/{stem}_{kind}_dark.png">\n'
                    f'  <img src="{rel}/{stem}_{kind}_light.png" alt="{alt}">\n</picture>')

        block = "\n\n".join([
            pic("ladder", f"TFLOP/s of each kernel against cuBLAS at {args.size}"),
            table.rstrip(),
            pic("sizes", "Best kernel against cuBLAS across matrix sizes"),
        ])
        update_readme(root / "README.md", block)


if __name__ == "__main__":
    main()