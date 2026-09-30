"""Command line entry point.

    python -m pcsurv.run --synthetic
    python -m pcsurv.run --expression data/tcga_expr.tsv --cdr data/TCGA-CDR.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import filter_usable, load_tcga, make_synthetic
from .evaluate import run_benchmark, summarise, to_markdown, win_rate
from .models import BASELINE, CONTROL, REGISTRY
from .plots import make_all


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pan-cancer survival model benchmark.")
    p.add_argument("--expression", type=Path, help="Genes x samples matrix (Xena).")
    p.add_argument("--cdr", type=Path, help="TCGA-CDR supplemental table.")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--endpoint", default="PFI", choices=["OS", "PFI", "DSS", "DFI"])
    p.add_argument("--name", default=None)
    p.add_argument("--models", nargs="+", default=sorted(REGISTRY))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--min-events", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    if args.synthetic:
        data = make_synthetic()
    elif args.expression and args.cdr:
        data = load_tcga(
            args.expression, args.cdr, endpoint=args.endpoint,
            name=args.name or f"tcga_{args.endpoint.lower()}",
        )
        data = filter_usable(data, min_events=args.min_events)
    else:
        p.error("Pass --synthetic, or both --expression and --cdr.")

    print(f"\n{data}\n")
    print(data.event_counts().to_string(index=False))
    print(f"\nCross-validating {len(args.models)} models "
          f"across {len(data.cancer_types)} cancer types\n")

    results = run_benchmark(data, args.models, n_splits=args.splits,
                            seeds=tuple(args.seeds))
    if results.empty:
        print("\nNo cancer type had enough events to evaluate.")
        return 1

    summary = summarise(results)
    wins_clinical = win_rate(results, BASELINE)
    wins_control = win_rate(results, CONTROL) if CONTROL in results.model.values else None

    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / f"{data.name}_folds.csv", index=False)
    summary.to_csv(args.out / f"{data.name}_summary.csv", index=False)
    wins_clinical.to_csv(args.out / f"{data.name}_vs_clinical.csv", index=False)
    table = to_markdown(summary)
    (args.out / f"{data.name}_summary.md").write_text(table + "\n")

    if not args.no_plots:
        for fig in make_all(results, args.out / "figures", data.name):
            print(f"  figure: {fig}")

    print(f"\nPooled across cancer types — {data.name}:\n")
    print(table)
    print("\nAgainst the clinical-only baseline:\n")
    print(wins_clinical.to_string(index=False))
    if wins_control is not None:
        print("\nAgainst the random signature control:\n")
        print(wins_control.to_string(index=False))
    print(f"\nWritten to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
