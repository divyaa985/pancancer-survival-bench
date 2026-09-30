"""The evaluation protocol.

Survival data breaks the habits carried over from classification, and three of
those breakages are handled explicitly here.

**Censoring.** Most patients are alive at last follow-up, so their survival
time is a lower bound rather than a value. Treating the outcome as binary or
as a regression target throws that away and biases everything downstream.
Risk scores are therefore compared with concordance, which only asks whether
the patient who died first was given the higher risk, and only among pairs
where that comparison is defined.

**Harrell's C is biased.** It depends on the censoring distribution of the
test set, so the same model scores differently in two cohorts that differ only
in follow-up length. Uno's C reweights by the inverse probability of censoring
and is the better number when cohorts are compared. Both are reported, and
where they disagree, that disagreement is the finding.

**Folds must be stratified by event, not just by size.** A random split of a
cohort with 15% events can hand a fold almost no events, and concordance on
that fold is noise wearing a number's clothing.
"""

from __future__ import annotations

import time as _time
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

from .data import SurvivalData
from .models import build


@dataclass
class FoldResult:
    dataset: str
    cancer_type: str
    model: str
    fold: int
    seed: int
    harrell_c: float
    uno_c: float
    n_train: int
    n_test: int
    n_events_train: int
    n_events_test: int
    fit_seconds: float


def _uno_c(y_train: np.ndarray, y_test: np.ndarray, risk: np.ndarray) -> float:
    """Uno's C, restricted to a follow-up window both sets actually cover.

    The IPCW estimator is undefined beyond the last observed training time, so
    the evaluation is truncated there. Skipping this is a common source of
    crashes and, worse, of silently unstable numbers.
    """
    try:
        tau = min(y_train["time"].max(), y_test["time"].max())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(
                concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0]
            )
    except (ValueError, ZeroDivisionError):
        return float("nan")


def run_cancer_type(
    data: SurvivalData,
    cancer: str,
    model_names: list[str],
    n_splits: int = 5,
    seeds: tuple[int, ...] = (0, 1, 2),
    min_events_per_fold: int = 5,
) -> list[FoldResult]:
    """Repeated stratified cross-validation within one cancer type."""
    subset = data.for_cancer(cancer)
    y = subset.y
    results: list[FoldResult] = []

    if int(subset.event.sum()) < n_splits * min_events_per_fold:
        return results

    for seed in seeds:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(y)), subset.event)
        ):
            y_train, y_test = y[train_idx], y[test_idx]
            if y_train["event"].sum() < 2 or y_test["event"].sum() < 2:
                continue

            clin_train = subset.clinical.iloc[train_idx]
            clin_test = subset.clinical.iloc[test_idx]
            expr_train = subset.expression[train_idx]
            expr_test = subset.expression[test_idx]

            for model_name in model_names:
                start = _time.perf_counter()
                try:
                    model = build(model_name, seed=seed + 100 * fold).fit(
                        clin_train, expr_train, y_train
                    )
                    risk = model.predict_risk(clin_test, expr_test)
                except Exception as exc:  # noqa: BLE001 - a failed fit is a result
                    print(f"    {model_name} failed on {cancer} fold {fold}: {exc}")
                    continue
                elapsed = _time.perf_counter() - start

                harrell = concordance_index_censored(
                    y_test["event"], y_test["time"], risk
                )[0]

                results.append(
                    FoldResult(
                        dataset=data.name,
                        cancer_type=cancer,
                        model=model_name,
                        fold=fold,
                        seed=seed,
                        harrell_c=float(harrell),
                        uno_c=_uno_c(y_train, y_test, risk),
                        n_train=len(train_idx),
                        n_test=len(test_idx),
                        n_events_train=int(y_train["event"].sum()),
                        n_events_test=int(y_test["event"].sum()),
                        fit_seconds=round(elapsed, 2),
                    )
                )
    return results


def run_benchmark(
    data: SurvivalData,
    model_names: list[str],
    n_splits: int = 5,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    """Every cancer type, every model, every fold, every seed."""
    rows: list[FoldResult] = []
    for cancer in data.cancer_types:
        found = run_cancer_type(data, cancer, model_names, n_splits, seeds)
        if not found:
            print(f"  {cancer}: skipped, too few events")
            continue
        summary = (
            pd.DataFrame([asdict(r) for r in found])
            .groupby("model")["harrell_c"]
            .mean()
            .round(3)
        )
        best = summary.idxmax()
        print(f"  {cancer:<10} best: {best} ({summary[best]:.3f})")
        rows.extend(found)
    return pd.DataFrame([asdict(r) for r in rows])


def summarise(results: pd.DataFrame, metric: str = "harrell_c") -> pd.DataFrame:
    """Mean and spread per model, pooled across cancer types.

    Cancer types are averaged first so that a type with many folds does not
    dominate. The pooled number is a summary, not the headline: the per-cancer
    table is where the real variation lives.
    """
    per_cancer = results.groupby(["model", "cancer_type"])[metric].mean().reset_index()
    agg = (
        per_cancer.groupby("model")[metric]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "min": f"{metric}_min",
                "max": f"{metric}_max",
                "count": "n_cancer_types",
            }
        )
        .sort_values(f"{metric}_mean", ascending=False)
        .round(3)
    )
    seconds = results.groupby("model")["fit_seconds"].mean().round(2)
    return agg.join(seconds).reset_index()


def margin_over(results: pd.DataFrame, reference: str,
                metric: str = "harrell_c") -> pd.DataFrame:
    """Each model minus a reference model, matched on cancer type, fold and seed.

    Matching matters. Comparing pooled means across models hides the fact that
    a model can win on average while losing in most cancer types, simply by
    winning hugely in one.
    """
    keys = ["cancer_type", "fold", "seed"]
    ref = (
        results[results.model == reference]
        .set_index(keys)[metric]
        .rename("reference")
    )
    other = results[results.model != reference].set_index(keys)
    joined = other.join(ref, how="inner").reset_index()
    joined["margin"] = joined[metric] - joined["reference"]
    return joined


def win_rate(results: pd.DataFrame, reference: str,
             metric: str = "harrell_c") -> pd.DataFrame:
    """How often each model beats the reference, per cancer type.

    A mean margin of +0.02 could be a consistent small edge or one lucky
    cancer type. This distinguishes them.
    """
    joined = margin_over(results, reference, metric)
    per_cancer = (
        joined.groupby(["model", "cancer_type"])["margin"].mean().reset_index()
    )
    return (
        per_cancer.assign(beats_reference=lambda d: d.margin > 0)
        .groupby("model")
        .agg(
            mean_margin=("margin", "mean"),
            cancer_types_won=("beats_reference", "sum"),
            cancer_types_total=("beats_reference", "size"),
        )
        .assign(win_rate=lambda d: (d.cancer_types_won / d.cancer_types_total).round(3))
        .sort_values("mean_margin", ascending=False)
        .round(3)
        .reset_index()
    )


def to_markdown(summary: pd.DataFrame, metric: str = "harrell_c") -> str:
    header = "| model | mean C | ± sd | min | max | types | sec |"
    lines = [header, "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(
            f"| `{r.model}` | **{r[f'{metric}_mean']:.3f}** | {r[f'{metric}_std']:.3f} | "
            f"{r[f'{metric}_min']:.3f} | {r[f'{metric}_max']:.3f} | "
            f"{int(r.n_cancer_types)} | {r.fit_seconds:.1f} |"
        )
    return "\n".join(lines)
