"""Tests that run on CPU in seconds with no download."""

import numpy as np
import pytest

from pcsurv.data import filter_usable, make_synthetic
from pcsurv.evaluate import margin_over, run_benchmark, summarise, win_rate
from pcsurv.models import BASELINE, CONTROL, REGISTRY, build


def small():
    return make_synthetic(n_per_cancer=150, n_cancers=3, n_genes=120, seed=0)


def test_synthetic_has_censoring_and_events():
    """A survival dataset with no censoring is not testing survival code."""
    d = small()
    assert 0 < d.event.mean() < 1
    assert (d.time > 0).all()


def test_y_is_a_valid_structured_survival_array():
    d = small()
    y = d.y
    assert y.dtype.names == ("event", "time")
    assert y["event"].dtype == bool
    assert len(y) == len(d.time)


def test_subset_keeps_every_field_aligned():
    d = small()
    sub = d.for_cancer(d.cancer_types[0])
    assert len(sub.clinical) == len(sub.time) == len(sub.expression)
    assert set(sub.cancer_type) == {d.cancer_types[0]}


def test_filter_usable_drops_underpowered_cancer_types():
    d = make_synthetic(n_per_cancer=40, n_cancers=3, n_genes=60, seed=1)
    kept = filter_usable(d, min_events=1000, min_patients=1000)
    assert len(kept.time) == 0


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_model_fits_and_predicts_finite_risk(name):
    d = small()
    sub = d.for_cancer(d.cancer_types[0])
    n = len(sub.time)
    tr, te = np.arange(n // 2), np.arange(n // 2, n)
    model = build(name, seed=0).fit(
        sub.clinical.iloc[tr], sub.expression[tr], sub.y[tr]
    )
    risk = model.predict_risk(sub.clinical.iloc[te], sub.expression[te])
    assert risk.shape == (len(te),)
    assert np.isfinite(risk).all()


def test_build_rejects_unknown_model():
    with pytest.raises(KeyError):
        build("no_such_model")


def test_folds_never_share_patients():
    """The split must be disjoint, which concordance alone would not reveal."""
    d = small()
    res = run_benchmark(d, [BASELINE], n_splits=5, seeds=(0,))
    for (cancer, seed), group in res.groupby(["cancer_type", "seed"]):
        assert group.n_test.sum() == group.n_train.iloc[0] + group.n_test.iloc[0]


def test_concordance_lies_in_the_unit_interval():
    d = small()
    res = run_benchmark(d, [BASELINE, CONTROL], seeds=(0,))
    assert res.harrell_c.between(0, 1).all()
    finite_uno = res.uno_c.dropna()
    assert finite_uno.between(0, 1).all()


def test_expression_beats_clinical_when_the_signal_is_in_the_genes():
    """The sanity check the benchmark rests on.

    The generator puts real prognostic signal in a handful of genes, so a model
    that uses expression must beat one that does not. If this fails, the
    pipeline is measuring something other than what it claims.
    """
    d = make_synthetic(n_per_cancer=250, n_cancers=3, n_genes=200,
                       expression_strength=1.2, seed=3)
    res = run_benchmark(d, [BASELINE, "clinical_plus_expression"], seeds=(0,))
    means = summarise(res).set_index("model")["harrell_c_mean"]
    assert means["clinical_plus_expression"] > means[BASELINE]


def test_margin_is_matched_on_cancer_fold_and_seed():
    """An unmatched comparison would silently average across different folds."""
    d = small()
    res = run_benchmark(d, [BASELINE, CONTROL], seeds=(0,))
    joined = margin_over(res, BASELINE)
    assert (joined.model != BASELINE).all()
    assert len(joined) == len(res[res.model != BASELINE])
    assert joined.margin.notna().all()


def test_win_rate_counts_cancer_types_not_folds():
    d = small()
    res = run_benchmark(d, [BASELINE, CONTROL], seeds=(0,))
    wins = win_rate(res, BASELINE)
    assert (wins.cancer_types_total == len(d.cancer_types)).all()
    assert wins.win_rate.between(0, 1).all()


def test_figures_are_written(tmp_path):
    from pcsurv.plots import make_all

    d = small()
    res = run_benchmark(d, [BASELINE, CONTROL, "expression_coxnet"], seeds=(0,))
    paths = make_all(res, tmp_path, "test")
    assert len(paths) == 4
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)
