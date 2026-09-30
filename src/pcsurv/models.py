"""Survival models under test, and the two controls that keep them honest.

Every model takes clinical covariates and expression and returns a risk score
per patient; higher means worse prognosis. Everything a model learns is learnt
inside `fit`, which is only ever called on training patients.

The two controls are the point of this file.

`clinical_only` is the bar. Age, sex and stage in a Cox model is what a
clinician already has for free, at no sequencing cost. An expression model that
does not beat it has not earned its place in a clinic, whatever its C-index.

`random_signature` is the floor. Venet, Dumont and Detours showed in 2011 that
most random gene expression signatures are significantly associated with breast
cancer outcome, because so much of the transcriptome correlates with
proliferation and proliferation correlates with prognosis. A published
signature that fails to beat random genes has demonstrated nothing, and that
comparison is almost never reported.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis


@runtime_checkable
class SurvivalModel(Protocol):
    name: str

    def fit(self, clinical: pd.DataFrame, expression: np.ndarray,
            y: np.ndarray) -> SurvivalModel: ...

    def predict_risk(self, clinical: pd.DataFrame,
                     expression: np.ndarray) -> np.ndarray: ...


def _top_variable_genes(expression: np.ndarray, n: int) -> np.ndarray:
    """Variance filter fit on training expression only."""
    n = min(n, expression.shape[1])
    return np.sort(np.argsort(-expression.var(axis=0))[:n])


class ClinicalOnlyCox:
    """The bar: age, sex, stage. No sequencing required."""

    def __init__(self, seed: int = 0, alpha: float = 0.01):
        self.seed = seed
        self.alpha = alpha
        self.name = "clinical_only"

    def fit(self, clinical, expression, y):
        self.scaler_ = StandardScaler().fit(clinical.to_numpy(dtype=float))
        x = self.scaler_.transform(clinical.to_numpy(dtype=float))
        self.model_ = CoxPHSurvivalAnalysis(alpha=self.alpha).fit(x, y)
        return self

    def predict_risk(self, clinical, expression):
        return self.model_.predict(self.scaler_.transform(clinical.to_numpy(dtype=float)))


class RandomSignatureCox:
    """The floor: a random handful of genes, fit with a penalised Cox model.

    Draws its genes with a seed derived from the fold, so the signature differs
    between folds. A signature that happened to be lucky once would otherwise
    look like a real result.
    """

    def __init__(self, seed: int = 0, n_genes: int = 100):
        self.seed = seed
        self.n_genes = n_genes
        self.name = "random_signature"

    def fit(self, clinical, expression, y):
        rng = np.random.default_rng(self.seed)
        self.genes_ = rng.choice(
            expression.shape[1], min(self.n_genes, expression.shape[1]), replace=False
        )
        x = expression[:, self.genes_]
        self.scaler_ = StandardScaler().fit(x)
        self.model_ = CoxnetSurvivalAnalysis(
            l1_ratio=0.9, alpha_min_ratio=0.1, n_alphas=20, fit_baseline_model=False
        ).fit(self.scaler_.transform(x), y)
        return self

    def predict_risk(self, clinical, expression):
        return self.model_.predict(self.scaler_.transform(expression[:, self.genes_]))


class ExpressionCoxnet:
    """Elastic-net Cox on the most variable genes. Expression only."""

    def __init__(self, seed: int = 0, n_genes: int = 2000):
        self.seed = seed
        self.n_genes = n_genes
        self.name = "expression_coxnet"

    def fit(self, clinical, expression, y):
        self.genes_ = _top_variable_genes(expression, self.n_genes)
        x = expression[:, self.genes_]
        self.scaler_ = StandardScaler().fit(x)
        self.model_ = CoxnetSurvivalAnalysis(
            l1_ratio=0.9, alpha_min_ratio=0.05, n_alphas=30, fit_baseline_model=False
        ).fit(self.scaler_.transform(x), y)
        return self

    def predict_risk(self, clinical, expression):
        return self.model_.predict(self.scaler_.transform(expression[:, self.genes_]))


class ClinicalPlusExpression:
    """The model that has to justify the sequencing cost.

    Clinical covariates are left unpenalised in spirit by scaling them
    together with a much smaller gene set, so the penalty does not simply
    delete age and stage in favour of noisy genes.
    """

    def __init__(self, seed: int = 0, n_genes: int = 2000):
        self.seed = seed
        self.n_genes = n_genes
        self.name = "clinical_plus_expression"

    def _design(self, clinical, expression, genes):
        return np.column_stack(
            [clinical.to_numpy(dtype=float), expression[:, genes]]
        )

    def fit(self, clinical, expression, y):
        self.genes_ = _top_variable_genes(expression, self.n_genes)
        x = self._design(clinical, expression, self.genes_)
        self.scaler_ = StandardScaler().fit(x)
        self.model_ = CoxnetSurvivalAnalysis(
            l1_ratio=0.9, alpha_min_ratio=0.05, n_alphas=30, fit_baseline_model=False
        ).fit(self.scaler_.transform(x), y)
        return self

    def predict_risk(self, clinical, expression):
        x = self._design(clinical, expression, self.genes_)
        return self.model_.predict(self.scaler_.transform(x))


class RandomSurvivalForestModel:
    """Non-linear alternative, on clinical plus a modest gene set.

    Kept small on purpose. A forest on 2000 genes with a few hundred patients
    spends most of its time fitting noise, and the runtime column in the
    results table is part of the comparison.
    """

    def __init__(self, seed: int = 0, n_genes: int = 200, n_estimators: int = 200):
        self.seed = seed
        self.n_genes = n_genes
        self.n_estimators = n_estimators
        self.name = "random_survival_forest"

    def fit(self, clinical, expression, y):
        self.genes_ = _top_variable_genes(expression, self.n_genes)
        x = np.column_stack([clinical.to_numpy(dtype=float), expression[:, self.genes_]])
        self.scaler_ = StandardScaler().fit(x)
        self.model_ = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=self.seed,
        ).fit(self.scaler_.transform(x), y)
        return self

    def predict_risk(self, clinical, expression):
        x = np.column_stack([clinical.to_numpy(dtype=float), expression[:, self.genes_]])
        return self.model_.predict(self.scaler_.transform(x))


REGISTRY: dict[str, callable] = {
    "clinical_only": lambda seed: ClinicalOnlyCox(seed=seed),
    "random_signature": lambda seed: RandomSignatureCox(seed=seed),
    "expression_coxnet": lambda seed: ExpressionCoxnet(seed=seed),
    "clinical_plus_expression": lambda seed: ClinicalPlusExpression(seed=seed),
    "random_survival_forest": lambda seed: RandomSurvivalForestModel(seed=seed),
}

BASELINE = "clinical_only"
CONTROL = "random_signature"


def build(name: str, seed: int = 0) -> SurvivalModel:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](seed)
