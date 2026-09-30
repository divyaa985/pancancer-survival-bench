"""Loading TCGA pan-cancer survival data, and a synthetic generator for tests.

Two things in here are decisions rather than plumbing, and both are documented
where they are made:

1. Which survival endpoint to use. Overall survival is the obvious choice and
   often the wrong one. Liu et al. 2018 (TCGA-CDR) recommend the
   progression-free interval for 27 of the 33 TCGA cancer types and overall
   survival for 23, because follow-up in TCGA is short and OS in indolent
   cancers is dominated by deaths from other causes. The default here is PFI.

2. Which patients to keep. A cancer type with 12 events cannot support a
   model comparison, however tempting the extra row in the results table is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ENDPOINTS = {
    "OS": ("OS", "OS.time"),
    "PFI": ("PFI", "PFI.time"),
    "DSS": ("DSS", "DSS.time"),
    "DFI": ("DFI", "DFI.time"),
}

CLINICAL_COLUMNS = ["age_at_initial_pathologic_diagnosis", "gender", "ajcc_pathologic_tumor_stage"]


@dataclass
class SurvivalData:
    """Clinical covariates, expression, outcome, and cancer type per patient."""

    clinical: pd.DataFrame
    expression: np.ndarray
    time: np.ndarray
    event: np.ndarray
    cancer_type: np.ndarray
    gene_names: np.ndarray
    endpoint: str
    name: str

    def __post_init__(self) -> None:
        n = len(self.time)
        if not all(
            len(a) == n
            for a in (self.clinical, self.expression, self.event, self.cancer_type)
        ):
            raise ValueError("clinical, expression, time, event, cancer_type must align")

    @property
    def y(self) -> np.ndarray:
        """Structured array in the format scikit-survival expects."""
        return np.array(
            list(zip(self.event.astype(bool), self.time.astype(float))),
            dtype=[("event", "?"), ("time", "<f8")],
        )

    @property
    def cancer_types(self) -> list[str]:
        return sorted(pd.unique(self.cancer_type).tolist())

    def subset(self, mask: np.ndarray) -> SurvivalData:
        return SurvivalData(
            clinical=self.clinical.loc[mask].reset_index(drop=True),
            expression=self.expression[mask],
            time=self.time[mask],
            event=self.event[mask],
            cancer_type=self.cancer_type[mask],
            gene_names=self.gene_names,
            endpoint=self.endpoint,
            name=self.name,
        )

    def for_cancer(self, cancer: str) -> SurvivalData:
        return self.subset(self.cancer_type == cancer)

    def event_counts(self) -> pd.DataFrame:
        """Patients and events per cancer type, which decides what is usable."""
        df = pd.DataFrame({"cancer_type": self.cancer_type, "event": self.event})
        out = df.groupby("cancer_type")["event"].agg(["size", "sum"])
        out.columns = ["n_patients", "n_events"]
        out["event_rate"] = (out.n_events / out.n_patients).round(3)
        return out.sort_values("n_events", ascending=False).reset_index()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SurvivalData(name={self.name!r}, n={len(self.time)}, "
            f"genes={self.expression.shape[1]}, types={len(self.cancer_types)}, "
            f"endpoint={self.endpoint}, events={int(self.event.sum())})"
        )


def filter_usable(
    data: SurvivalData, min_events: int = 30, min_patients: int = 100
) -> SurvivalData:
    """Drop cancer types that cannot support a model comparison.

    The conventional rule of thumb is roughly ten events per covariate. With
    penalised models that is not a hard limit, but a cancer type with a handful
    of events produces a concordance index whose confidence interval spans most
    of the usable range, and including it makes the results table look more
    informative than it is.
    """
    counts = data.event_counts()
    keep = counts[
        (counts.n_events >= min_events) & (counts.n_patients >= min_patients)
    ].cancer_type
    mask = np.isin(data.cancer_type, keep.to_numpy())
    dropped = sorted(set(data.cancer_types) - set(keep))
    if dropped:
        print(f"  dropped {len(dropped)} cancer types with too few events: {', '.join(dropped)}")
    return data.subset(mask)


def load_tcga(
    expression_path: str | Path,
    cdr_path: str | Path,
    endpoint: str = "PFI",
    n_genes: int = 5000,
    name: str = "tcga",
) -> SurvivalData:
    """Load TCGA pan-cancer expression and the curated clinical data resource.

    Parameters
    ----------
    expression_path
        Genes x samples matrix from UCSC Xena (log2 normalised RSEM).
    cdr_path
        TCGA-CDR supplemental table from Liu et al. 2018, as .xlsx or .tsv.
    n_genes
        Keep this many genes by variance across all samples. Note that this is
        an unsupervised filter computed once on the full cohort: it never sees
        the outcome, so it does not leak, but it is recorded here so a reader
        can decide whether they agree.
    """
    expression_path, cdr_path = Path(expression_path), Path(cdr_path)
    for path in (expression_path, cdr_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. See README 'Getting the data'.")

    if endpoint not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of {sorted(ENDPOINTS)}")
    event_col, time_col = ENDPOINTS[endpoint]

    cdr = (
        pd.read_excel(cdr_path)
        if cdr_path.suffix in {".xlsx", ".xls"}
        else pd.read_csv(cdr_path, sep="\t")
    )
    cdr = cdr.rename(columns={"bcr_patient_barcode": "patient", "type": "cancer_type"})
    needed = ["patient", "cancer_type", event_col, time_col]
    cdr = cdr[needed + [c for c in CLINICAL_COLUMNS if c in cdr.columns]].dropna(
        subset=[event_col, time_col]
    )
    cdr = cdr[pd.to_numeric(cdr[time_col], errors="coerce") > 0]

    expr = pd.read_csv(expression_path, sep="\t", index_col=0)
    # Xena ships genes x samples; patients are the first 12 characters of the
    # TCGA barcode, and duplicated samples per patient are averaged.
    expr = expr.T
    expr.index = [str(i)[:12] for i in expr.index]
    expr = expr.groupby(level=0).mean()

    shared = sorted(set(expr.index) & set(cdr.patient))
    if not shared:
        raise ValueError("No overlapping patient barcodes between expression and CDR.")
    cdr = cdr.set_index("patient").loc[shared]
    expr = expr.loc[shared]

    expr = expr.loc[:, expr.notna().all(axis=0)]
    top = expr.var(axis=0).nlargest(min(n_genes, expr.shape[1])).index
    expr = expr[top]

    clinical = _tidy_clinical(cdr)

    return SurvivalData(
        clinical=clinical.reset_index(drop=True),
        expression=expr.to_numpy(dtype=np.float32),
        time=pd.to_numeric(cdr[time_col]).to_numpy(dtype=float),
        event=pd.to_numeric(cdr[event_col]).to_numpy(dtype=int),
        cancer_type=cdr["cancer_type"].astype(str).to_numpy(),
        gene_names=np.asarray(top),
        endpoint=endpoint,
        name=name,
    )


def _tidy_clinical(cdr: pd.DataFrame) -> pd.DataFrame:
    """Age, sex and a collapsed stage, which is all TCGA reliably provides.

    Stage is collapsed to early (I/II) against late (III/IV) rather than kept
    as four levels. TCGA stage annotation is inconsistent across cancer types
    and several types have no stage at all, so a finer split mostly encodes
    which cancer type a patient has.
    """
    out = pd.DataFrame(index=cdr.index)
    age = pd.to_numeric(
        cdr.get("age_at_initial_pathologic_diagnosis", pd.Series(index=cdr.index)),
        errors="coerce",
    )
    out["age"] = age.fillna(age.median())

    gender = cdr.get("gender", pd.Series(index=cdr.index)).astype(str).str.upper()
    out["is_female"] = (gender == "FEMALE").astype(float)

    stage = cdr.get("ajcc_pathologic_tumor_stage", pd.Series(index=cdr.index)).astype(str)
    late = stage.str.contains("III|IV", regex=True, na=False)
    known = stage.str.contains("Stage", na=False)
    out["late_stage"] = np.where(known, late.astype(float), np.nan)
    out["late_stage"] = out["late_stage"].fillna(out["late_stage"].median()
                                                 if out["late_stage"].notna().any() else 0.0)
    out["stage_unknown"] = (~known).astype(float)
    return out


def make_synthetic(
    n_per_cancer: int = 250,
    n_cancers: int = 4,
    n_genes: int = 300,
    n_informative: int = 10,
    clinical_strength: float = 0.8,
    expression_strength: float = 0.6,
    censoring_rate: float = 0.55,
    seed: int = 0,
) -> SurvivalData:
    """Synthetic survival data where both clinical and expression carry signal.

    Risk is a linear combination of age, stage and a small number of genes, so
    a correct pipeline should find that expression adds something over clinical
    alone. Event times are exponential given that risk, then independently
    censored. Each cancer type gets its own baseline hazard, which is what
    makes pooling across types non-trivial.
    """
    rng = np.random.default_rng(seed)
    n = n_per_cancer * n_cancers

    cancer_type = np.repeat([f"CANCER_{i}" for i in range(n_cancers)], n_per_cancer)
    baseline = np.repeat(rng.uniform(0.5, 2.0, n_cancers), n_per_cancer)

    age = rng.normal(62, 11, n).clip(25, 90)
    is_female = rng.integers(0, 2, n).astype(float)
    late_stage = rng.integers(0, 2, n).astype(float)
    clinical = pd.DataFrame(
        {
            "age": age,
            "is_female": is_female,
            "late_stage": late_stage,
            "stage_unknown": np.zeros(n),
        }
    )

    expression = rng.normal(0, 1, (n, n_genes)).astype(np.float32)
    informative = rng.choice(n_genes, n_informative, replace=False)
    weights = rng.normal(0, 1, n_informative)

    clinical_risk = clinical_strength * (
        0.03 * (age - age.mean()) + 0.9 * late_stage
    )
    expression_risk = expression_strength * (expression[:, informative] @ weights)
    log_hazard = clinical_risk + expression_risk
    log_hazard = (log_hazard - log_hazard.mean()) / log_hazard.std()

    event_time = rng.exponential(1.0 / (baseline * np.exp(log_hazard)))
    censor_time = rng.exponential(
        np.quantile(event_time, 1 - censoring_rate) * 2.0, n
    )
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)

    return SurvivalData(
        clinical=clinical,
        expression=expression,
        time=time * 365.0,
        event=event,
        cancer_type=cancer_type,
        gene_names=np.array([f"GENE_{i}" for i in range(n_genes)]),
        endpoint="synthetic",
        name="synthetic",
    )
