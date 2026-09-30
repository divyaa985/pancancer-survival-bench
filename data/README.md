# data/

Datasets are not committed. This folder is where they go.

## TCGA-CDR-SupplementalTableS1.xlsx

The TCGA Pan-Cancer Clinical Data Resource from Liu et al. 2018, *Cell*
(https://doi.org/10.1016/j.cell.2018.02.052). Curated, quality-assessed
survival endpoints for all 33 TCGA cancer types, with per-cancer
recommendations on which endpoint to use. Available from the GDC PanCanAtlas
publication page and mirrored on UCSC Xena.

Use this rather than the raw TCGA clinical downloads. The raw files have
inconsistent endpoint definitions across cancer types, which is the problem
this resource was built to solve.

## tcga_pancan_expression.tsv

Pan-cancer RNA-seq, genes by samples, log2 normalised, from the UCSC Xena
browser (https://xenabrowser.net). Roughly 11,000 samples.

The loader transposes it, collapses sample barcodes to the 12-character
patient barcode, averages duplicate samples per patient, and keeps the 5000
most variable genes.

```bash
python -m pcsurv.run \
  --expression data/tcga_pancan_expression.tsv \
  --cdr data/TCGA-CDR-SupplementalTableS1.xlsx \
  --endpoint PFI --name tcga_pfi
```
