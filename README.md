# Carbon Transition Risk and Asset Stranding in Agricultural Enterprises

Which agricultural enterprises are most exposed to carbon transition risk — and which would become stranded assets if carbon were priced? This project answers that with a modular Python pipeline that constructs emissions proxies from enterprise-level financial and climate data, prices them under three NGFS-inspired carbon-price scenarios, and flags enterprises whose projected carbon costs exceed projected revenues. Built as a team research project in the CentraleSupélec × ESSEC double-degree curriculum.

## Approach

**Data.** The [Agriculture Financial Risk Dataset](https://www.kaggle.com/datasets/programmer3/agriculture-financial-risk-dataset) (Kaggle): 4,981 enterprises × 17 variables — financials (revenue, expenses, loan amount, debt-to-equity), climate exposure (average temperature, rainfall, drought index, flood risk score), and structure (region, enterprise size, quarter).

**Pipeline** (`scripts/run_pipeline.py` runs every step end-to-end):

1. **Feature engineering** — financial ratios (profit margin, cost ratio, debt ratio) and a composite climate stress index: `0.4·drought + 0.4·flood + 0.2·z(temperature)`.
2. **Preprocessing** — one-hot encoding of categoricals, standardization of numerics.
3. **Emissions proxies** — emissions are not observed in the data, so four candidate proxies are built as weighted z-score composites of activity scale (log expenses or log revenue), input-cost intensity, climate stress, and leverage. Their stability is compared via mean absolute deviation and top-decile overlap before fixing a baseline.
4. **Models** — Ridge regressions (5-fold CV alpha tuning over a log grid, 80/20 holdout) predicting net profit and the baseline emissions proxy.
5. **Scenario pricing** — carbon cost indices under three carbon prices benchmarked on NGFS Phase 3 (REMIND) scenarios, in USD 2010: Delayed Transition ($10/t), Net Zero 2050 ($110/t), Divergent Net Zero ($300/t). See `images/ngfs_bar_plot.png` for the benchmark levels.
6. **Projection & stranding** — 5-year revenue paths (2% growth, NPV at 5% discount), per-scenario future profit and carbon-risk indices, and a stranded flag where projected carbon cost exceeds projected revenue under the severest scenario (Divergent Net Zero).

**Scope assumptions** (stated up front given data limits): the dataset is enterprise-level, not investor-level; emissions are proxied, not observed; carbon prices are scenario-based, not forecast. The goal is a climate-financial risk pipeline, not precise emissions accounting.

## Contents

| Path | What it is |
|---|---|
| `scripts/run_pipeline.py` | Single entry point — runs the full pipeline |
| `scripts/eda_report.py` | EDA entry point — figures, describe table, summary |
| `src/config.py` | All paths, scenario prices, and tunable defaults |
| `src/features.py` | Financial ratios + climate stress index |
| `src/preprocessing.py` | Encoding and scaling |
| `src/proxies.py` | Emissions proxy variants + stability metrics |
| `src/models.py` | Ridge training with CV alpha tuning |
| `src/carbon.py`, `src/outputs.py` | Scenario carbon costs, revenue projection, stranding analysis |
| `src/eda.py`, `src/io.py`, `src/reporting.py` | Plots, I/O, markdown/CSV writers |
| `data/AgriRiskFin_Dataset.csv` | Raw Kaggle dataset |
| `data/data_cleaned.csv` | Preprocessed snapshot |
| `data/final_results_with_stranding_analysis.csv` | Full per-enterprise output (66 columns) |
| `outputs/` | Figures, tables, and reports from the last run |

## Results

- **Proxy choice matters more than proxy weights.** Proxies sharing a scale base agree strongly (expenses-based v1/v3: 70% top-decile overlap; revenue-based v2/v4: 71%), while proxies across bases agree weakly (23–28%). The scale variable, not the weighting scheme, drives who lands in the top risk decile. Full table: `outputs/tables/proxy_stability.md`.
- **Model metrics confirm the dataset's deterministic structure.** Both Ridge models reach R² ≈ 1.0 (RMSE ≈ 1e-6): net profit is an accounting identity of revenue and expenses in this synthetic dataset, and the proxy is a linear composite of available features. The models serve as the pipeline's prediction stage, not as evidence of out-of-sample power.
- **Stranding under the severest scenario.** At $300/t (Divergent Net Zero), every enterprise in the sample flags as stranded. Because risk indices are z-score based (not unit-consistent dollars), the flag is best read as a scenario-severity ordering rather than a calibrated default probability — a limitation the index framing makes explicit.
- Per-enterprise scenario scores (carbon cost, adjusted profit, carbon risk, future indices) are exported to `data/final_results_with_stranding_analysis.csv`.

## Running

Python 3.11. From the repo root:

```bash
pip install pandas numpy scikit-learn matplotlib tabulate

python scripts/eda_report.py    # EDA: histograms, correlation heatmap, describe table
python scripts/run_pipeline.py  # full pipeline: proxies, models, scenarios, stranding
```

Outputs are written to `outputs/figures/`, `outputs/tables/`, `outputs/reports/`, and `data/`. All defaults (random seed, CV splits, horizon, growth/discount rates, scenario prices) live in `src/config.py`.

## Team

Oscar Caudreliez, with Daniil, Oulaya, Ulysse, and Yuhan — team research project at CentraleSupélec / ESSEC.
