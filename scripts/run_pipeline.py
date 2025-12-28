import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.config import (
    RAW_DATA_PATH, CLEAN_DATA_PATH,
    FIG_DIR, TABLE_DIR, REPORT_DIR,
    CARBON_PRICE_SCENARIOS_USD2010, PipelineConfig
)
from src.io import ensure_dirs, load_raw_data, save_clean_data
from src.features import add_financial_ratios, add_climate_stress
from src.preprocessing import build_clean_dataset
from src.proxies import add_proxy_variants, compare_proxies, choose_baseline_proxy
from src.models import train_ridge_regression, predict_column
from src.outputs import (
    add_environmental_risk_index,
    project_revenue_paths, discounted_revenue_npv,
    add_future_profit_index, add_future_carbon_risk_index,
    add_stranded_flag, reconstruct_categories, add_climate_profile
)
from src.reporting import save_table_csv, save_table_md, write_markdown

def main():
    cfg = PipelineConfig()
    ensure_dirs(FIG_DIR, TABLE_DIR, REPORT_DIR)

    # 1) Load
    df_raw = load_raw_data(RAW_DATA_PATH)

    # 2) Features (raw-space)
    df_feat = add_financial_ratios(df_raw)
    df_feat = add_climate_stress(df_feat)

    # 3) Clean dataset (dummies + scaling)
    df_cleaned = build_clean_dataset(df_feat)

    # Save cleaned snapshot
    save_clean_data(df_cleaned, CLEAN_DATA_PATH)

    # 4) Emissions proxy variants + stability table
    df_cleaned = add_proxy_variants(df_cleaned)
    stability = compare_proxies(df_cleaned, q=cfg.top_risk_q)
    save_table_csv(stability, TABLE_DIR / "proxy_stability.csv")
    save_table_md(stability, TABLE_DIR / "proxy_stability.md")

    baseline_proxy = choose_baseline_proxy(df_cleaned, preferred="Emissions_Proxy_v1")

    # 5) Models (D1-A, D2-A)
    # D1-A: predict Net_Profit (exclude leakage + labels + derived outputs)
    exclude_profit = [
        "Net_Profit",
        "Financial_Risk_Level",
        "Profit_Margin",  # leakage for Net_Profit
    ]

    # a text variable that's better off excluded
    exclude_other = [
        "Enterprise_Size",
        "Climate_Profile",
        "Financial_Risk_Level"
    ]

    # excluded everything that can hurt the calculations
    exclude_total_v1 = exclude_profit + exclude_other

    profit_res = train_ridge_regression(
        df_cleaned, target="Net_Profit",
        feature_exclude=exclude_total_v1,
        random_state=cfg.random_state,
        cv_splits=cfg.cv_splits
    )
    df_cleaned = predict_column(df_cleaned, profit_res.model, profit_res.features, "Pred_Net_Profit")

    # D2-A: predict baseline emissions proxy
    exclude_em = [baseline_proxy]  # exclude target itself

    exclude_total_v2 = exclude_em + exclude_other

    em_res = train_ridge_regression(
        df_cleaned, target=baseline_proxy,
        feature_exclude=exclude_total_v2,
        random_state=cfg.random_state,
        cv_splits=cfg.cv_splits
    )
    df_cleaned = predict_column(df_cleaned, em_res.model, em_res.features, "Pred_Emissions_Proxy")
    df_cleaned["Future_Emissions_Proxy"] = df_cleaned["Pred_Emissions_Proxy"]

    # 6) Environmental risk index (scenario multipliers)
    df_cleaned = add_environmental_risk_index(df_cleaned, CARBON_PRICE_SCENARIOS_USD2010, "Future_Emissions_Proxy")

    # 7) Future revenue (simple cross-sectional approach: model Revenue directly)
    # For full alignment with your notebook: you can train a revenue Ridge here too.
    # For skeleton: reuse Pred_Net_Profit as a placeholder or keep your own revenue model file.
    df_cleaned["Pred_Future_Revenue"] = df_cleaned["Revenue"]  # placeholder; replace with trained revenue model

    df_cleaned = project_revenue_paths(df_cleaned, "Pred_Future_Revenue", years=cfg.years, growth_rate=cfg.growth_rate)
    df_cleaned["Discounted_Future_Revenues"] = discounted_revenue_npv(df_cleaned, years=cfg.years, discount_rate=cfg.discount_rate)

    # 8) Future profit & risk indices (index-based; consistent with proxy)
    df_cleaned = add_future_profit_index(df_cleaned, CARBON_PRICE_SCENARIOS_USD2010, "Pred_Future_Revenue", "Future_Emissions_Proxy")
    df_cleaned = add_future_carbon_risk_index(df_cleaned, CARBON_PRICE_SCENARIOS_USD2010, "Pred_Future_Revenue")

    # 9) Stranding analysis
    severe = "Divergent Net Zero"
    df_cleaned = add_stranded_flag(df_cleaned, scenario=severe)
    df_cleaned = reconstruct_categories(df_cleaned)
    df_cleaned = add_climate_profile(df_cleaned)

    # Export key outputs
    save_clean_data(df_cleaned, TABLE_DIR / "final_dataset_snapshot.csv", index=False)

    # Quick summary markdown
    md = (
        "# Pipeline Run Summary\n\n"
        f"- random_state: {cfg.random_state}\n"
        f"- baseline proxy: {baseline_proxy}\n\n"
        "## Model metrics\n"
        f"- Net_Profit Ridge: alpha={profit_res.best_alpha:.4g}, RMSE={profit_res.test_rmse:.4g}, R2={profit_res.test_r2:.4g}\n"
        f"- EmissionsProxy Ridge: alpha={em_res.best_alpha:.4g}, RMSE={em_res.test_rmse:.4g}, R2={em_res.test_r2:.4g}\n"
    )
    write_markdown(REPORT_DIR / "run_summary.md", md)

if __name__ == "__main__":
    main()
