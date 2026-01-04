import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.config import RAW_DATA_PATH, FIG_DIR, TABLE_DIR, REPORT_DIR
from src.io import ensure_dirs, load_raw_data
from src.eda import compute_eda_summary, save_describe_table, plot_histograms, plot_corr_heatmap, plot_scatter
from src.reporting import write_markdown, format_eda_summary


def main():
    ensure_dirs(FIG_DIR, TABLE_DIR, REPORT_DIR)

    df = load_raw_data(RAW_DATA_PATH)
    summary = compute_eda_summary(df)

    # Tables
    save_describe_table(df, TABLE_DIR / "describe.csv")

    # Figures (edit cols list as you like)
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    plot_histograms(df, num_cols, FIG_DIR / "histograms.png")
    plot_corr_heatmap(df, FIG_DIR / "corr_heatmap.png")
    if "Avg_Temperature" in df.columns and "Net_Profit" in df.columns:
        plot_scatter(df, "Avg_Temperature", "Net_Profit", FIG_DIR / "scatter_temp_profit.png")

    # Markdown report
    md = format_eda_summary(summary)
    write_markdown(REPORT_DIR / "eda_summary.md", md)

if __name__ == "__main__":
    main()
