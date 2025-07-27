import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_and_combine_data(csv_files):
    """Load and concatenate all CSV files into a single DataFrame."""
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['report_date'] = df['report_date'].apply(lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")))
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


def get_whole_year_stats(df, bool_cols, num_cols):
    """Calculate feature statistics for the entire dataset (full year)."""
    stats = {}
    # Boolean features: % True
    for col in bool_cols:
        stats[f"{col}_pct_true"] = df[col].astype(int).mean() * 100
    # Numerical features
    for col in num_cols:
        non_neg = df.loc[df[col] >= 0, col]
        stats[f"{col}_mean"] = non_neg.mean()
        stats[f"{col}_std"] = non_neg.std()
    return pd.DataFrame(stats, index=["full_year"])


def sample_and_summarize_by_month_wide(df, bool_cols, num_cols, days_per_month=5, consecutive=False, random_seed=42):
    """Sample random days and return monthly summaries in wide format (all features in one row)."""
    np.random.seed(random_seed)
    monthly_stats = []

    df['year_month'] = df['report_date'].dt.to_period('M')
    for name, group in df.groupby('year_month'):
        if name.strftime('%Y-%m') == '2024-02':
            continue

        dates = group['report_date'].dt.date.unique()
        if len(dates) < days_per_month:
            continue

        if consecutive:
            start_idx = np.random.randint(0, len(dates) - days_per_month + 1)
            sampled_dates = dates[start_idx: start_idx + days_per_month]
        else:
            sampled_dates = np.random.choice(dates, size=days_per_month, replace=False)

        sampled_df = group[group['report_date'].dt.date.isin(sampled_dates)]

        # Calculate stats for this month's sample
        stats = {
            'month': name.strftime('%Y-%m'),
            'sample_type': 'consecutive_5days' if consecutive else 'nonconsecutive_5days'
        }
        # Boolean features
        for col in bool_cols:
            stats[f"{col}_pct_true"] = sampled_df[col].astype(int).mean() * 100
        # Numerical features
        for col in num_cols:
            non_neg = sampled_df.loc[sampled_df[col] >= 0, col]
            stats[f"{col}_mean"] = non_neg.mean()
            stats[f"{col}_std"] = non_neg.std()
        monthly_stats.append(stats)

    return pd.DataFrame(monthly_stats)


if __name__ == "__main__":
    # 1. Load data
    csv_files = glob.glob("../../results/*.csv")
    full_df = load_and_combine_data(csv_files)

    # 2. Define features
    bool_cols = ["new_domain", "control_over_dns", "domain_indexed",
                 "known_hosting", "is_archived", "is_on_root", "is_subdomain"]
    num_cols = ["between_archives_distance", "phish_archives_distance"]

    # 3. Get monthly samples (both consecutive and non-consecutive)
    consecutive_monthly = sample_and_summarize_by_month_wide(full_df, bool_cols, num_cols, consecutive=True)
    nonconsecutive_monthly = sample_and_summarize_by_month_wide(full_df, bool_cols, num_cols, consecutive=False)

    # 4. Get full-year stats
    full_year_stats = get_whole_year_stats(full_df, bool_cols, num_cols)
    full_year_stats['month'] = 'full_year'
    full_year_stats['sample_type'] = 'full_year'

    # 5. Combine all results
    all_results = pd.concat([
        consecutive_monthly,
        nonconsecutive_monthly,
        full_year_stats
    ]).reset_index(drop=True)

    print("Monthly Feature Distribution:")
    print(all_results)
    all_results.to_csv("monthly_feature_comparison.csv", index=False)
