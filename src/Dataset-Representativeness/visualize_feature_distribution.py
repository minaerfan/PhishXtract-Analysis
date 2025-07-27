import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib as mpl
mpl.rcParams.update({
    "figure.titlesize": 22,
    "axes.titlesize"  : 20,
    "axes.labelsize"  : 18,
    "xtick.labelsize" : 18,
    "ytick.labelsize" : 18,
    "legend.fontsize" : 18,
})

# --- Load Data ---
print("Loading data...")
all_samples = pd.read_csv("monthly_feature_comparison.csv")
labeled_raw = pd.read_csv("random_forest_predictions_April.csv")

# --- Process Labeled Data ---
print("Processing labeled data...")
bool_cols = ['new_domain', 'control_over_dns', 'domain_indexed',
             'known_hosting', 'is_archived', 'is_on_root', 'is_subdomain']
num_cols = ['between_archives_distance', 'phish_archives_distance']

labeled_stats = {
    'month': '2024-04-labeled',
    'sample_type': 'labeled',
    'dataset': 'Labeled (Apr 20-25)'
}

for col in bool_cols:
    labeled_stats[f"{col}_pct_true"] = labeled_raw[col].mean() * 100

for col in num_cols:
    labeled_stats[f"{col}_mean"] = labeled_raw[col].mean()
    labeled_stats[f"{col}_std"] = labeled_raw[col].std()

labeled_df = pd.DataFrame([labeled_stats])

# --- Prepare Comparison Groups ---
print("Preparing comparison groups...")
consecutive_samples = all_samples[all_samples['sample_type'].str.contains('consecutive')].copy()
nonconsecutive_samples = all_samples[all_samples['sample_type'].str.contains('nonconsecutive')].copy()

consecutive_samples.loc[:, 'dataset'] = "Consecutive"
nonconsecutive_samples.loc[:, 'dataset'] = "Non-Consecutive"
all_samples.loc[:, 'dataset'] = "All Periods"
labeled_df.loc[:, 'dataset'] = "Labeled"


# ---  Plotting ---
def create_uniform_scaled_plots(comparison_df, title_suffix, file_suffix):
    print(f"Creating {title_suffix} plot with uniform scales...")

    bool_features = [f"{col}_pct_true" for col in bool_cols]
    num_features = [f"{col}_mean" for col in num_cols]
    all_features = bool_features + num_features

    # Create figure with dynamic layout
    n_cols = 3
    n_rows = int(np.ceil(len(all_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.suptitle(f"Feature Comparison: {title_suffix}",
                 y=1.02, fontsize=16, weight='bold')

    for i, feature in enumerate(all_features):
        ax = axes.flatten()[i]

        # Boxplot and stripplot
        sns.boxplot(data=comparison_df, x='dataset', y=feature,
                    color='lightblue', width=0.4, ax=ax)
        sns.stripplot(data=labeled_df, x='dataset', y=feature,
                      color='red', size=14, jitter=False, ax=ax)

        # Set uniform scale for boolean features
        if '_pct_true' in feature:
            ax.set_ylim(0, 100)

        # Formatting
        title = feature.replace('_pct_true', '').replace('_mean', '')
        ax.set_title(title, pad=10, weight='bold')
        ax.set_ylabel('Percentage (%)' if '_pct_true' in feature else 'Mean Distance',
                      labelpad=10)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=0)

        # Add grid for readability
        ax.grid(True, axis='y', linestyle=':', alpha=0.2)

    # Hide empty subplots
    for j in range(i + 1, n_rows * n_cols):
        axes.flatten()[j].axis('off')

    plt.tight_layout()
    plt.savefig(f'uniform_scale_{file_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: uniform_scale_{file_suffix}.png")


# Generate all three plot types
for df, suffix in [(consecutive_samples, 'consecutive'),
                   (nonconsecutive_samples, 'non_consecutive'),
                   (all_samples, 'all')]:
    create_uniform_scaled_plots(
        pd.concat([df, labeled_df]),
        title_suffix=f"{suffix.replace('_', ' ').title()} 5-Day Periods",
        file_suffix=suffix
    )


# --- Statistical Testing ---
def run_statistical_tests(reference_df, comparison_name):
    print(f"\n=== Statistical Comparison vs. {comparison_name} ===")
    print("{:<25} {:<10} {:<10}".format("Feature", "p-value", "Significance"))
    print("-" * 50)

    bool_features = [f"{col}_pct_true" for col in bool_cols]
    num_features = [f"{col}_mean" for col in num_cols]

    for feature in bool_features + num_features:
        try:
            stat, p = mannwhitneyu(
                [labeled_df[feature].iloc[0]],
                reference_df[feature].dropna(),
                alternative='two-sided'
            )
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print("{:<25} {:<10.4f} {:<10}".format(
                feature.replace('_pct_true', '').replace('_mean', ''),
                p,
                stars
            ))
        except Exception as e:
            print("{:<25} {:<10} {}".format(
                feature.replace('_pct_true', '').replace('_mean', ''),
                "Error",
                str(e)[:30] + "..."
            ))

    print("=" * 50 + "\n")


# Run comparisons
run_statistical_tests(consecutive_samples, "Consecutive Periods")
run_statistical_tests(nonconsecutive_samples, "Non-Consecutive Periods")
run_statistical_tests(all_samples, "All Periods")


# --- Labeled Dataset Feature Statistics ---
print("\nLabeled Dataset (April 20-25) Feature Statistics:")
print("="*50)

# Boolean Features Summary
print("\nBoolean Features (% True):")
print("-"*30)
bool_stats = []
for col in bool_cols:
    percent_true = labeled_raw[col].mean() * 100
    bool_stats.append([col, f"{percent_true:.2f}%"])
print(pd.DataFrame(bool_stats, columns=["Feature", "% True"]).to_string(index=False))

# Numeric Features Summary
print("\n\nNumeric Features:")
print("-"*30)
num_stats = []
for col in num_cols:
    mean_val = labeled_raw[col].mean()
    std_val = labeled_raw[col].std()
    num_stats.append([col, f"{mean_val:.2f}", f"{std_val:.2f}"])
print(pd.DataFrame(num_stats, columns=["Feature", "Mean", "Std Dev"]).to_string(index=False))

print("\n" + "="*50 + "\n")