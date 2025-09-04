import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Load LLaMA 3.2-3B-Instruct tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Paths to HF datasets (update paths if needed)
data_splits = {
    "train": "/home/arism/datasets/train/dataset_1_with_descriptions",
    "val": "/home/arism/datasets/val/dataset_1_with_descriptions", 
    "test": "/home/arism/datasets/test/dataset_1_with_descriptions",
}

# Columns to ignore
ignore_cols = {"soma_joinid"}
ignore_suffix = "_ontology_term_id"
skip_cols = {"natural_desc", "input_ids"}

# Create output directory
os.makedirs("analysis_output", exist_ok=True)

# Create text summary
summary_lines = []

# Collect all data for aggregation
all_dfs = []

def analyze_text_quality(text_series, name):
    texts = text_series.fillna("").astype(str)
    word_counts = texts.str.split().str.len()
    char_counts = texts.str.len()
    sentence_counts = texts.str.count(r'[.!?]+')
    token_lengths = [len(tokenizer.encode(text)) for text in texts]
    token_efficiency = [char_counts.iloc[i] / max(1, token_lengths[i]) for i in range(len(char_counts))]
    return {
        'avg_words': word_counts.mean(),
        'avg_chars': char_counts.mean(), 
        'avg_sentences': sentence_counts.mean(),
        'avg_tokens': np.mean(token_lengths),
        'avg_chars_per_token': np.mean(token_efficiency),
        'empty_texts': (texts == "").sum(),
        'very_short_texts': (word_counts < 5).sum(),
    }

def check_data_quality(df):
    quality_report = {
        'total_rows': len(df),
        'missing_values_by_col': df.isnull().sum().to_dict(),
        'columns_with_missing': (df.isnull().sum() > 0).sum(),
    }
    try:
        hashable_cols = []
        for col in df.columns:
            try:
                first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if first_val is not None:
                    hash(first_val)
                    hashable_cols.append(col)
            except (TypeError, IndexError):
                continue
        if hashable_cols:
            duplicate_count = df[hashable_cols].duplicated().sum()
            quality_report['duplicate_rows'] = duplicate_count
            quality_report['duplicate_check_cols'] = hashable_cols
        else:
            quality_report['duplicate_rows'] = "Cannot check (no hashable columns)"
            quality_report['duplicate_check_cols'] = []
    except Exception as e:
        quality_report['duplicate_rows'] = f"Error checking duplicates: {str(e)}"
        quality_report['duplicate_check_cols'] = []

    text_cols = df.select_dtypes(include=['object']).columns
    encoding_issues = {}
    for col in text_cols:
        try:
            weird_chars = df[col].astype(str).str.contains(r'[^\x00-\x7F]', na=False).sum()
            encoding_issues[col] = weird_chars
        except Exception:
            encoding_issues[col] = "Error checking encoding"
    quality_report['potential_encoding_issues'] = encoding_issues
    return quality_report

def analyze_split(name, hf_ds):
    if hasattr(hf_ds, 'to_pandas'):
        num_rows = len(hf_ds)
        column_names = list(hf_ds.column_names)
        df = hf_ds.to_pandas()
    else:
        df = hf_ds
        num_rows = len(df)
        column_names = list(df.columns)

    cols_to_analyze = [c for c in column_names if c not in ignore_cols and not c.endswith(ignore_suffix)]

    split_dir = f"analysis_output/{name}"
    os.makedirs(split_dir, exist_ok=True)

    split_summary = [f"===== {name.upper()} ANALYSIS ====="]
    split_summary.append(f"Total rows: {len(df)}")
    split_summary.append(f"Total columns: {len(df.columns)}")
    split_summary.append("")

    quality_report = check_data_quality(df)
    split_summary.append("DATA QUALITY:")
    for key, value in quality_report.items():
        split_summary.append(f"  {key}: {value}")
    split_summary.append("")

    if "length" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(df["length"].dropna(), bins=50, alpha=0.7, color='skyblue')
        plt.title(f"Length Distribution - {name}")
        plt.xlabel("Length")
        plt.ylabel("Frequency")
        plt.axvline(df["length"].mean(), color='red', linestyle='--', label=f'Mean: {df["length"].mean():.1f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{split_dir}/length_distribution.png", dpi=300)
        plt.close()

    if "natural_desc" in df.columns:
        token_lengths = [len(tokenizer.encode(str(x))) for x in df["natural_desc"].fillna("")]
        df["tokenized_length"] = token_lengths

        plt.figure(figsize=(8, 6))
        plt.hist(token_lengths, bins=50, alpha=0.7, color='lightgreen')
        plt.title(f"Tokenized natural_desc Length - {name}")
        plt.xlabel("# tokens")
        plt.ylabel("Frequency")
        plt.axvline(np.mean(token_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(token_lengths):.1f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{split_dir}/tokenized_desc_lengths.png", dpi=300)
        plt.close()

        text_quality = analyze_text_quality(df["natural_desc"], name)
        split_summary.append("TEXT QUALITY (natural_desc):")
        for key, value in text_quality.items():
            split_summary.append(f"  {key}: {value}")
        split_summary.append("")

    metrics = {}
    for col in cols_to_analyze:
        col_metrics = {'dtype': str(df[col].dtype)}
        if col in skip_cols:
            metrics[col] = col_metrics
            continue
        if df[col].dtype == "object" or df[col].dtype == "string":
            value_counts = df[col].value_counts(dropna=False)
            col_metrics.update({
                "unique_count": value_counts.shape[0],
                "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_frequent_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                "null_count": df[col].isnull().sum(),
                "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
            })
            top_values_df = pd.DataFrame({
                'value': value_counts.index,
                'count': value_counts.values,
                'percentage': (value_counts.values / len(df)) * 100
            })
            top_values_df.to_csv(f"{split_dir}/{name}_{col}_all_values.csv", index=False)
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_metrics.update({
                "count": df[col].count(),
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "25%": df[col].quantile(0.25),
                "50%": df[col].quantile(0.50),
                "75%": df[col].quantile(0.75),
                "max": df[col].max(),
                "null_count": df[col].isnull().sum(),
                "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
                "zeros": (df[col] == 0).sum(),
                "negatives": (df[col] < 0).sum() if df[col].dtype in ['int64', 'float64'] else 0,
            })
        metrics[col] = col_metrics

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.to_csv(f"{split_dir}/{name}_metrics.csv")

    split_summary.append("COLUMN ANALYSIS:")
    for col, m in metrics.items():
        split_summary.append(f"\n{col}:")
        for k, v in m.items():
            if isinstance(v, float):
                split_summary.append(f"  {k}: {v:.4f}")
            else:
                split_summary.append(f"  {k}: {v}")

    summary_lines.extend(split_summary)
    summary_lines.append("\n" + "="*80 + "\n")
    return df

split_stats = {}
for split, path in data_splits.items():
    print(f"Processing {split}...")
    try:
        hf_ds = load_from_disk(path)
        df = analyze_split(split, hf_ds)
        all_dfs.append(df)
        split_stats[split] = len(df)
    except Exception as e:
        print(f"Error processing {split}: {e}")
        continue

if len(all_dfs) > 1:
    comparison_summary = ["===== CROSS-SPLIT COMPARISON ====="]
    comparison_summary.append("Dataset sizes:")
    for split, size in split_stats.items():
        comparison_summary.append(f"  {split}: {size:,} rows")
    common_cols = set(all_dfs[0].columns)
    for df in all_dfs[1:]:
        common_cols &= set(df.columns)
    comparison_summary.append(f"\nCommon columns: {len(common_cols)}")
    comparison_summary.append(f"Common columns: {list(common_cols)}")
    summary_lines.extend(comparison_summary)

if all_dfs:
    all_df = pd.concat(all_dfs, ignore_index=True)
    print("Processing aggregated dataset...")
    analyze_split("combined", all_df)
    if len(split_stats) > 1:
        plt.figure(figsize=(10, 6))
        splits, sizes = zip(*split_stats.items())
        plt.bar(splits, sizes)
        plt.title('Dataset Split Sizes')
        plt.ylabel('Number of Samples')
        plt.xlabel('Split')
        for i, size in enumerate(sizes):
            plt.text(i, size + max(sizes)*0.01, f'{size:,}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('analysis_output/split_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

summary_header = [
    "COMPREHENSIVE DATASET ANALYSIS REPORT",
    "=" * 50,
    f"Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Tokenizer used: {tokenizer.name_or_path}",
    f"Total splits analyzed: {len(split_stats)}",
    f"Total samples: {sum(split_stats.values()):,}",
    "",
]

with open("analysis_output/comprehensive_summary.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(summary_header + summary_lines))

print("\nAnalysis complete! Results saved to:")
print("- analysis_output/<split>/<split>_metrics.csv (per-split metrics)")
print("- analysis_output/<split>/<split>_<col>_top_values.csv (top values)")
print("- analysis_output/comprehensive_summary.txt")
print("- analysis_output/<split>/*.png (visualizations per split)")
print("- analysis_output/split_comparison.png")
