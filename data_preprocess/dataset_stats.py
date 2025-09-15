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

# Paths to HF datasets (updated to handle multiple datasets per split)
base_path = "/home/arism/datasets"
data_splits = {
    "train": os.path.join(base_path, "zip_dataset/train"),
    "val": os.path.join(base_path, "zip_dataset/val"),
    "test": os.path.join(base_path, "zip_dataset/test"),
}

# List of dataset prefixes to load (dataset_1 to dataset_8)
dataset_prefixes = [f"dataset_{i}_with_descriptions" for i in range(1, 9)]

# Columns to ignore
ignore_cols = {"soma_joinid"}
ignore_suffix = "_ontology_term_id"
skip_cols = {"natural_desc", "input_ids"}

# Diversity variables
variables = ['tissue_general', 'dev_stage_group', 'cell_type', 'sex', 'is_disease', 'disease']

# Create output directory
os.makedirs("analysis_output", exist_ok=True)

# Create text summary
summary_lines = []

def shannon_diversity(series):
    """Compute Shannon entropy (in bits) from value proportions."""
    # Remove null values and get clean series
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0.0
    
    proportions = clean_series.value_counts(normalize=True)
    if len(proportions) <= 1:
        return 0.0
    return - (proportions * np.log2(proportions)).sum()

def normalized_shannon_diversity(series):
    """Normalize Shannon index by max possible entropy (log2(n_categories))"""
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0.0
    
    num_categories = clean_series.nunique()
    if num_categories <= 1:
        return 0.0
    raw_shannon = shannon_diversity(series)
    max_shannon = np.log2(num_categories)
    return raw_shannon / max_shannon

def analyze_text_quality(text_series, name):
    """Analyze quality metrics for text data"""
    texts = text_series.fillna("").astype(str)
    word_counts = texts.str.split().str.len().fillna(0)
    char_counts = texts.str.len()
    sentence_counts = texts.str.count(r'[.!?]+').fillna(0)
    
    # Safe tokenization with error handling
    token_lengths = []
    for text in texts:
        try:
            token_lengths.append(len(tokenizer.encode(str(text))))
        except Exception:
            token_lengths.append(0)
    
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
    """Comprehensive data quality assessment"""
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values_by_col': df.isnull().sum().to_dict(),
        'columns_with_missing': (df.isnull().sum() > 0).sum(),
    }
    
    # Safe duplicate checking
    try:
        # Only check duplicates on a subset of columns to avoid memory issues
        check_cols = [col for col in df.columns[:10] if df[col].dtype in ['object', 'int64', 'float64']]
        if check_cols:
            duplicate_count = df[check_cols].duplicated().sum()
            quality_report['duplicate_rows'] = duplicate_count
            quality_report['duplicate_check_cols'] = check_cols
        else:
            quality_report['duplicate_rows'] = "No suitable columns for duplicate check"
            quality_report['duplicate_check_cols'] = []
    except Exception as e:
        quality_report['duplicate_rows'] = f"Error checking duplicates: {str(e)}"
        quality_report['duplicate_check_cols'] = []

    # Check for encoding issues in text columns
    text_cols = df.select_dtypes(include=['object']).columns
    encoding_issues = {}
    for col in text_cols[:5]:  # Limit to first 5 text columns to avoid performance issues
        try:
            weird_chars = df[col].astype(str).str.contains(r'[^\x00-\x7F]', na=False).sum()
            encoding_issues[col] = weird_chars
        except Exception:
            encoding_issues[col] = "Error checking encoding"
    quality_report['potential_encoding_issues'] = encoding_issues
    
    return quality_report

def load_and_combine_datasets(dataset_paths):
    """Safely load and combine multiple HF datasets"""
    dfs = []
    load_info = []
    
    print(f"Loading {len(dataset_paths)} datasets...")
    for i, ds_path in enumerate(dataset_paths):
        try:
            print(f"  Loading dataset {i+1}/{len(dataset_paths)}: {os.path.basename(ds_path)}")
            ds = load_from_disk(ds_path)
            df = ds.to_pandas()
            
            # Basic validation
            if len(df) == 0:
                print(f"    ⚠️ Dataset is empty, skipping")
                continue
                
            dfs.append(df)
            load_info.append({
                'path': ds_path,
                'name': os.path.basename(ds_path),
                'rows': len(df),
                'columns': len(df.columns)
            })
            print(f"    ✅ Loaded: {len(df):,} rows, {len(df.columns)} columns")
            
        except Exception as e:
            print(f"    ❌ Failed to load {ds_path}: {e}")
            continue

    if not dfs:
        raise ValueError("No datasets were successfully loaded")

    # Combine datasets with proper error handling
    print("  Combining datasets...")
    try:
        # Use sort=False to maintain column order and avoid unnecessary sorting
        combined_df = pd.concat(dfs, ignore_index=True, sort=False)
        print(f"  ✅ Successfully combined into {len(combined_df):,} total rows")
        return combined_df, load_info
        
    except Exception as e:
        print(f"  ❌ Error combining datasets: {e}")
        raise

def analyze_split(split_name, dataset_paths):
    """Analyze a data split by combining all its datasets"""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING SPLIT: {split_name.upper()}")
    print(f"{'='*60}")
    
    # Load and combine datasets
    combined_df, load_info = load_and_combine_datasets(dataset_paths)
    
    # Create split-specific output directory
    split_dir = f"analysis_output/{split_name}"
    os.makedirs(split_dir, exist_ok=True)

    # Initialize split summary
    split_summary = [f"===== {split_name.upper()} ANALYSIS ====="]
    split_summary.append(f"Datasets loaded: {len(load_info)}")
    for info in load_info:
        split_summary.append(f"  - {info['name']}: {info['rows']:,} rows, {info['columns']} cols")
    split_summary.append(f"Combined total rows: {len(combined_df):,}")
    split_summary.append(f"Combined total columns: {len(combined_df.columns)}")
    split_summary.append("")

    # --- DATA QUALITY REPORT ---
    print("Performing data quality analysis...")
    quality_report = check_data_quality(combined_df)
    split_summary.append("DATA QUALITY:")
    for key, value in quality_report.items():
        if isinstance(value, dict):
            split_summary.append(f"  {key}:")
            for k, v in value.items():
                split_summary.append(f"    {k}: {v}")
        else:
            split_summary.append(f"  {key}: {value}")
    split_summary.append("")

    # --- LENGTH DISTRIBUTION ---
    if "length" in combined_df.columns:
        print("Creating length distribution plot...")
        try:
            plt.figure(figsize=(10, 6))
            length_data = combined_df["length"].dropna()
            plt.hist(length_data, bins=min(50, len(length_data.unique())), alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f"Length Distribution - {split_name} (Combined)", fontsize=14)
            plt.xlabel("Length", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            mean_length = length_data.mean()
            plt.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{split_dir}/length_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            split_summary.append("LENGTH STATISTICS:")
            split_summary.append(f"  Mean: {mean_length:.2f}")
            split_summary.append(f"  Std: {length_data.std():.2f}")
            split_summary.append(f"  Min: {length_data.min()}")
            split_summary.append(f"  Max: {length_data.max()}")
            split_summary.append("")
            
        except Exception as e:
            print(f"  ❌ Error creating length distribution plot: {e}")

    # --- TOKENIZED NATURAL_DESC ANALYSIS ---
    if "natural_desc" in combined_df.columns:
        print("Analyzing tokenized natural descriptions...")
        try:
            natural_desc_clean = combined_df["natural_desc"].fillna("").astype(str)
            
            print("  Tokenizing descriptions...")
            token_lengths = []
            for i, text in enumerate(natural_desc_clean):
                if i % 10000 == 0 and i > 0:
                    print(f"    Processed {i:,}/{len(natural_desc_clean):,} descriptions")
                try:
                    token_lengths.append(len(tokenizer.encode(str(text))))
                except Exception:
                    token_lengths.append(0)
            
            combined_df["tokenized_length"] = token_lengths

            plt.figure(figsize=(10, 6))
            plt.hist(token_lengths, bins=min(50, len(set(token_lengths))), alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title(f"Tokenized natural_desc Length - {split_name} (Combined)", fontsize=14)
            plt.xlabel("Number of tokens", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            mean_tokens = np.mean(token_lengths)
            plt.axvline(mean_tokens, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_tokens:.1f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{split_dir}/tokenized_desc_lengths.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Text quality analysis
            text_quality = analyze_text_quality(combined_df["natural_desc"], split_name)
            split_summary.append("TEXT QUALITY (natural_desc):")
            for key, value in text_quality.items():
                if isinstance(value, float):
                    split_summary.append(f"  {key}: {value:.2f}")
                else:
                    split_summary.append(f"  {key}: {value}")
            split_summary.append("")
            
        except Exception as e:
            print(f"  ❌ Error in natural_desc analysis: {e}")

    # --- DIVERSITY METRICS ---
    print("Computing diversity metrics...")
    diversity_scores = {}
    diversity_details = {}
    
    split_summary.append("DIVERSITY METRICS (Normalized Shannon Index):")
    
    for col in variables:
        if col not in combined_df.columns:
            split_summary.append(f"  {col}: N/A (column missing)")
            diversity_scores[col] = None
            continue
            
        try:
            # Get clean data for this column
            clean_data = combined_df[col].dropna()
            n_unique = clean_data.nunique()
            n_total = len(clean_data)
            n_missing = combined_df[col].isnull().sum()
            
            if n_unique == 0:
                score = 0.0
            else:
                score = normalized_shannon_diversity(combined_df[col])
            
            diversity_scores[col] = score
            diversity_details[col] = {
                'n_unique': n_unique,
                'n_total': n_total,
                'n_missing': n_missing,
                'score': score
            }
            
            split_summary.append(f"  {col}: {score:.4f} (unique: {n_unique}, total: {n_total}, missing: {n_missing})")
            
        except Exception as e:
            print(f"  ❌ Error computing diversity for {col}: {e}")
            diversity_scores[col] = None

    # Overall average diversity
    valid_scores = [s for s in diversity_scores.values() if s is not None and isinstance(s, (int, float))]
    average_diversity = np.mean(valid_scores) if valid_scores else 0.0
    split_summary.append(f"  📊 Average Normalized Diversity: {average_diversity:.4f}")
    split_summary.append("")

    # Save diversity details
    diversity_df = pd.DataFrame.from_dict(diversity_details, orient='index')
    if not diversity_df.empty:
        diversity_df.to_csv(f"{split_dir}/{split_name}_diversity_details.csv")

    # --- COLUMN ANALYSIS ---
    print("Performing detailed column analysis...")
    cols_to_analyze = [c for c in combined_df.columns 
                      if c not in ignore_cols and not c.endswith(ignore_suffix)]
    metrics = {}

    for col in cols_to_analyze:
        if col in skip_cols:
            continue
            
        try:
            col_metrics = {'dtype': str(combined_df[col].dtype)}
            
            if combined_df[col].dtype == "object" or pd.api.types.is_string_dtype(combined_df[col]):
                # Categorical/string analysis
                value_counts = combined_df[col].value_counts(dropna=False)
                col_metrics.update({
                    "unique_count": len(value_counts),
                    "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                    "most_frequent_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    "null_count": combined_df[col].isnull().sum(),
                    "null_percentage": (combined_df[col].isnull().sum() / len(combined_df)) * 100,
                })
                
                # Save top values (limit to prevent huge files)
                top_values_df = pd.DataFrame({
                    'value': value_counts.index[:1000],  # Top 1000 values max
                    'count': value_counts.values[:1000],
                    'percentage': (value_counts.values[:1000] / len(combined_df)) * 100
                })
                top_values_df.to_csv(f"{split_dir}/{split_name}_{col}_top_values.csv", index=False)
                
            elif pd.api.types.is_numeric_dtype(combined_df[col]):
                # Numeric analysis
                numeric_data = combined_df[col].dropna()
                col_metrics.update({
                    "count": len(numeric_data),
                    "mean": numeric_data.mean() if len(numeric_data) > 0 else None,
                    "std": numeric_data.std() if len(numeric_data) > 0 else None,
                    "min": numeric_data.min() if len(numeric_data) > 0 else None,
                    "25%": numeric_data.quantile(0.25) if len(numeric_data) > 0 else None,
                    "50%": numeric_data.quantile(0.50) if len(numeric_data) > 0 else None,
                    "75%": numeric_data.quantile(0.75) if len(numeric_data) > 0 else None,
                    "max": numeric_data.max() if len(numeric_data) > 0 else None,
                    "null_count": combined_df[col].isnull().sum(),
                    "null_percentage": (combined_df[col].isnull().sum() / len(combined_df)) * 100,
                    "zeros": (numeric_data == 0).sum() if len(numeric_data) > 0 else 0,
                    "negatives": (numeric_data < 0).sum() if len(numeric_data) > 0 else 0,
                })
                
            metrics[col] = col_metrics
            
        except Exception as e:
            print(f"  ❌ Error analyzing column {col}: {e}")
            metrics[col] = {'dtype': str(combined_df[col].dtype), 'error': str(e)}

    # Save metrics
    if metrics:
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
        metrics_df.to_csv(f"{split_dir}/{split_name}_column_metrics.csv")

    # Add column analysis to summary (abbreviated)
    split_summary.append("COLUMN ANALYSIS (Top 10 columns):")
    for i, (col, m) in enumerate(list(metrics.items())[:10]):
        split_summary.append(f"\n{col} ({m.get('dtype', 'unknown')}):")
        for k, v in m.items():
            if k not in ['dtype'] and v is not None:
                if isinstance(v, float):
                    split_summary.append(f"  {k}: {v:.4f}")
                else:
                    split_summary.append(f"  {k}: {v}")
        if i >= 9:  # Limit to first 10 columns in summary
            remaining = len(metrics) - 10
            if remaining > 0:
                split_summary.append(f"\n... and {remaining} more columns (see CSV file)")
            break

    print(f"✅ Split '{split_name}' analysis complete!")
    return combined_df, diversity_scores, split_summary

def compare_splits(split_dataframes, all_diversity_scores):
    """Compare metrics across different data splits"""
    
    print(f"\n{'='*60}")
    print("CROSS-SPLIT COMPARISON")
    print(f"{'='*60}")
    
    comparison_summary = ["===== CROSS-SPLIT COMPARISON ====="]
    
    # Dataset sizes
    comparison_summary.append("Dataset sizes:")
    total_samples = 0
    for split, df in split_dataframes.items():
        size = len(df)
        total_samples += size
        comparison_summary.append(f"  {split}: {size:,} rows")
    comparison_summary.append(f"  TOTAL: {total_samples:,} rows")
    comparison_summary.append("")

    # Column consistency
    all_columns = {}
    for split, df in split_dataframes.items():
        all_columns[split] = set(df.columns)
    
    if len(all_columns) > 1:
        common_cols = set.intersection(*all_columns.values())
        comparison_summary.append(f"Common columns across all splits: {len(common_cols)}")
        
        # Check for split-specific columns
        for split, cols in all_columns.items():
            unique_cols = cols - common_cols
            if unique_cols:
                comparison_summary.append(f"Columns only in {split}: {sorted(list(unique_cols))}")
        comparison_summary.append("")

    # Diversity comparison
    if all_diversity_scores:
        comparison_summary.append("Diversity Comparison (Normalized Shannon Index):")
        comparison_summary.append(f"{'Split':<10} {'Avg Diversity':<15} {'Variables'}")
        comparison_summary.append("-" * 50)
        
        for split, scores in all_diversity_scores.items():
            valid_scores = [s for s in scores.values() if s is not None and isinstance(s, (int, float))]
            avg_diversity = np.mean(valid_scores) if valid_scores else 0.0
            n_vars = len([s for s in scores.values() if s is not None])
            comparison_summary.append(f"{split:<10} {avg_diversity:<15.4f} {n_vars}")
            
        comparison_summary.append("")
        
        # Variable-wise comparison
        comparison_summary.append("Variable-wise Diversity Comparison:")
        all_vars = set()
        for scores in all_diversity_scores.values():
            all_vars.update(scores.keys())
            
        for var in sorted(all_vars):
            comparison_summary.append(f"\n{var}:")
            for split, scores in all_diversity_scores.items():
                score = scores.get(var)
                if score is not None:
                    comparison_summary.append(f"  {split}: {score:.4f}")
                else:
                    comparison_summary.append(f"  {split}: N/A")

    return comparison_summary

def analyze_final_combined(all_dataframes):
    """Analyze the final combined dataset from all splits"""
    
    print(f"\n{'='*60}")
    print("FINAL COMBINED DATASET ANALYSIS")
    print(f"{'='*60}")
    
    print("Combining all splits...")
    try:
        final_combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        print(f"✅ Final combined dataset: {len(final_combined_df):,} rows, {len(final_combined_df.columns)} columns")
    except Exception as e:
        print(f"❌ Error creating final combined dataset: {e}")
        return []
    
    # Create output directory
    final_dir = "analysis_output/final_combined"
    os.makedirs(final_dir, exist_ok=True)

    final_summary = ["\n===== FINAL COMBINED DATASET (ALL SPLITS) ====="]
    final_summary.append(f"Total rows: {len(final_combined_df):,}")
    final_summary.append(f"Total columns: {len(final_combined_df.columns)}")
    final_summary.append("")

    # Quality check
    quality_report = check_data_quality(final_combined_df)
    final_summary.append("FINAL COMBINED DATA QUALITY:")
    final_summary.append(f"  Missing values (any): {(final_combined_df.isnull().sum() > 0).sum()} columns")
    final_summary.append(f"  Total missing values: {final_combined_df.isnull().sum().sum():,}")
    final_summary.append("")

    # Diversity analysis
    print("Computing final diversity metrics...")
    final_diversity = {}
    final_summary.append("FINAL DIVERSITY METRICS:")
    
    for col in variables:
        if col not in final_combined_df.columns:
            final_summary.append(f"  {col}: N/A (column missing)")
            continue
            
        try:
            clean_data = final_combined_df[col].dropna()
            n_unique = clean_data.nunique()
            n_total = len(clean_data)
            n_missing = final_combined_df[col].isnull().sum()
            
            if n_unique == 0:
                score = 0.0
            else:
                score = normalized_shannon_diversity(final_combined_df[col])
            
            final_diversity[col] = {
                'score': score,
                'n_unique': n_unique,
                'n_total': n_total,
                'n_missing': n_missing
            }
            
            final_summary.append(f"  {col}: {score:.4f} (unique: {n_unique:,}, total: {n_total:,}, missing: {n_missing:,})")
            
        except Exception as e:
            print(f"  ❌ Error computing final diversity for {col}: {e}")

    # Overall diversity
    valid_scores = [d['score'] for d in final_diversity.values() if isinstance(d.get('score'), (int, float))]
    final_avg_diversity = np.mean(valid_scores) if valid_scores else 0.0
    final_summary.append(f"  📊 Final Average Normalized Diversity: {final_avg_diversity:.4f}")

    # Save final diversity details
    final_diversity_df = pd.DataFrame.from_dict(final_diversity, orient='index')
    if not final_diversity_df.empty:
        final_diversity_df.to_csv(f"{final_dir}/final_diversity_metrics.csv")

    # ===== NEW CODE: TOKEN LENGTH DISTRIBUTION ANALYSIS =====
    # Length distribution for combined dataset
    if "length" in final_combined_df.columns:
        print("Creating length distribution plot for combined dataset...")
        try:
            plt.figure(figsize=(10, 6))
            length_data = final_combined_df["length"].dropna()
            plt.hist(length_data, bins=min(50, len(length_data.unique())), alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f"Length Distribution - Final Combined Dataset", fontsize=14)
            plt.xlabel("Length", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            mean_length = length_data.mean()
            plt.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{final_dir}/final_combined_length_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            final_summary.append("FINAL COMBINED LENGTH STATISTICS:")
            final_summary.append(f"  Mean: {mean_length:.2f}")
            final_summary.append(f"  Std: {length_data.std():.2f}")
            final_summary.append(f"  Min: {length_data.min()}")
            final_summary.append(f"  Max: {length_data.max()}")
            final_summary.append("")
            
        except Exception as e:
            print(f"  ❌ Error creating final combined length distribution plot: {e}")

    # Tokenized natural_desc analysis for combined dataset
    if "natural_desc" in final_combined_df.columns:
        print("Analyzing tokenized natural descriptions for combined dataset...")
        try:
            natural_desc_clean = final_combined_df["natural_desc"].fillna("").astype(str)
            
            print("  Tokenizing descriptions for combined dataset...")
            token_lengths = []
            total_descriptions = len(natural_desc_clean)
            
            for i, text in enumerate(natural_desc_clean):
                if i % 10000 == 0 and i > 0:
                    print(f"    Processed {i:,}/{total_descriptions:,} descriptions ({i/total_descriptions*100:.1f}%)")
                try:
                    token_lengths.append(len(tokenizer.encode(str(text))))
                except Exception:
                    token_lengths.append(0)
            
            final_combined_df["tokenized_length"] = token_lengths

            # Create tokenized length distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(token_lengths, bins=min(50, len(set(token_lengths))), alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title(f"Tokenized natural_desc Length - Final Combined Dataset", fontsize=14)
            plt.xlabel("Number of tokens", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            mean_tokens = np.mean(token_lengths)
            plt.axvline(mean_tokens, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_tokens:.1f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{final_dir}/final_combined_tokenized_desc_lengths.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Save detailed token length statistics to CSV
            token_stats_df = pd.DataFrame({
                'statistic': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'zeros', 'max_tokens_1000+'],
                'value': [
                    len(token_lengths),
                    np.mean(token_lengths),
                    np.std(token_lengths),
                    np.min(token_lengths),
                    np.percentile(token_lengths, 25),
                    np.percentile(token_lengths, 50),
                    np.percentile(token_lengths, 75),
                    np.max(token_lengths),
                    sum(1 for x in token_lengths if x == 0),
                    sum(1 for x in token_lengths if x >= 1000)
                ]
            })
            token_stats_df.to_csv(f"{final_dir}/final_combined_token_length_stats.csv", index=False)

            # Text quality analysis for combined dataset
            text_quality = analyze_text_quality(final_combined_df["natural_desc"], "final_combined")
            
            # Save text quality metrics to CSV
            text_quality_df = pd.DataFrame([text_quality]).T
            text_quality_df.columns = ['value']
            text_quality_df.index.name = 'metric'
            text_quality_df.to_csv(f"{final_dir}/final_combined_text_quality_metrics.csv")
            
            final_summary.append("FINAL COMBINED TEXT QUALITY (natural_desc):")
            for key, value in text_quality.items():
                if isinstance(value, float):
                    final_summary.append(f"  {key}: {value:.2f}")
                else:
                    final_summary.append(f"  {key}: {value}")
            
            final_summary.append(f"\nFINAL COMBINED TOKEN LENGTH STATISTICS:")
            final_summary.append(f"  Mean tokens: {mean_tokens:.2f}")
            final_summary.append(f"  Std tokens: {np.std(token_lengths):.2f}")
            final_summary.append(f"  Min tokens: {np.min(token_lengths)}")
            final_summary.append(f"  Max tokens: {np.max(token_lengths)}")
            final_summary.append(f"  Descriptions with 1000+ tokens: {sum(1 for x in token_lengths if x >= 1000):,}")
            final_summary.append("")
            
        except Exception as e:
            print(f"  ❌ Error in final combined natural_desc analysis: {e}")
    # ===== END TOKEN LENGTH ANALYSIS =====

    # ===== NEW CODE: ADD COLUMN ANALYSIS AND CSV GENERATION =====
    print("Performing detailed column analysis for combined dataset...")
    cols_to_analyze = [c for c in final_combined_df.columns 
                      if c not in ignore_cols and not c.endswith(ignore_suffix)]
    final_metrics = {}

    for col in cols_to_analyze:
        if col in skip_cols:
            continue
            
        try:
            col_metrics = {'dtype': str(final_combined_df[col].dtype)}
            
            if final_combined_df[col].dtype == "object" or pd.api.types.is_string_dtype(final_combined_df[col]):
                # Categorical/string analysis
                value_counts = final_combined_df[col].value_counts(dropna=False)
                col_metrics.update({
                    "unique_count": len(value_counts),
                    "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                    "most_frequent_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    "null_count": final_combined_df[col].isnull().sum(),
                    "null_percentage": (final_combined_df[col].isnull().sum() / len(final_combined_df)) * 100,
                })
                
                # Save top values for combined dataset (limit to prevent huge files)
                top_values_df = pd.DataFrame({
                    'value': value_counts.index[:1000],  # Top 1000 values max
                    'count': value_counts.values[:1000],
                    'percentage': (value_counts.values[:1000] / len(final_combined_df)) * 100
                })
                top_values_df.to_csv(f"{final_dir}/final_combined_{col}_top_values.csv", index=False)
                
            elif pd.api.types.is_numeric_dtype(final_combined_df[col]):
                # Numeric analysis
                numeric_data = final_combined_df[col].dropna()
                col_metrics.update({
                    "count": len(numeric_data),
                    "mean": numeric_data.mean() if len(numeric_data) > 0 else None,
                    "std": numeric_data.std() if len(numeric_data) > 0 else None,
                    "min": numeric_data.min() if len(numeric_data) > 0 else None,
                    "25%": numeric_data.quantile(0.25) if len(numeric_data) > 0 else None,
                    "50%": numeric_data.quantile(0.50) if len(numeric_data) > 0 else None,
                    "75%": numeric_data.quantile(0.75) if len(numeric_data) > 0 else None,
                    "max": numeric_data.max() if len(numeric_data) > 0 else None,
                    "null_count": final_combined_df[col].isnull().sum(),
                    "null_percentage": (final_combined_df[col].isnull().sum() / len(final_combined_df)) * 100,
                    "zeros": (numeric_data == 0).sum() if len(numeric_data) > 0 else 0,
                    "negatives": (numeric_data < 0).sum() if len(numeric_data) > 0 else 0,
                })
                
            final_metrics[col] = col_metrics
            
        except Exception as e:
            print(f"  ❌ Error analyzing column {col}: {e}")
            final_metrics[col] = {'dtype': str(final_combined_df[col].dtype), 'error': str(e)}

    # Save final combined column metrics
    if final_metrics:
        final_metrics_df = pd.DataFrame.from_dict(final_metrics, orient="index")
        final_metrics_df.to_csv(f"{final_dir}/final_combined_column_metrics.csv")
        print(f"✅ Saved final combined column metrics: {final_dir}/final_combined_column_metrics.csv")

    # Add abbreviated column analysis to summary
    final_summary.append("\nFINAL COMBINED COLUMN ANALYSIS (Top 10 columns):")
    for i, (col, m) in enumerate(list(final_metrics.items())[:10]):
        final_summary.append(f"\n{col} ({m.get('dtype', 'unknown')}):")
        for k, v in m.items():
            if k not in ['dtype'] and v is not None:
                if isinstance(v, float):
                    final_summary.append(f"  {k}: {v:.4f}")
                else:
                    final_summary.append(f"  {k}: {v}")
        if i >= 9:  # Limit to first 10 columns in summary
            remaining = len(final_metrics) - 10
            if remaining > 0:
                final_summary.append(f"\n... and {remaining} more columns (see CSV file)")
            break
    # ===== END NEW CODE =====
        
    # Save final combined basic stats
    final_stats = {
        'total_rows': len(final_combined_df),
        'total_columns': len(final_combined_df.columns),
        'total_missing_values': final_combined_df.isnull().sum().sum(),
        'avg_diversity': final_avg_diversity,
        'memory_usage_mb': final_combined_df.memory_usage(deep=True).sum() / (1024**2)
    }
    
    with open(f"{final_dir}/final_summary_stats.txt", "w") as f:
        for key, value in final_stats.items():
            f.write(f"{key}: {value}\n")

    print(f"✅ Final combined analysis complete!")
    return final_summary
# ===== MAIN EXECUTION =====

def main():
    print("🚀 Starting Comprehensive Dataset Analysis")
    print(f"Tokenizer: {tokenizer.name_or_path}")
    print(f"Dataset base path: {base_path}")
    print(f"Expected dataset prefixes: {dataset_prefixes}")
    
    # Storage for results
    split_dataframes = {}
    all_diversity_scores = {}
    all_summaries = []

    # Process each split
    for split, split_base_path in data_splits.items():
        print(f"\n🔍 Processing split: {split}")
        
        # Find available datasets
        dataset_paths = []
        for prefix in dataset_prefixes:
            full_path = os.path.join(split_base_path, prefix)
            if os.path.exists(full_path):
                dataset_paths.append(full_path)
            else:
                print(f"  ⚠️ Dataset not found: {full_path}")
        
        if not dataset_paths:
            print(f"  ❌ No datasets found for split '{split}' in {split_base_path}")
            continue

        try:
            combined_df, diversity_scores, split_summary = analyze_split(split, dataset_paths)
            split_dataframes[split] = combined_df
            all_diversity_scores[split] = diversity_scores
            all_summaries.extend(split_summary)
            
        except Exception as e:
            print(f"  ❌ Error processing split '{split}': {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cross-split comparison
    if len(split_dataframes) > 1:
        comparison_summary = compare_splits(split_dataframes, all_diversity_scores)
        all_summaries.extend(comparison_summary)

    # Final combined analysis
    if split_dataframes:
        final_summary = analyze_final_combined(list(split_dataframes.values()))
        all_summaries.extend(final_summary)

    # Generate comprehensive report
    print(f"\n📝 Generating comprehensive report...")
    summary_header = [
        "COMPREHENSIVE DATASET ANALYSIS REPORT",
        "=" * 60,
        f"Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tokenizer used: {tokenizer.name_or_path}",
        f"Total splits processed: {len(split_dataframes)}",
        f"Total samples analyzed: {sum(len(df) for df in split_dataframes.values()):,}",
        f"Diversity variables: {', '.join(variables)}",
        "",
        "SUMMARY OF OUTPUTS:",
        "Per-split analysis:",
        "- analysis_output/<split>/<split>_column_metrics.csv (detailed column statistics)",  
        "- analysis_output/<split>/<split>_diversity_details.csv (diversity metrics)",
        "- analysis_output/<split>/<split>_<col>_top_values.csv (top values per column)",
        "- analysis_output/<split>/*.png (visualizations)",
        "",
        "Combined dataset analysis:",
        "- analysis_output/final_combined/combined_column_metrics.csv (combined column statistics)",
        "- analysis_output/final_combined/combined_<col>_top_values.csv (top values for combined data)",
        "- analysis_output/final_combined/combined_tokenization_analysis.csv (tokenization details)",
        "- analysis_output/final_combined/final_diversity_metrics.csv (diversity metrics)",
        "- analysis_output/final_combined/*.png (combined visualizations)",
        "- analysis_output/final_combined/final_summary_stats.txt (summary statistics)",
        "",
        "Reports:",
        "- analysis_output/comprehensive_report.txt (this report)",
        "",
        "=" * 60,
        ""
    ]

    # Write comprehensive report
    try:
        with open("analysis_output/comprehensive_report.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(summary_header + all_summaries))
        print("✅ Comprehensive report saved to: analysis_output/comprehensive_report.txt")
    except Exception as e:
        print(f"❌ Error saving report: {e}")

    print(f"\n🎉 Analysis Complete!")
    print(f"📁 Results saved in: analysis_output/")
    print(f"📊 Splits processed: {list(split_dataframes.keys())}")
    print(f"📈 Total samples: {sum(len(df) for df in split_dataframes.values()):,}")

if __name__ == "__main__":
    main()