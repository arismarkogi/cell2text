import json
import re
import difflib
from typing import Optional, Tuple, List
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
from pathlib import Path

from sklearn.metrics import roc_auc_score, average_precision_score

def compute_avg_auc(labels: np.ndarray, preds: np.ndarray):
    """Compute AUROC and AUPRC per pathway, then average (macro style)."""
    n_pathways = labels.shape[1]
    aurocs, auprcs = [], []
    for i in range(n_pathways):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        # Skip if a pathway has only one class present (undefined AUROC/AUPRC)
        if len(np.unique(y_true)) < 2:
            continue
        aurocs.append(roc_auc_score(y_true, y_pred))
        auprcs.append(average_precision_score(y_true, y_pred))
    return (np.mean(aurocs) if aurocs else 0.0,
            np.mean(auprcs) if auprcs else 0.0)


def compute_flatten_auc(labels: np.ndarray, preds: np.ndarray):
    """Flatten all sample–pathway pairs into one vector (micro style)."""
    y_true = labels.ravel()
    y_pred = preds.ravel()
    # Handle degenerate case where all labels are 0/1
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    return auroc, auprc


# --- Step 1: Load pathway descriptions and create reverse mapping ---
target_pathways = [
    "HALLMARK_ANDROGEN_RESPONSE",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_UV_RESPONSE_DN",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_HEDGEHOG_SIGNALING",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_ALLOGRAFT_REJECTION",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK_CHOLESTEROL_HOMEOSTASIS",
    "HALLMARK_ANGIOGENESIS",
    "HALLMARK_NOTCH_SIGNALING",
    "HALLMARK_MYC_TARGETS_V2",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_P53_PATHWAY",
    "HALLMARK_PANCREAS_BETA_CELLS",
    "HALLMARK_HYPOXIA",
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    "HALLMARK_APICAL_SURFACE",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_MYOGENESIS",
    "HALLMARK_COMPLEMENT",
    "HALLMARK_ESTROGEN_RESPONSE_LATE",
    "HALLMARK_HEME_METABOLISM",
    "HALLMARK_ESTROGEN_RESPONSE_EARLY",
    "HALLMARK_APICAL_JUNCTION",
    "HALLMARK_XENOBIOTIC_METABOLISM",
    "HALLMARK_COAGULATION",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_GLYCOLYSIS",
    "HALLMARK_BILE_ACID_METABOLISM",
    "HALLMARK_KRAS_SIGNALING_UP",
    "HALLMARK_SPERMATOGENESIS",
    "HALLMARK_IL2_STAT5_SIGNALING",
    "HALLMARK_KRAS_SIGNALING_DN"
]

# Load full descriptions
with open("pathway_descriptions.json", "r") as f:
    full_pathway_descriptions = json.load(f)

# Filter to target pathways only
pathway_descriptions = {k: v for k, v in full_pathway_descriptions.items() if k in target_pathways}

# Build description → key mapping (only for target pathways)
desc_to_key = {}
for key, desc in pathway_descriptions.items():
    norm_desc = " ".join(desc.lower().split())
    desc_to_key[norm_desc] = key

# Define label space
all_pathways = target_pathways  # preserve order
n_pathways = len(all_pathways)
pathway_to_index = {key: i for i, key in enumerate(all_pathways)}

# --- Step 2: Fuzzy match description to HALLMARK key ---
def find_best_matching_key(description: str, threshold: float = 0.97) -> Optional[str]:
    norm_desc = " ".join(description.lower().split())
    best_match = None
    best_score = 0.0

    for ref_desc, key in desc_to_key.items():
        score = difflib.SequenceMatcher(None, norm_desc, ref_desc).ratio()
        if score > best_score:
            best_score = score
            best_match = key

    if best_score >= threshold:
        return best_match
    else:
        return None

# --- Step 3: Extract pathways from text ---
def extract_pathways(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract two pathway descriptions using fixed delimiters — NOT regex.
    Delimiters:
      part1_start = ". This cell is associated with "
      part2_start = ". Additionally, it involves "
    Second pathway goes until end of string.
    """
    part1_start = ". This cell is associated with "
    part2_start = ". Additionally, it involves "

    # Find first delimiter
    idx1 = text.find(part1_start)
    if idx1 == -1:
        return None, None

    # Start of pathway 1 description
    start_p1 = idx1 + len(part1_start)
    
    # Find start of pathway 2
    idx2 = text.find(part2_start, start_p1)
    if idx2 == -1:
        return None, None

    # Pathway 1 = from after part1_start to just before part2_start
    pathway1 = text[start_p1:idx2].strip()
    pathway1 += '.'  # Add period back for consistency

    # Pathway 2 = from after part2_start to end of string
    start_p2 = idx2 + len(part2_start)
    pathway2 = text[start_p2:].strip()
    pathway2 += '.'  # Add period back for consistency
    

    # Optional: remove trailing period if exists (but don't require it)
    if pathway2.endswith('.'):
        pathway2 = pathway2[:-1].strip()

    return pathway1, pathway2

# --- Step 4: Convert 2 pathways to binary vector ---
def pathways_to_vector(pathways: List[str]) -> np.ndarray:
    vec = np.zeros(n_pathways, dtype=int)
    for p in pathways:
        if p in pathway_to_index:
            vec[pathway_to_index[p]] = 1
    return vec



def process_sample(item):
    true_text = item["true"]
    pred_text = item["pred"]

    # Extract descriptions
    true_p1_desc, true_p2_desc = extract_pathways(true_text)
    pred_p1_desc, pred_p2_desc = extract_pathways(pred_text)

    # Map to HALLMARK keys
    true_p1_key = find_best_matching_key(true_p1_desc) if true_p1_desc else None
    true_p2_key = find_best_matching_key(true_p2_desc) if true_p2_desc else None
    pred_p1_key = find_best_matching_key(pred_p1_desc) if pred_p1_desc else None
    pred_p2_key = find_best_matching_key(pred_p2_desc) if pred_p2_desc else None

    true_keys = [k for k in [true_p1_key, true_p2_key] if k is not None]
    pred_keys = [k for k in [pred_p1_key, pred_p2_key] if k is not None]

    # Sort for order-invariant comparison
    true_keys_sorted = sorted(true_keys)
    pred_keys_sorted = sorted(pred_keys)

    # Convert to binary vectors
    true_vec = pathways_to_vector(true_keys)
    pred_vec = pathways_to_vector(pred_keys)

    return true_vec, pred_vec, {
        "true_descriptions": [true_p1_desc, true_p2_desc],
        "pred_descriptions": [pred_p1_desc, pred_p2_desc],
        "true_pathways": true_keys_sorted,
        "pred_pathways": pred_keys_sorted,
        "is_correct": true_keys_sorted == pred_keys_sorted
    }

def main():
    # --- Step 5: Handle command-line arguments and load predictions ---
    parser = argparse.ArgumentParser(description="Evaluate a model's pathway predictions.")
    parser.add_argument("predictions_file", type=str,
                        help="Path to the predictions JSON file.")
    args = parser.parse_args()

    # Create a Path object for easier file path manipulation
    pred_path = Path(args.predictions_file)
    if not pred_path.exists():
        print(f"Error: Predictions file not found at {pred_path}")
        return

    with open(pred_path, "r") as f:
        predictions = json.load(f)

    # --- Step 6: Process all samples ---
    print("Processing samples...")
    results = Parallel(n_jobs=-1)(delayed(process_sample)(item) for item in tqdm(predictions, desc="Mapping pathways"))

    labels_list, preds_list, sample_results = zip(*results)

    # --- Step 7: Convert to numpy arrays ---
    labels = np.array(labels_list)      # Shape: (n_samples, n_pathways)
    preds = np.array(preds_list)        # Shape: (n_samples, n_pathways)

    # --- Step 8: Calculate Metrics ---
    print("\nComputing metrics...")
    
    # Exact match accuracy (subset accuracy)
    subset_acc = accuracy_score(labels, preds)
    
    # Jaccard accuracy (partial credit for overlaps)
    
    # Alternative: sklearn's jaccard_score (sample-wise average)
    jaccard_acc = jaccard_score(labels, preds, average='samples', zero_division=0)
    
    # F1 scores
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    avg_auc = compute_avg_auc(labels, preds)
    flat_auc = compute_flatten_auc(labels, preds)

    metrics = {
        "subset_accuracy": subset_acc,
        "jaccard_accuracy": jaccard_acc,
        "avg_auroc": avg_auc[0],
        "avg_auprc": avg_auc[1],
        "flat_auroc": flat_auc[0],
        "flat_auprc": flat_auc[1],
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "num_samples": len(labels),
        "num_pathways": n_pathways
    }

    # Print results
    print(f"✅ Subset Accuracy:    {subset_acc:.4f}")
    print(f"✅ Jaccard Accuracy:   {jaccard_acc:.4f}")
    print(f"✅ Weighted F1:        {weighted_f1:.4f}")
    print(f"✅ Macro F1:           {macro_f1:.4f}")
    print(f"✅ Micro F1:           {micro_f1:.4f}")

    # Additional analysis
    print(f"\nAdditional Statistics:")
    total_true_labels = np.sum(labels)
    total_pred_labels = np.sum(preds)
    avg_true_per_sample = np.mean(np.sum(labels, axis=1))
    avg_pred_per_sample = np.mean(np.sum(preds, axis=1))
    
    print(f"  Total true labels: {total_true_labels}")
    print(f"  Total predicted labels: {total_pred_labels}")
    print(f"  Avg true labels per sample: {avg_true_per_sample:.2f}")
    print(f"  Avg predicted labels per sample: {avg_pred_per_sample:.2f}")

    # --- Step 9: Save everything to a dynamically named file ---
    # Get the base name of the predictions file (e.g., "predictions")
    base_name = pred_path.stem
    # Construct the new filename (e.g., "predictions_results.json")
    output_filename = f"pathway_results_{base_name}.json"
    output_path = pred_path.parent / output_filename
    
    output = {
        "metrics": metrics,
        #"sample_results": sample_results,
        "config": {
            "pathway_count": n_pathways,
            "pathways": all_pathways
        }
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to '{output_path}'")

if __name__ == "__main__":
    main()