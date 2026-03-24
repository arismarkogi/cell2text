import json
from collections import Counter
import pandas as pd # Using pandas for a cleaner table printout

# --- Configuration ---
# 1. Replace 'your_file.json' with the actual path to your file.
json_filename = '/home/arism/save/scGPT_disease_custom_20251023_225113/predictions_and_labels.json' 
# ---------------------

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000) # Optional: Widen the display area


def print_histograms(filename):
    """
    Loads a JSON file, counts items in 'predictions' and 'labels' lists,
    and prints the counts as a table.
    """
    try:
        # Open and load the JSON file
        with open(filename, 'r') as f:
            data = json.load(f)

        # Check if the required keys exist
        if 'predictions' not in data or 'labels' not in data:
            print(f"Error: The JSON file must contain keys named 'predictions' and 'labels'.")
            return

        # Get the lists
        predictions = data['predictions']
        labels = data['labels']

        # Count the occurrences of each item
        # Counter creates a dictionary-like object: {'category_name': count}
        predictions_counts = Counter(predictions)
        labels_counts = Counter(labels)

        print(f"File: {filename}\n")

        # --- Option 1: Print using pandas (Recommended for a clean table) ---
        print("--- Histograms / Counts Table ---")
        
        # Create DataFrames from the counters
        pred_df = pd.DataFrame(predictions_counts.items(), columns=['Category', 'Predictions Count'])
        label_df = pd.DataFrame(labels_counts.items(), columns=['Category', 'Labels Count'])
        
        # Merge the two dataframes on the 'Category' column
        # 'outer' merge ensures all categories from both lists are included
        df_merged = pd.merge(pred_df, label_df, on='Category', how='outer')
        
        # Fill missing values (where a category is in one list but not the other) with 0
        df_merged = df_merged.fillna(0)
        
        # Convert counts to integers (as fillna(0) might make them floats)
        df_merged['Predictions Count'] = df_merged['Predictions Count'].astype(int)
        df_merged['Labels Count'] = df_merged['Labels Count'].astype(int)
        
        # Set category as the index for cleaner printing
        df_merged = df_merged.set_index('Category')
        
        print(df_merged)


        # --- Option 2: Basic printing (if you don't have pandas) ---
        # print("\n--- Basic Counts ---")
        # print("\n'predictions' counts:")
        # for item, count in predictions_counts.items():
        #     print(f"  {item}: {count}")
        #
        # print("\n'labels' counts:")
        # for item, count in labels_counts.items():
        #     print(f"  {item}: {count}")

    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON. Check if the file is a valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Run the function ---
print_histograms(json_filename)