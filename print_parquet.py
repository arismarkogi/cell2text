import pandas as pd

def print_column_head_parquet(file_path, column_name, num_lines=5):
  """
  Prints the full content of a specific column for the first few rows
  from a Parquet file.

  Args:
    file_path (str): The path to the Parquet file.
    column_name (str): The name of the column to print.
    num_lines (int, optional): The number of lines to print. Defaults to 5.
  """
  try:
    df = pd.read_parquet(file_path, columns=[column_name])
    if column_name in df.columns:
      with pd.option_context('display.max_colwidth', None):
        print(df.head(num_lines))
    else:
      print(f"Error: Column '{column_name}' not found in the Parquet file.")
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
  except Exception as e:
    print(f"An error occurred: {e}")

# Example usage:
parquet_file = '/home/arismarkog/Desktop/datasets/processed.parquet'  # Replace with the actual path
selected_column = 'text_desc'    # Replace with the name of the column you want
print_column_head_parquet(parquet_file, selected_column)