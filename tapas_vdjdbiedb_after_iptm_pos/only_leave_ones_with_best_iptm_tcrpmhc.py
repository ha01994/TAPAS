import pandas as pd

def main():
    # 1. Read CSV file
    input_file = 'results_model_quality_metrics.csv'
    output_file = 'results_model_quality_metrics_vdjdbiedb_best.csv'
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: '{input_file}' file not found. Please check that the file is in the same folder.")
        return

    # 2. For each pdb_id, extract rows with the maximum iptm_tcrpmhc value.
    # Use idxmax() to get the index of the maximum value in each group.
    best_models_idx = df.groupby('pdb_id')['iptm_tcrpmhc'].idxmax()
    df_best = df.loc[best_models_idx]

    # (Optional) sort by pdb_id (if needed)
    # df_best = df_best.sort_values(by='pdb_id').reset_index(drop=True)

    # 3. Save the result to a new CSV file.
    df_best.to_csv(output_file, index=False)
    
    print(f"Successfully extracted only the best model and saved to '{output_file}' saved.")

if __name__ == '__main__':
    main()