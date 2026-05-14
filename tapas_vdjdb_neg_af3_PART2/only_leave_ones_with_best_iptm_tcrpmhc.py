import pandas as pd

def main():
    # 1. Read the input CSV
    input_file = 'results_model_quality_metrics.csv'
    output_file = 'results_model_quality_metrics_best.csv'

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(
            f"Error: could not find '{input_file}'. "
            "Make sure the file exists in the current directory."
        )
        return

    # 2. Keep one row per pdb_id with the largest iptm_tcrpmhc (idxmax per group)
    best_models_idx = df.groupby('pdb_id')['iptm_tcrpmhc'].idxmax()
    df_best = df.loc[best_models_idx]

    # Optional: sort by pdb_id
    # df_best = df_best.sort_values(by='pdb_id').reset_index(drop=True)

    # 3. Write the filtered table
    df_best.to_csv(output_file, index=False)

    print(f"Wrote best-scoring row per pdb_id to '{output_file}'.")

if __name__ == '__main__':
    main()
