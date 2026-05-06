import pandas as pd

# Method 1: specify filenames directly
files = ['results_model_quality_metrics_0.csv', 
         'results_model_quality_metrics_1.csv', 
         'results_model_quality_metrics_2.csv']
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv('results_model_quality_metrics.csv', index=False)