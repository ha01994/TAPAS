import subprocess

CONFIGS = [
    (
        "/shared/ha01994/iptm_filtered_neg_notdone_0_output",
        "sites_vdjdbiedb_neg_0.txt",
        "results_model_quality_metrics_0_best.csv",
    ),
    (
        "/shared/ha01994/iptm_filtered_neg_notdone_1_output",
        "sites_vdjdbiedb_neg_1.txt",
        "results_model_quality_metrics_1_best.csv",
    ),
    (
        "/shared/ha01994/iptm_filtered_neg_notdone_2_output",
        "sites_vdjdbiedb_neg_2.txt",
        "results_model_quality_metrics_2_best.csv",
    ),
]


for base_dir, output,filename in CONFIGS:
    subprocess.run(
        [
            "python",
            "make_sites_file.py",
            "--base_dir",
            base_dir,
            "--csv",
            filename,
            "--output",
            output,
            "--cutoff",
            "10.0",
            "--n_jobs",
            "16",
        ],
        check=True,
    )
