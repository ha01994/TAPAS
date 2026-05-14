import subprocess


subprocess.run('python make_sites_file.py \
                --base_dir af3_vdjdb/af_output/ \
                --csv /home/ha01994/tabpfn_vdjdb_af3/results_model_quality_metrics_best.csv \
                --output sites_af3_vdjdb.txt \
                --cutoff 10.0 \
                --n_jobs 16',shell=True)
