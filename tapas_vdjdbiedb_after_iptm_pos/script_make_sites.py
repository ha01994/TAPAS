import subprocess

subprocess.run('python make_sites_file.py \
                --base_dir /shared/ha01994/alphafast_vdjdb_iedb/vdjdb_iedb_alphafast_output/ \
                --csv results_vdjdbiedb_iptm_filtered_best.csv \
                --output sites_vdjdbiedb.txt \
                --cutoff 10.0 \
                --n_jobs 16',shell=True)
