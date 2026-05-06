import csv
import json
import os
import math


csv_file_path = 'vdjdb_iedb_inputs_alphafast.csv'
base_output_dir = "vdjdb_iedb_alphafast"

rows = []
with open(csv_file_path, mode='r') as f:
    r = csv.reader(f)
    for line in r:
        rows.append(line)



for jj in [0,1,2]: 
    if jj == 0: select_rows = rows[:6000]
    if jj == 1: select_rows = rows[6000:12000]
    if jj == 2: select_rows = rows[12000:]
        
    output_dir = f"{base_output_dir}_{jj}"    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for row in select_rows:        
        # CSV format assumed: id, va, vb, pep, hla, b2m
        entry_id = row[0]
        va_seq  = row[1]
        vb_seq  = row[2]
        pep_seq = row[3]
        hla_seq = row[4]
        b2m_seq = row[5]

        # Build AlphaFast (AlphaFold 3) input format
        data = {
            "name": entry_id,
            "sequences": [
                {"protein": {"id": ["A"], "sequence": vb_seq}},  # vb -> A
                {"protein": {"id": ["B"], "sequence": va_seq}},  # va -> B
                {"protein": {"id": ["C"], "sequence": hla_seq}}, # hla -> C
                {"protein": {"id": ["D"], "sequence": b2m_seq}}, # b2m -> D
                {"protein": {"id": ["E"], "sequence": pep_seq}}  # pep -> E
            ],
            "modelSeeds": [1, 2, 3],
            "dialect": "alphafold3",
            "version": 3
        }

        # Save file (filename: id.json)
        file_path = os.path.join(output_dir, f"{entry_id}.json")
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2)



