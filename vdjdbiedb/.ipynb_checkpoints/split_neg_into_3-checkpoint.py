import os
import shutil

src = "iptm_filtered_neg_notdone"
dirs = [
    "iptm_filtered_neg_notdone_0",
    "iptm_filtered_neg_notdone_1",
    "iptm_filtered_neg_notdone_2",
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

files = sorted([f for f in os.listdir(src) if f.endswith(".json")])
print(f"Total files: {len(files)}")

for i, fname in enumerate(files):
    src_path = os.path.join(src, fname)
    
    if i < 6500:
        dst = dirs[0]
    
    elif i < 13000:
        dst = dirs[1]
    
    else:
        dst = dirs[2]
    
    shutil.move(src_path, os.path.join(dst, fname))

for d in dirs:
    print(f"{d}: {len(os.listdir(d))} files")
