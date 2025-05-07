import os
import subprocess
import numpy as np
import struct
ROOT_DIR = "/data/HybridANNS/data/Experiment/labelfilterData"
GT_DIR = "/data/HybridANNS/data/Experiment/temp/UNG/gt"
GT_TOOL = "../../algorithm/UNG/build/tools/compute_groundtruth"

LABEL_TYPES = ["msong", "gist", "sift", "glove-100", "enron", "audio"]
BASE_IDS_E = ["1", "3-1", "3-2", "3-3", "3-4", "4", "5-1", "5-2", "5-3", "5-4"]
QUERY_IDS_E = ["1", "3_1", "3_2", "3_3", "3_4", "4", "5_1", "5_2", "5_3", "5_4"]
BASE_IDS_C = ["2-1", "2-2"]
QUERY_IDS_C = ["2_1", "2_2"]

os.makedirs(GT_DIR, exist_ok=True)




# === fvecs to bin  ===
def convert_fvecs_to_bin(fvecs_path, bin_path):
    vectors = []
    dim = None

    with open(fvecs_path, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # EOF
            d = struct.unpack('i', dim_bytes)[0]
            if dim is None:
                dim = d
            elif dim != d:
                raise ValueError(f"维度不一致：{dim} != {d}")
            vec = f.read(4 * d)
            if len(vec) != 4 * d:
                raise ValueError("向量长度错误")
            vectors.append(struct.unpack(f'{d}f', vec))

    vectors_np = np.array(vectors, dtype=np.float32)
    n = vectors_np.shape[0]

    with open(bin_path, 'wb') as f:
        f.write(struct.pack('i', n))       
        f.write(struct.pack('i', dim))     
        f.write(vectors_np.tobytes())      

    print(f"✅ 转换成功: {fvecs_path} → {bin_path} (n={n}, dim={dim})")

# === label  ===
def convert_label_file(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()[1:]  
    with open(output_path, 'w') as f:
        for line in lines:
            line = line.strip()
            if line:
                values = line.split()
                f.write(','.join(values) + '\n')

# ===  GT ===
def generate_groundtruth(label_type, base_id, query_id, scenario):
    print(f"\n🚀 Generating GT ({scenario}) for {label_type} | base_id={base_id}, query_id={query_id}")

    base_label = os.path.join(ROOT_DIR, "labels", label_type, f"label_{base_id}.txt")
    query_label = os.path.join(ROOT_DIR, "query_label", label_type, f"{query_id}.txt")
    new_base_label = os.path.join(ROOT_DIR, "labels", label_type, f"ung_label_{base_id}.txt")
    new_query_label = os.path.join(ROOT_DIR, "query_label", label_type, f"ung_{query_id}.txt")

    base_fvecs = os.path.join(ROOT_DIR, "datasets", label_type, f"{label_type}_base.fvecs")
    query_fvecs = os.path.join(ROOT_DIR, "datasets", label_type, f"{label_type}_query.fvecs")
    base_bin = os.path.join(ROOT_DIR, "datasets", label_type, f"{label_type}_base.bin")
    query_bin = os.path.join(ROOT_DIR, "datasets", label_type, f"{label_type}_query.bin")
    gt_file = os.path.join(GT_DIR, f"{label_type}_{query_id}.bin")

    #  fvecs
    if not os.path.exists(base_bin) and os.path.exists(base_fvecs):
        convert_fvecs_to_bin(base_fvecs, base_bin)
    if not os.path.exists(query_bin) and os.path.exists(query_fvecs):
        convert_fvecs_to_bin(query_fvecs, query_bin)

    #  label
    if os.path.exists(base_label):
        convert_label_file(base_label, new_base_label)
    else:
        print(f"❌ Missing base label: {base_label}")
        return

    if os.path.exists(query_label):
        convert_label_file(query_label, new_query_label)
    else:
        print(f"❌ Missing query label: {query_label}")
        return

    #  compute_groundtruth 
    cmd = [
        GT_TOOL,
        "--data_type", "float",
        "--dist_fn", "L2",
        "--scenario", scenario,
        "--K", "10",
        "--num_threads", "32",
        "--base_bin_file", base_bin,
        "--base_label_file", new_base_label,
        "--query_bin_file", query_bin,
        "--query_label_file", new_query_label,
        "--gt_file", gt_file
    ]
    subprocess.run(cmd, check=True)

def main():
    for label_type in LABEL_TYPES:
        for base_id, query_id in zip(BASE_IDS_E, QUERY_IDS_E):
            generate_groundtruth(label_type, base_id, query_id, "equality") 
        for base_id, query_id in zip(BASE_IDS_C, QUERY_IDS_C):
            generate_groundtruth(label_type, base_id, query_id, "containment")

if __name__ == "__main__":
    main()