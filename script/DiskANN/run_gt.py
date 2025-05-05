import subprocess
import struct
import numpy as np
from pathlib import Path

# ====== 配置 ======
root_dir = Path("/data/HybridANNS/data/Experiment/labelfilterData")
temp_gt_dir = Path("/data/HybridANNS/data/Experiment/temp/diskann/gt")
utils_dir = Path("../../algorithm/DiskANN/build/apps/utils")

label_types = ["glove-100", "audio", "sift", "gist", "msong", "enron"]
basic_ids = ["1", "2-1", "2-2", "3-1", "3-2", "3-3", "3-4", "4", "5-1", "5-2", "5-3", "5-4"]
query_ids = ["1", "2_1", "2_2", "3_1", "3_2", "3_3", "3_4", "4", "5_1", "5_2", "5_3", "5_4"]


# === fvecs to bin 转换 ===
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
        f.write(struct.pack('i', n))       # int32 写 n
        f.write(struct.pack('i', dim))     # int32 写 dim
        f.write(vectors_np.tobytes())      # 写所有向量数据

    print(f"✅ 转换成功: {fvecs_path} → {bin_path} (n={n}, dim={dim})")
# ====== 遍历 ======
for label_type in label_types:
    for basic_id, query_id in zip(basic_ids, query_ids):
        # --- 文件路径 ---
        old_base_label_file = root_dir / "labels" / label_type / f"label_{basic_id}.txt"
        old_query_label_file = root_dir / "query_label" / label_type / f"{query_id}.txt"
        new_base_label_file = root_dir / "labels" / label_type / f"diskann_label_{basic_id}.txt"
        new_query_label_file = root_dir / "query_label" / label_type / f"diskann_{query_id}.txt"

        base_fvecs = root_dir / "datasets" / label_type / f"{label_type}_base.fvecs"
        query_fvecs = root_dir / "datasets" / label_type / f"{label_type}_query.fvecs"

        base_bin_file = root_dir / "datasets" / label_type / f"{label_type}_base.bin"
        query_bin_file = root_dir / "datasets" / label_type / f"{label_type}_query.bin"

        gt_file = temp_gt_dir / f"{label_type}_{query_id}.bin"

        # --- fvecs 转 bin ---
        if not base_bin_file.exists():
            print(f"🔁 Converting {base_fvecs} to {base_bin_file} ...")
            convert_fvecs_to_bin(str(base_fvecs), str(base_bin_file))
        if not query_bin_file.exists():
            print(f"🔁 Converting {query_fvecs} to {query_bin_file} ...")
            convert_fvecs_to_bin(str(query_fvecs), str(query_bin_file))

        # --- label 转换（去掉首行 + 用逗号隔开）---
        if old_base_label_file.exists():
            with old_base_label_file.open("r") as fin, new_base_label_file.open("w") as fout:
                lines = fin.readlines()[1:]  # 跳过首行
                for line in lines:
                    tokens = line.strip().split()
                    fout.write(",".join(tokens) + "\n")
        else:
            print(f"⚠️ Base label file not found: {old_base_label_file}")
            continue

        if old_query_label_file.exists():
            with old_query_label_file.open("r") as fin, new_query_label_file.open("w") as fout:
                lines = fin.readlines()[1:]  # 跳过首行
                for line in lines:
                    tokens = line.strip().split()
                    fout.write(",".join(tokens) + "\n")
        else:
            print(f"⚠️ Query label file not found: {old_query_label_file}")
            continue

        # --- 构建 groundtruth ---
        print(f"🚀 Generating groundtruth for {label_type} / {query_id} ...")
        subprocess.run([
            str(utils_dir / "compute_groundtruth_for_filters"),
            "--data_type", "float",
            "--dist_fn", "l2",
            "--K", "10",
            "--base_file", str(base_bin_file),
            "--label_file", str(new_base_label_file),
            "--query_file", str(query_bin_file),
            "--filter_label_file", str(new_query_label_file),
            "--gt_file", str(gt_file),
        ], check=True)