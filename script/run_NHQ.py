#!/usr/bin/env python3
import os
import subprocess
import argparse

# ========= 固定路径 =========
BASE_DATA_PATH = "/data/HybridANNS/data"
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
NHQ_PATH = os.path.join(SCRIPT_PATH, "../algorithm/NHQ-master")

LABEL_PATH = os.path.join(BASE_DATA_PATH, "Experiment/labelfilterData/labels")
QUERY_PATH = os.path.join(BASE_DATA_PATH, "Experiment/labelfilterData/query_label")
GROUNDTRUTH_PATH = os.path.join(BASE_DATA_PATH, "Experiment/labelfilterData/gt")
INDEX_PATH = os.path.join(BASE_DATA_PATH, "Experiment/Result/labelfilter/indexdata/NHQ")
# 修正数据文件的路径
DATA_PATH = os.path.join(BASE_DATA_PATH, "Experiment/labelfilterData/datasets")

# ========= 所有数据集参数 =========
dataset_params = {
    "audio":      "K=100; L=130; ITER=12; S=10; R=100; RANGE=20; PL=250; B=0.6; M=1.0",
    "enron":      "K=100; L=100; ITER=12; S=10; R=100; RANGE=20; PL=50; B=0.2; M=1.0",
    "gist":       "K=100; L=100; ITER=12; S=10; R=300; RANGE=20; PL=50; B=0.6; M=1.0",
    "glove-100":  "K=100; L=100; ITER=12; S=10; R=100; RANGE=20; PL=150; B=0.2; M=1.0",
    "msong":      "K=400; L=420; ITER=12; S=10; R=300; RANGE=80; PL=150; B=0.6; M=1.0",
    "sift":       "K=100; L=100; ITER=12; S=10; R=300; RANGE=20; PL=350; B=0.4; M=1.0"
}

# ========= 查询集映射 =========
query_sets = {
    "1": "query_set_1", "2_1": "query_set_2_1", "2_2": "query_set_2_2",
    "3_1": "query_set_3_1", "3_2": "query_set_3_2", "3_3": "query_set_3_3", "3_4": "query_set_3_4",
    "4": "query_set_4", "5_1": "query_set_5_1", "5_2": "query_set_5_2", "5_3": "query_set_5_3", "5_4": "query_set_5_4",
    "6": "query_set_6",
   
}

# ========= 参数解析 =========
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", help="Specify datasets (default: all)")
parser.add_argument("--queries", nargs="+", help="Specify query keys (default: all)")
args = parser.parse_args()

# ========= 编译模块 =========
def compile_all():
    import shutil

    # 定义需要清理的 build 目录
    clean_dirs = [
        "../algorithm/NHQ-master/NHQ-NPG_kgraph/build",
        "../algorithm/NHQ-master/NPG_kgraph/build"
    ]
    
    # 先清理旧的 build 目录（防止 CMakeCache.txt 错误）
    for build_dir in clean_dirs:
        abs_path = os.path.abspath(os.path.join(SCRIPT_PATH, build_dir))
        if os.path.exists(abs_path):
            print(f"🧹 Removing old build directory: {abs_path}")
            shutil.rmtree(abs_path)

    # 重新执行构建命令
    cmds = [
        "cd ../algorithm/NHQ-master/NHQ-NPG_kgraph && mkdir -p build && cd build && cmake .. && make -j && cd ../../../..",
        "cd ../algorithm/NHQ-master/NHQ-NPG_nsw && make -j && cd examples/cpp && make -j && cd ../../../..",
        "cd ../algorithm/NHQ-master/NPG_kgraph && mkdir -p build && cd build && cmake .. && make -j && cd ../../../..",
        "cd ../algorithm/NHQ-master/NPG_nsw/n2 && make -j && cd examples/cpp && make -j && cd ../../../../.."
    ]

    for cmd in cmds:
        print(f"🛠️ {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    print("✅ Compilation completed\n")


# ========= 构建索引 =========
def build_index(dataset, query_key):
    # 检查必要文件是否存在
    base_vecs = os.path.join(DATA_PATH, dataset, f"{dataset}_base.fvecs")
    label_file = os.path.join(LABEL_PATH, dataset, f"label_{query_key.replace('_', '-')}.txt")
    
    # 检查基本数据文件是否存在
    if not os.path.isfile(base_vecs):
        print(f"❌ Base vector file not found: {base_vecs}")
        return False
    
    if not os.path.isfile(label_file):
        print(f"❌ Label file not found: {label_file}")
        return False
    
    params = dict(x.split("=") for x in dataset_params[dataset].split("; "))
    index_dir = os.path.join(INDEX_PATH, f"NHQ_{dataset}_kgraph_{query_key.replace('_', '-')}")

    os.makedirs(index_dir, exist_ok=True)

    bin_path = os.path.join(NHQ_PATH, "NHQ-NPG_kgraph/build/tests/test_dng_index")
    
    # 检查执行文件是否存在
    if not os.path.isfile(bin_path):
        print(f"❌ Binary file not found: {bin_path}")
        return False
    
    cmd = [
        "taskset", "-c", "0-31", bin_path,
        base_vecs, label_file,
        os.path.join(index_dir, "index"),
        os.path.join(index_dir, "table"),
        params["K"], params["L"], params["ITER"], params["S"], params["R"],
        params["RANGE"], params["PL"], params["B"], params["M"]
    ]
    print(f"🔨 Building index: {dataset}, query {query_key}")
    print(f"📦 Command: {' '.join(str(c) for c in cmd)}\n")
    subprocess.run(cmd, check=True)
    return True

# ========= 执行查询 =========
def search(dataset, query_key):
    base_file = os.path.join(DATA_PATH, dataset, f"{dataset}_base.fvecs")
    query_vecs = os.path.join(DATA_PATH, dataset, f"{dataset}_query.fvecs")
    label_file = os.path.join(QUERY_PATH, dataset, f"{query_key}.txt")
    gt_file = os.path.join(GROUNDTRUTH_PATH, dataset, f"gt-{query_sets[query_key]}.ivecs")
    index_dir = os.path.join(INDEX_PATH, f"NHQ_{dataset}_kgraph_{query_key.replace('_', '-')}")
    
    # 检查所有必需文件
    required_files = {
        "Base vector file": base_file,
        "Query vector file": query_vecs,
        "Label file": label_file,
        "Ground truth file": gt_file,
        "Index file": os.path.join(index_dir, "index"),
        "Table file": os.path.join(index_dir, "table")
    }
    
    # 检查每个文件是否存在
    missing_files = []
    for desc, file_path in required_files.items():
        if not os.path.isfile(file_path):
            missing_files.append(f"{desc}: {file_path}")
    
    # 如果有缺失文件，报告并返回
    if missing_files:
        print(f"⚠️ Missing files for {dataset}/{query_key}:")
        for missing in missing_files:
            print(f"   - {missing}")
        return False

    bin_path = os.path.join(NHQ_PATH, "NHQ-NPG_kgraph/build/tests/test_dng_optimized_search")
    
    # 检查执行文件是否存在
    if not os.path.isfile(bin_path):
        print(f"❌ Binary file not found: {bin_path}")
        return False
    
    cmd = [bin_path,
           os.path.join(index_dir, "index"),
           os.path.join(index_dir, "table"),
           base_file, query_vecs, label_file, gt_file, dataset]

    print(f"🔍 Running search: {dataset}, query {query_key}")
    print(f"📦 Command: {' '.join(str(c) for c in cmd)}\n")
    subprocess.run(cmd, check=True)
    return True

# ========= 主流程 =========
if __name__ == "__main__":
    # compile_all()
    datasets = args.datasets if args.datasets else list(dataset_params.keys())
    queries = args.queries if args.queries else list(query_sets.keys())

    for dataset in datasets:
        if dataset not in dataset_params:
            print(f"❌ Unknown dataset: {dataset}")
            continue
        for query_key in queries:
            if query_key not in query_sets:
                print(f"❌ Unknown query: {query_key}")
                continue
            try:
                if build_index(dataset, query_key):
                    search(dataset, query_key)
                else:
                    print(f"⚠️ Skipping search for {dataset}, query {query_key} due to index build failure")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed on {dataset}, query {query_key}: {e}")