# /*******************************************************
#  🔴 IMPORTANT NOTE:
#  
#  Since the text2image dataset is too large, 
#  you MUST COMMENT OUT the original `train` function 
#  in `CAPS/include/cluster.h` and use the MODIFIED train 
#  function (the commented part in `Kmeans::train`).
#  
#  >>> This is CRUCIAL for CAPS to work with large datasets <<<


import os
import subprocess

# 自动推导根路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/Experiment"))
CAPS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../algorithm/CAPS"))
TEMP_DIR = os.path.abspath(os.path.join(EXPERIMENT_ROOT, "../../data/Experiment/temp/CAPS/"))

# 数据集列表

dataset = "text2image"

# nprobe 值（500~30000 步长 500）
nprobe_values = [str(i) for i in range(500, 30500, 500)]

# 可执行文件路径
INDEX_EXEC = os.path.join(CAPS_DIR, "index")
QUERY_EXEC = os.path.join(CAPS_DIR, "query")

def build_caps_binaries():
    print(f"🛠️ Running make in {CAPS_DIR} ...")
    try:
        subprocess.run(["make", "index"], cwd=CAPS_DIR, check=True)
        subprocess.run(["make", "query"], cwd=CAPS_DIR, check=True)
        print("✅ CAPS binaries built successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error building CAPS binaries: {e}")
        exit(1)

def run_commands(dataset):    
    print(f"\n🔹 Running for dataset={dataset}")
    base_path = os.path.join(EXPERIMENT_ROOT, "labelfilterData/datasets", dataset, f"{dataset}_base.fvecs")
    base_attr = os.path.join(EXPERIMENT_ROOT, "labelfilterData/labels", dataset, f"label_1.txt")
    index_path = os.path.join(TEMP_DIR, "index", f"{dataset}_1")
    os.makedirs(index_path, exist_ok=True)  

    query_data = os.path.join(EXPERIMENT_ROOT, "labelfilterData/datasets", dataset, f"1_query.fvecs")
    query_attr = os.path.join(EXPERIMENT_ROOT, "labelfilterData/query_label", dataset, f"1.txt")
    gt_path = os.path.join(EXPERIMENT_ROOT, "labelfilterData/gt", dataset, f"gt-query_set_1.ivecs")
    result_path = os.path.join(TEMP_DIR, "result", f"{dataset}_1")
    os.makedirs(result_path, exist_ok=True) 
        # 构建索引
    subprocess.run([
        INDEX_EXEC, base_path, base_attr, index_path, "1024", "kmeans", "3"
    ], check=True)

    # 单线程查询
    for nprobe in nprobe_values:
        subprocess.run([
            QUERY_EXEC, base_path, base_attr, query_data, query_attr,
            index_path, gt_path, "1024", "kmeans", "3", nprobe, "1", dataset, result_path
        ], check=True)

       
            


def main():
    build_caps_binaries()
    run_commands(dataset)
if __name__ == "__main__":
    main()