
import subprocess
import os

# 🗂️ 通用路径前缀配置
BASE_PATH = "/data/HybridANNS/data/Experiment/rangefilterData"
RESULT_PATH = "/data/HybridANNS/data/Experiment/temp/SeRF/result"
BUILD_PATH = "../../algorithm/SeRF/build/benchmark"

# 🔧 不同数据集的配置
tasks = [
    {
        "binary": os.path.join(BUILD_PATH, "text2image"),
        "name": "text2image",
        "N": "10000000",
        "dataset_path": f"{BASE_PATH}/datasets/text2image/text2image-base.fvecs", 
        "query_path": f"{BASE_PATH}/datasets/text2image/text2image-query.fvecs",
        "range_file_prefix": f"{BASE_PATH}/query_range/text2image/text2image-200-euclidean_queries_2pow-", 
        "result_prefix": f"{RESULT_PATH}/text2image_2pow-"
    }
]

# 📈 多个 2pow 范围值
range_values = [2]

# 🏃 执行命令
cmd_id = 1
for task in tasks:
    for r in range_values:
        cmd = [
            task["binary"], "-N", task["N"],
            "-dataset_path", task["dataset_path"],
            "-query_path", task["query_path"],
            "-range_file", f"{task['range_file_prefix']}{r}_ranges.txt",
            "-result_file", f"{task['result_prefix']}{r}.csv"
        ]

        print(f"\n🟢 正在执行命令 {cmd_id}: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"✅ 命令 {cmd_id} 执行成功！输出如下：\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"❌ 命令 {cmd_id} 执行失败！错误如下：\n{e.stderr}")
        cmd_id += 1