import subprocess
import os

base_dir = "/data/HybridANNS/data/Experiment"
labelfilter_dir = os.path.join(base_dir, "labelfilterData/datasets/text2image")
label_dir =os.path.join(base_dir, "labelfilterData/labels/text2image")
query_label_dir = os.path.join(base_dir, "labelfilterData/query_label/text2image")
ung_index_dir = os.path.join(base_dir, "temp/UNG-ood/index/text2image")
ung_result_dir = os.path.join(base_dir, "temp/UNG-ood/result/text2image")
gt_output_dir = os.path.join(base_dir, "temp/UNG-ood/gt")

base_bin_file = os.path.join(labelfilter_dir, "text2image_base.fbin")
query_bin_file = os.path.join(labelfilter_dir, "text2image_query.fbin")
base_label_file = os.path.join(label_dir, "ung_label_1.txt")
query_label_file = os.path.join(query_label_dir, "ung_1.txt")
gt_file = os.path.join(gt_output_dir, "text2image.bin")

# num_cross_edges = ["1", "2", "6", "12"]
# num_entry_points_list = ["2", "4", "6", "8", "12", "24", "72", "128"]
num_cross_edges = ["6"]
num_entry_points_list = ["6"]
Lsearch_values = ["10", "15", "20", "25", "30", "50", "80", "100", "120", "140", "150", "170", "190", "200", "500", "1000", "2000"]

def build_commands(num_cross_edge):
    print(f"🛠️  Building UNG index with num_cross_edges={num_cross_edge}...")
    subprocess.run([
        "../../algorithm/UNG/build/apps/build_UNG_index",
        "--data_type", "float",
        "--dist_fn", "L2",
        "--num_threads", "32",
        "--max_degree", "32",
        "--Lbuild", "100",
        "--alpha", "1.2",
        "--base_bin_file", base_bin_file,
        "--base_label_file", base_label_file,
        "--index_path_prefix", os.path.join(ung_index_dir, f"{num_cross_edge}/"),
        "--scenario", "equality",
        "--num_cross_edges", num_cross_edge
    ], check=True)

def search_commands(num_cross_edge, num_entry_points):
    print(f"🔍  Searching UNG index with num_cross_edges={num_cross_edge}, entry_points={num_entry_points}...")
    subprocess.run([
        "../../algorithm/UNG/build/apps/search_UNG_index",
        "--data_type", "float",
        "--dist_fn", "L2",
        "--num_threads", "1",
        "--K", "10",
        "--base_bin_file", base_bin_file,
        "--base_label_file", base_label_file,
        "--query_bin_file", query_bin_file,
        "--gt_file", gt_file,
        "--query_label_file", query_label_file,
        "--index_path_prefix", os.path.join(ung_index_dir, f"{num_cross_edge}/"),
        "--result_path_prefix", os.path.join(ung_result_dir, f"{num_cross_edge}_{num_entry_points}_16/"),
        "--scenario", "equality",
        "--num_entry_points", str(num_entry_points),
        "--Lsearch"
    ] + Lsearch_values, check=True)

def get_gt():
    print("📦  Generating groundtruth...")
    subprocess.run([
        "../../algorithm/UNG/build/tools/compute_groundtruth",
        "--data_type", "float",
        "--dist_fn", "L2",
        "--scenario", "Equality",
        "--K", "10",
        "--num_threads", "32",
        "--base_bin_file", base_bin_file,
        "--base_label_file", base_label_file,
        "--query_bin_file", query_bin_file,
        "--query_label_file", query_label_file,
        "--gt_file", gt_file
    ], check=True)

def main():
    get_gt()
    for num_cross_edge in num_cross_edges:
        build_commands(num_cross_edge)
        for num_entry_points in num_entry_points_list:
            search_commands(num_cross_edge, num_entry_points)

if __name__ == "__main__":
    main()