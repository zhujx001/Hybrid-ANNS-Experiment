import os
import subprocess

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


labelfilter_data = "/data/HybridANNS/data/Experiment/labelfilterData"
stitched_temp = "/data/HybridANNS/data/Experiment/temp/stitched_ood"
filtered_temp = "/data/HybridANNS/data/Experiment/temp/filtered_ood"

K = 10
alpha = 1.2 
T_s = 1 
T_m = 16
T_b = 32

# filtered 
R_e = 128
L_e = 180

# stitched 
s_R_e = 48
s_L_e = 200
s_SR_e = 96


L_values_e = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,600, 650, 700, 750, 800, 850, 900, 950, 1000,1100,1200,1300,1400,3000]

def run_commands(): 
    # ============ Stitched  build index============
    index_path = os.path.join(stitched_temp, "index")
    result_path = os.path.join(stitched_temp, "result")
    os.makedirs(index_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    subprocess.run([
        "../../algorithm/DiskANN/build/apps/build_stitched_index",
        "--data_type", "float",
        "--data_path", os.path.join(labelfilter_data, "datasets/text2image/text2image_base.fbin"),
        "--index_path_prefix", f"{index_path}/_",
        "-R", f"{s_R_e}",
        "-L", f"{s_L_e}",
        "--stitched_R", f"{s_SR_e}",
        "--alpha", f"{alpha}",
        "-T", f"{T_b}",
        "--label_file", os.path.join(labelfilter_data, "labels/text2image/ung_label_1.txt")
    ], check=True)

    # Stitched single thread
    for L in L_values_e:
        subprocess.run([
            "../../algorithm/DiskANN/build/apps/search_memory_index",
            "--data_type", "float",
            "--dist_fn", "l2",
            "--index_path_prefix", f"{index_path}/_",
            "--query_file", os.path.join(labelfilter_data, "datasets/text2image/text2image_query.fbin"),
            "--gt_file", os.path.join(labelfilter_data, "gt/text2image/gt-query_set_1.bin"),
            "--query_filters_file", os.path.join(labelfilter_data, "query_label/text2image/ung_1.txt"),
            "-K", f"{K}",
            "-T", f"{T_s}",
            "-L", f"{L}",
            "--result_path", os.path.join(result_path, "1_")
        ], check=True)

    # ============ Filtered build index ============
    index_path = os.path.join(filtered_temp, "index")
    result_path = os.path.join(filtered_temp, "result")
    os.makedirs(index_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    subprocess.run([
        "../../algorithm/DiskANN/build/apps/build_memory_index", 
        "--data_type", "float",
        "--dist_fn", "l2",
        "--data_path", os.path.join(labelfilter_data, "datasets/text2image/text2image_base.fbin"),
        "--index_path_prefix", f"{index_path}/_",
        "-R", f"{R_e}",
        "--FilteredLbuild", f"{L_e}",
        "--alpha", f"{alpha}",
        "-T", f"{T_b}",
        "--label_file", os.path.join(labelfilter_data, "labels/text2image/ung_label_1.txt")
    ], check=True)

    # Filtered single thread
    for L in L_values_e:
        subprocess.run([
            "../../algorithm/DiskANN/build/apps/search_memory_index",
            "--data_type", "float",
            "--dist_fn", "l2",
            "--index_path_prefix", f"{index_path}/_",
            "--query_file", os.path.join(labelfilter_data, "datasets/text2image/text2image_query.fbin"),
            "--gt_file", os.path.join(labelfilter_data, "gt/text2image/gt-query_set_1.bin"),
            "--query_filters_file", os.path.join(labelfilter_data, "query_label/text2image/ung_1.txt"),
            "-K", f"{K}",
            "-T", f"{T_s}",
            "-L", f"{L}",
            "--result_path", os.path.join(result_path, "1_")
        ], check=True)    
       

def main():
    run_commands()

if __name__ == "__main__":
    main()