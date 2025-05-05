
import subprocess
# Define datasets, basic IDs, and query IDs
datasets = ["msong", "gist", "sift", "glove-100", "enron", "audio"]
basic_ids = ["1" ,"3-1" ,"3-2" ,"3-3" ,"3-4" ,"4" ,"5-1", "5-2" ,"5-3", "5-4"]
query_ids = ["1" ,"3_1" ,"3_2" ,"3_3" ,"3_4" ,"4" ,"5_1", "5_2" ,"5_3", "5_4"]

basic_ids_2 = ["2-1", "2-2"]
query_ids_2 = ["2_1", "2_2"]

def run_ung_commands(dataset, basic_id, query_id, scenario_b,scenario_s):
    labelfilter_root = "/data/HybridANNS/data/Experiment/labelfilterData"
    ung_temp_root = "/data/HybridANNS/data/Experiment/temp/UNG"

    base_bin_file = f"{labelfilter_root}/datasets/{dataset}/{dataset}_base.bin"
    base_label_file = f"{labelfilter_root}/labels/{dataset}/ung_label_{basic_id}.txt"
    query_bin_file = f"{labelfilter_root}/datasets/{dataset}/{dataset}_query.bin"
    query_label_file = f"{labelfilter_root}/query_label/{dataset}/ung_{query_id}.txt"

    index_path_prefix = f"{ung_temp_root}/index/{dataset}_{query_id}/"   # meta file contains information about building the index
    result_path_prefix = f"{ung_temp_root}/result/{dataset}_{query_id}/"
    gt_file = f"{ung_temp_root}/gt/{dataset}_{query_id}.bin"
    
    # Build UNG index
    print(f"Building UNG index for {dataset}...")
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
        "--index_path_prefix", index_path_prefix, 
        "--scenario", scenario_b, 
        "--num_cross_edges", "6"
    ], check=True)

    # Search UNG index (single thread)
    print(f"Searching UNG index for {dataset} (single thread)")
    subprocess.run([
        "../../algorithm/UNG/build/apps/search_UNG_index", 
        "--data_type", "float", 
        "--dist_fn", "L2", 
        "--num_threads", "1", 
        "--K", "10", 
        "--base_bin_file", base_bin_file, 
        "--base_label_file", base_label_file, 
        "--query_bin_file", query_bin_file, 
        "--query_label_file", query_label_file, 
        "--gt_file", gt_file, 
        "--index_path_prefix", index_path_prefix, 
        "--result_path_prefix", result_path_prefix + "1/", 
        "--scenario", scenario_s, 
        "--num_entry_points", "16", 
        "--Lsearch", "10", "25", "30", "50", "80", "100", "120", "140", "150", "170", "190", "200", "210", "230", "240", "250", "260", "280", "290", "300", "350", "400", "450", "500", "550", "600", "650", "700", "750", "800", "850", "900", "950", "1000", "1100", "1200", "1300", "1400", "1500", "1600", "1700", "1800", "1900", "2000", "3000"
    ], check=True)

    # Search UNG index (multi-threaded)
    print(f"Searching UNG index for {dataset} (16 threads)...")
    subprocess.run([
        "../../algorithm/UNG/build/apps/search_UNG_index", 
        "--data_type", "float", 
        "--dist_fn", "L2", 
        "--num_threads", "16", 
        "--K", "10", 
        "--base_bin_file", base_bin_file, 
        "--base_label_file", base_label_file, 
        "--query_bin_file", query_bin_file, 
        "--query_label_file", query_label_file, 
        "--gt_file", gt_file, 
        "--index_path_prefix", index_path_prefix, 
        "--result_path_prefix", result_path_prefix + "16/", 
        "--scenario", scenario_s, 
        "--num_entry_points", "16", 
        "--Lsearch", "10", "25", "30", "50", "80", "100", "120", "140", "150", "170", "190", "200", "210", "230", "240", "250", "260", "280", "290", "300", "350", "400", "450", "500", "550", "600", "650", "700", "750", "800", "850", "900", "950", "1000", "1100", "1200", "1300", "1400", "1500", "1600", "1700", "1800", "1900", "2000", "3000"
    ], check=True)

# Main function
def main():
    # Step 1: Run GT generation shell script
    print("🟡 Running groundtruth generation script (run_gt.sh)...")
    # subprocess.run(["bash", "run_gt.sh"], check=True)

    # Step 2: Continue with indexing and searching
    # Loop over datasets and execute commands for both equality and containment scenarios
    # for dataset in datasets:
    #     for basic_id, query_id in zip(basic_ids, query_ids):
    #         run_ung_commands(dataset, basic_id, query_id, "equality","equality")
    for dataset in datasets:
        for basic_id, query_id in zip(basic_ids_2, query_ids_2):
            run_ung_commands(dataset, basic_id, query_id, "general","containment")

if __name__ == "__main__":
    main()