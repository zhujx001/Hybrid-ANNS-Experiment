import os
import subprocess

# Define datasets and IDs
datasets = ["audio", "sift", "gist", "glove-100", "msong", "enron"]
basic_ids = ["1", "3-1", "3-2", "3-3", "3-4", "4", "5-1", "5-2", "5-3", "5-4"]
query_ids = ["1", "3_1", "3_2", "3_3", "3_4", "4", "5_1", "5_2", "5_3", "5_4"]
basic_ids_c=["2-1","2-2"]
query_ids_c=["2_1","2_2"]

# Define parameters
K = 10
alpha = 1.2
T_s, T_m, T_b = 1, 16, 32
L_search = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400]


# Filtered and stitched index parameters
R_e, L_e, s_R_e, s_L_e, s_SR_e = 96, 90, 32, 100, 64
R_c, L_c, s_R_c, s_L_c, s_SR_c = 128, 180, 48, 200, 96


# Function to create necessary directories
def create_dirs(dataset, query_id, index_type):
    index_path = f"/data/HybridANNS/data/Experiment/temp/diskann/index/{dataset}_{query_id}_{index_type}/"
    result_path = f"/data/HybridANNS/data/Experiment/temp/diskann/result/{dataset}_{query_id}_{index_type}/"
    
    # Check and create index_path
    if not os.path.exists(index_path):
        os.makedirs(index_path)
        print(f"✅ Created directory: {index_path}")
    else:
        print(f"ℹ️ Directory already exists: {index_path}")
    
    # Check and create result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"✅ Created directory: {result_path}")
    else:
        print(f"ℹ️ Directory already exists: {result_path}")
    
    return index_path, result_path


# Function to run the search command (single or multi-threaded)
def search_index(index_type, dataset, query_id, K, T, L, index_path, result_path, num_threads=1):
    print(f"Searching {index_type} index for {dataset} with L={L} ({num_threads} thread)...")
    subprocess.run([
        "../../algorithm/DiskANN/build/apps/search_memory_index",
        "--data_type", "float",
        "--dist_fn", "l2",
        "--index_path_prefix", f"{index_path}_",
        "--query_file", f"/data/HybridANNS/data/Experiment/labelfilterData/datasets/{dataset}/{dataset}_query.bin",
        "--gt_file", f"/data/filter-yjy/diskann/gt/{dataset}_{query_id}.bin",
        "--query_filters_file", f"/data/HybridANNS/data/Experiment/labelfilterData/query_label/{dataset}/diskann_{query_id}.txt",
        "-K", f"{K}",
        "-T", f"{T}",
        "-L", f"{L}",
        "--result_path", f"{result_path}{num_threads}_"
    ])

# Main function to run commands for filtered indexes
def run_filtered(dataset, basic_id, query_id, index_type,R,L):
    # Build the index
    index_path, result_path = create_dirs(dataset, query_id, index_type)
    subprocess.run([
        "../../algorithm/DiskANN/build/apps/build_memory_index" ,
        "--data_type", "float",
        "--dist_fn", "l2",
        "--data_path", f"/data/HybridANNS/data/Experiment/labelfilterData/datasets/{dataset}/{dataset}_base.bin",
        "--index_path_prefix", f"{index_path}_",
        "-R", f"{R}",
        "-L", f"{L}",
        "--alpha", f"{alpha}",
        "-T", f"{T_b}", 
        "--label_file", f"/data/HybridANNS/data/Experiment/labelfilterData/labels/{dataset}/diskann_label_{basic_id}.txt"
    ], check=True)
    # Run single-threaded search for each L value
    for L in L_search:
        search_index(index_type, dataset, query_id, K, T_s, L, index_path, result_path)

    # Run multi-threaded search (16 threads)
    for L in L_search:
        search_index(index_type, dataset, query_id, K, T_m, L, index_path, result_path, num_threads=16)
# Main function to run commands for stitched indexes
def run_stitched(dataset, basic_id, query_id, index_type,R,L,SR):
    # Build the index
    index_path, result_path = create_dirs(dataset, query_id, index_type)
    subprocess.run([
            "../../algorithm/DiskANN/build/apps/build_stitched_index" ,
            "--data_type", "float",
            "--data_path", f"/data/HybridANNS/data/Experiment/labelfilterData/datasets/{dataset}/{dataset}_base.bin",
            "--index_path_prefix", f"{index_path}_",
            "-R", f"{R}",
            "-L", f"{L}",
            "--stitched_R", f"{SR}",
            "--alpha", f"{alpha}",
            "-T", f"{T_b}", 
            "--label_file", f"/data/HybridANNS/data/Experiment/labelfilterData/labels/{dataset}/diskann_label_{basic_id}.txt"
        ], check=True)

    # Run single-threaded search for each L value
    for L in L_search:
        search_index(index_type, dataset, query_id, K, T_s, L, index_path, result_path)

    # Run multi-threaded search (16 threads)
    for L in L_search:
        search_index(index_type, dataset, query_id, K, T_m, L, index_path, result_path, num_threads=16)
# Loop through datasets and execute the commands
def main():
    # Step 1: Run GT generation shell script
    print("🟡 Running groundtruth generation script (run_gt.sh)...")
    # subprocess.run(["bash", "run_gt.sh"], check=True)

    # Step 2: Continue with indexing and searching
    for dataset in datasets:
        for basic_id, query_id in zip(basic_ids_e, query_ids_e):
            run_filtered(dataset, basic_id, query_id, index_type='filtered', R=R_e, L=L_e)
        for basic_id, query_id in zip(basic_ids_e, query_ids_e):
            run_stitched(dataset, basic_id, query_id, index_type='stitched', R=s_R_e, L=s_L_e, SR=s_SR_e)
        for basic_id, query_id in zip(basic_ids_c, query_ids_c):
            run_filtered(dataset, basic_id, query_id, index_type='filtered', R=R_c, L=L_c)
        for basic_id, query_id in zip(basic_ids_c, query_ids_c):
            run_stitched(dataset, basic_id, query_id, index_type='stitched', R=s_R_c, L=s_L_c, SR=s_SR_c)

if __name__ == "__main__":
    main()
    
