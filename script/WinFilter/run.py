#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse

# Base paths with relative structure
DATA_BASE_PATH = "../../data/Experiment/rangefilterData"
RESULTS_BASE_PATH = "../../data/Experiment/Result/WinFilter"
CODE_PATH = "../../algorithm/RangeFilteredANN/experiments/run_our_method.py"

# Create output directories
os.makedirs(RESULTS_BASE_PATH, exist_ok=True)
os.makedirs(f"{RESULTS_BASE_PATH}/build_times", exist_ok=True)
os.makedirs(f"{RESULTS_BASE_PATH}/results", exist_ok=True)
os.makedirs(f"{RESULTS_BASE_PATH}/index_cache", exist_ok=True)

# Dataset configurations with dimensions
DATASET_CONFIGS = {
    "deep": "96-euclidean",
    "wit": "2048-euclidean",
    "text2image": "200-euclidean",
    "yt8mAudio": "128-euclidean"
}

# Get available filter widths for each dataset
def get_available_filter_widths(dataset):
    """Return list of available filter widths for a dataset based on existing files."""
    base_dir = f"{DATA_BASE_PATH}/gt/{dataset}"
    widths = []
    
    # Common filter widths to check
    common_widths = ["2pow-2", "2pow-4", "2pow-6", "2pow-8"]
    
    for width in common_widths:
        gt_file = f"{base_dir}/{dataset}-{DATASET_CONFIGS[dataset]}_queries_{width}_gt.npy"
        range_file = f"{DATA_BASE_PATH}/query_range/{dataset}/{dataset}-{DATASET_CONFIGS[dataset]}_queries_{width}_ranges.npy"
        if os.path.exists(gt_file) and os.path.exists(range_file):
            widths.append(width)
    
    return widths

def run_experiment(dataset, filter_width):
    """Run range filtering experiment for a specific dataset and filter width."""
    print(f"\n{'='*80}")
    print(f"Running experiment for dataset: {dataset}, filter width: {filter_width}")
    print(f"{'='*80}\n")
    
    dataset_dim = DATASET_CONFIGS[dataset]
    
    # Construct file paths
    dataset_file = f"{DATA_BASE_PATH}/datasets/{dataset}/{dataset}-{dataset_dim}.npy"
    queries_file = f"{DATA_BASE_PATH}/datasets/{dataset}/{dataset}-{dataset_dim}_queries.npy"
    filter_values_file = f"{DATA_BASE_PATH}/labels/{dataset}/{dataset}-{dataset_dim}_filter-values.npy"
    ranges_file_template = f"{DATA_BASE_PATH}/query_range/{dataset}/{dataset}-{dataset_dim}_queries_{{filter_width}}_ranges.npy"
    gt_file_template = f"{DATA_BASE_PATH}/gt/{dataset}/{dataset}-{dataset_dim}_queries_{{filter_width}}_gt.npy"
    
    # Verify files exist
    for file in [dataset_file, queries_file, filter_values_file]:
        if not os.path.exists(file):
            print(f"ERROR: File not found: {file}")
            return False
    
    # Dataset-specific results paths
    results_dir = f"{RESULTS_BASE_PATH}/results/{dataset}"
    build_times_dir = f"{RESULTS_BASE_PATH}/build_times"
    index_cache_dir = f"{RESULTS_BASE_PATH}/index_cache/{dataset}"
    
    # Create directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(index_cache_dir, exist_ok=True)
    os.makedirs(f"{index_cache_dir}-super_opt_postfiltering", exist_ok=True)
    
    # Construct command
    cmd = [
        "python3",
        CODE_PATH,
        f"--data_path={DATA_BASE_PATH}",
        f"--index_cache_path={RESULTS_BASE_PATH}/index_cache",
        f"--results_path={results_dir}",
        f"--build_times_path={build_times_dir}",
        f"--dataset={dataset}",
        f"--dataset_file={dataset_file}",
        f"--queries_file={queries_file}",
        f"--filter_values_file={filter_values_file}",
        f"--gt_file_template={gt_file_template}",
        f"--ranges_file_template={ranges_file_template}",
        f"--metric=Euclidian",
        f"--build_threads=32",
        f"--search_threads=1",
        f"--num_queries=10000",
        f"--experiment_filter_width={filter_width}",
        "--super_opt_postfiltering",  # Run all filtering methods
        f"--results_file_prefix={filter_width}_"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute the command
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return False

def main():
    """Run experiments for all datasets with their available filter widths."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory to ensure relative paths work
    os.chdir(script_dir)
    
    # Update paths to be relative to the script location
    global DATA_BASE_PATH, RESULTS_BASE_PATH, CODE_PATH
    DATA_BASE_PATH = os.path.abspath(DATA_BASE_PATH)
    RESULTS_BASE_PATH = os.path.abspath(RESULTS_BASE_PATH)
    CODE_PATH = os.path.abspath(CODE_PATH)
    
    print(f"Using data path: {DATA_BASE_PATH}")
    print(f"Results will be saved to: {RESULTS_BASE_PATH}")
    print(f"Using code at: {CODE_PATH}")
    
    parser = argparse.ArgumentParser(description="Run WinFilter experiments on multiple datasets")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()),
                        help="Specific datasets to run (default: all)")
    args = parser.parse_args()
    
    successful = []
    failed = []
    
    for dataset in args.datasets:
        if dataset not in DATASET_CONFIGS:
            print(f"Warning: Unknown dataset '{dataset}', skipping")
            continue
            
        filter_widths = get_available_filter_widths(dataset)
        
        if not filter_widths:
            print(f"Warning: No valid filter widths found for dataset '{dataset}', skipping")
            continue
            
        print(f"Found filter widths for {dataset}: {filter_widths}")
        
        for filter_width in filter_widths:
            if run_experiment(dataset, filter_width):
                successful.append((dataset, filter_width))
            else:
                failed.append((dataset, filter_width))
    
    # Print summary
    print("\n\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"\nSuccessful experiments ({len(successful)}):")
    for dataset, filter_width in successful:
        print(f"  - {dataset} (filter width: {filter_width})")
        print(f"    Results: {RESULTS_BASE_PATH}/results/{dataset}/{filter_width}_{dataset}_results.csv")
    
    if failed:
        print(f"\nFailed experiments ({len(failed)}):")
        for dataset, filter_width in failed:
            print(f"  - {dataset} (filter width: {filter_width})")
    
    print("\nResults locations:")
    print(f"  - CSV Results: {RESULTS_BASE_PATH}/results/<dataset>/<filter_width>_<dataset>_results.csv")
    print(f"  - Build Times: {RESULTS_BASE_PATH}/build_times/<filter_width>_<dataset>_build_times.csv")
    print(f"  - Index Cache: {RESULTS_BASE_PATH}/index_cache/<dataset>/")

if __name__ == "__main__":
    main()