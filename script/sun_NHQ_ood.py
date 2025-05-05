#!/usr/bin/env python3
import os
import subprocess
import sys

def check_file(file_path, description):
    """Check if a file exists and print status"""
    if not os.path.isfile(file_path):
        print(f"❌ {description} not found: {file_path}")
        return False
    print(f"✅ Found: {file_path}")
    return True

def main():
    # ========= Set Parameters =========
    params = {
        "K": "100",
        "L": "100",
        "ITER": "12",
        "S": "10",
        "R": "300",
        "RANGE": "20",
        "PL": "350",
        "B": "0.4",
        "M": "1.0"
    }

    # ========= Set Base Paths =========
    BASE_DATA_PATH = "/data/HybridANNS/data"
    SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
    NHQ_BIN_PATH = os.path.join(SCRIPT_PATH, "../algorithm/NHQ-master")

    # ========= Set File Paths =========
    DATASET = "text2image"
    QUERY_KEY = "1"

    # Data paths
    BASE_VECS = os.path.join(BASE_DATA_PATH, "Experiment/labelfilterData/datasets", DATASET, f"{DATASET}_base.fvecs")
    QUERY_VECS = os.path.join(BASE_DATA_PATH, "Experiment/labelfilterData/datasets", DATASET, f"{DATASET}_query.fvecs")
    LABEL_FILE = os.path.join(BASE_DATA_PATH, "Experiment/labelfilterData/labels", DATASET, f"label_{QUERY_KEY}.txt")
    QUERY_LABEL_FILE = os.path.join(BASE_DATA_PATH, "Experiment/labelfilterData/query_label", DATASET, f"{QUERY_KEY}.txt")
    GROUNDTRUTH_FILE = os.path.join(BASE_DATA_PATH, "Experiment/labelfilterData/gt", DATASET, f"gt-query_set_{QUERY_KEY}.ivecs")

    # Output paths
    INDEX_DIR = os.path.join(BASE_DATA_PATH, "Experiment/Result/labelfilter/indexdata/NHQ", f"NHQ_{DATASET}_kgraph_{QUERY_KEY}")

    # ========= Check Files Exist =========
    print("🔍 Checking if required files exist...")
    
    # Check input files
    required_files = [
        (BASE_VECS, "Base vector file"),
        (LABEL_FILE, "Label file"),
        (QUERY_VECS, "Query vector file"),
        (QUERY_LABEL_FILE, "Query label file"),
        (GROUNDTRUTH_FILE, "Ground truth file")
    ]
    
    all_files_exist = True
    for file_path, description in required_files:
        if not check_file(file_path, description):
            all_files_exist = False
    
    # Check executables
    INDEX_BIN = os.path.join(NHQ_BIN_PATH, "NHQ-NPG_kgraph/build/tests/test_dng_index")
    SEARCH_BIN = os.path.join(NHQ_BIN_PATH, "NHQ-NPG_kgraph/build/tests/test_dng_optimized_search")
    
    if not check_file(INDEX_BIN, "Index binary"):
        all_files_exist = False
    if not check_file(SEARCH_BIN, "Search binary"):
        all_files_exist = False
        
    if not all_files_exist:
        print("❌ Some required files are missing. Exiting.")
        sys.exit(1)

    # ========= Create Index Directory =========
    print(f"📁 Creating index directory: {INDEX_DIR}")
    os.makedirs(INDEX_DIR, exist_ok=True)

    # ========= Build Index =========
    print(f"🔨 Building index for {DATASET}, query {QUERY_KEY}...")
    
    index_cmd = [
        "taskset", "-c", "0-31", 
        INDEX_BIN,
        BASE_VECS, 
        LABEL_FILE, 
        os.path.join(INDEX_DIR, "index"), 
        os.path.join(INDEX_DIR, "table"),
        params["K"], params["L"], params["ITER"], params["S"], params["R"],
        params["RANGE"], params["PL"], params["B"], params["M"]
    ]
    
    print(f"📦 Command: {' '.join(index_cmd)}")
    
    try:
        subprocess.run(index_cmd, check=True)
        print("✅ Index built successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Index building failed with error: {e}")
        sys.exit(1)

    # ========= Perform Search =========
    print(f"🔍 Running search for {DATASET}, query {QUERY_KEY}...")
    
    search_cmd = [
        SEARCH_BIN,
        os.path.join(INDEX_DIR, "index"),
        os.path.join(INDEX_DIR, "table"),
        BASE_VECS,
        QUERY_VECS,
        QUERY_LABEL_FILE,
        GROUNDTRUTH_FILE,
        DATASET
    ]
    
    print(f"📦 Command: {' '.join(search_cmd)}")
    
    try:
        subprocess.run(search_cmd, check=True)
        print("✅ Search completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Search failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()