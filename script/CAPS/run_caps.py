import os
import subprocess


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/Experiment"))
CAPS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../algorithm/CAPS"))
TEMP_DIR = os.path.abspath(os.path.join(EXPERIMENT_ROOT, "../../data/Experiment/temp/CAPS/"))


datasets = ["audio", "sift", "gist", "glove-100", "msong", "enron"]


basic_ids_e = ["1", "3-1", "3-2", "3-3", "3-4", "4", "5-1", "5-2", "5-3", "5-4", "6"]
query_ids_e = ["1", "3_1", "3_2", "3_3", "3_4", "4", "5_1", "5_2", "5_3", "5_4", "6"]

basic_ids_c=["2-1","2-2"]
query_ids_c=["2_1","2_2"]

nprobe_values = [str(i) for i in range(500, 30500, 500)]


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

def run_commands_e(dataset):
    
    for basic_id, query_id in zip(basic_ids_e, query_ids_e):
        print(f"\n🔹 Running for dataset={dataset}, query_id={query_id}")

        base_path = os.path.join(EXPERIMENT_ROOT, "labelfilterData/datasets", dataset, f"{dataset}_base.fvecs")
        base_attr = os.path.join(EXPERIMENT_ROOT, "labelfilterData/labels", dataset, f"label_{basic_id}.txt")
        index_path = os.path.join(TEMP_DIR, "index", f"{dataset}_{query_id}")
        os.makedirs(index_path, exist_ok=True) 

        query_data = os.path.join(EXPERIMENT_ROOT, "labelfilterData/datasets", dataset, f"{dataset}_query.fvecs")
        query_attr = os.path.join(EXPERIMENT_ROOT, "labelfilterData/query_label", dataset, f"{query_id}.txt")
        gt_path = os.path.join(EXPERIMENT_ROOT, "labelfilterData/gt", dataset, f"gt-query_set_{query_id}.ivecs")
        result_path = os.path.join(TEMP_DIR, "result", f"{dataset}_{query_id}")
        os.makedirs(result_path, exist_ok=True) 
       
        subprocess.run([
            INDEX_EXEC, base_path, base_attr, index_path, "1024", "kmeans", "3"
        ], check=True)

        
        for nprobe in nprobe_values:
            subprocess.run([
                QUERY_EXEC, base_path, base_attr, query_data, query_attr,
                index_path, gt_path, "1024", "kmeans", "3", nprobe, "1", dataset, result_path
            ], check=True)

     
        for nprobe in nprobe_values:
            print(f"[🚀 Multi Thread] nprobe={nprobe}")
            subprocess.run([
                QUERY_EXEC, base_path, base_attr, query_data, query_attr,
                index_path, gt_path, "1024", "kmeans", "3", nprobe, "16", dataset, result_path
            ], check=True)
            
def run_commands_c(dataset): 
    
    for basic_id, query_id in zip(basic_ids_c, query_ids_c):
        print(f"\n🔹 Running for dataset={dataset}, query_id={query_id}")

        base_path = os.path.join(EXPERIMENT_ROOT, "labelfilterData/datasets", dataset, f"{dataset}_base.fvecs")
        base_attr = os.path.join(EXPERIMENT_ROOT, "labelfilterData/labels", dataset, f"label_{basic_id}.txt")
        index_path = os.path.join(TEMP_DIR, "index", f"{dataset}_{query_id}")
        os.makedirs(index_path, exist_ok=True)  

        query_data = os.path.join(EXPERIMENT_ROOT, "labelfilterData/datasets", dataset, f"{dataset}_query.fvecs")
        query_attr = os.path.join(EXPERIMENT_ROOT, "labelfilterData/query_label", dataset, f"caps_{query_id}.txt")
        gt_path = os.path.join(EXPERIMENT_ROOT, "labelfilterData/gt", dataset, f"gt-query_set_{query_id}.ivecs")
        result_path = os.path.join(TEMP_DIR, "result", f"{dataset}_{query_id}")
        os.makedirs(result_path, exist_ok=True)  
       
        subprocess.run([
            INDEX_EXEC, base_path, base_attr, index_path, "1024", "kmeans", "3"
        ], check=True)
        
        
      
        for nprobe in nprobe_values:
            subprocess.run([ 
                QUERY_EXEC, base_path, base_attr, query_data, query_attr,
                index_path, gt_path, "1024", "kmeans", "3", nprobe, "1", dataset, result_path
            ], check=True)

        
        for nprobe in nprobe_values:
            print(f"[🚀 Multi Thread] nprobe={nprobe}")
            subprocess.run([
                QUERY_EXEC, base_path, base_attr, query_data, query_attr,
                index_path, gt_path, "1024", "kmeans", "3", nprobe, "16", dataset, result_path
            ], check=True)

def main():
    build_caps_binaries()
    for dataset in datasets:
        run_commands_e(dataset)
        run_commands_c(dataset)

if __name__ == "__main__":
    main()