import logging
import multiprocessing as mp
import concurrent.futures
import numpy as np
import random
import time
import traceback
from typing import Iterable, Union, List, Tuple  # Use typing.Iterable instead of collections.abc.Iterable
from pymilvus import Collection
import os


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Milvus configuration (replace with your actual configuration)
MILVUS_HOST = "222.20.98.71"
MILVUS_PORT = "19530"
MILVUS_COLLECTION_NAME = "gist_label"
VECTOR_DIMENSION = 128  # Replace with your vector dimension

# Search parameters
K = 10  # Default k for search
DURATION = 10  # Seconds each concurrency runs
CONCURRENCIES = [16]  # List of concurrency levels to test


def connect_to_milvus():
    from pymilvus import connections
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)


def load_collection(collection_name):
    collection = Collection(collection_name)
    collection.load()
    return collection


def search_embedding(collection, query, top_k, filters, j_value):
    """Perform a search on a query embedding and return results."""
    search_params = {"metric_type": "L2", "params": {"nprobe": j_value}}  # Adjust based on your index type
    results = collection.search(
        data=[query],
        anns_field="image_embedding",
        param=search_params,
        limit=top_k,
        expr=filters,
    )
    return len(results[0])  # Return the number of results (for simplicity)


def search_worker(
    collection_name: str,
    test_data: List[List[float]],
    q: mp.Queue,
    cond: mp.Condition,
    k: int,
    conditions: List,
    columns: List,
    j_value: int,
) -> Tuple[int, float, List[float]]:
    q.put(1)  # Signal that the process is ready
    with cond:
        cond.wait()  # Wait for the main process to notify

    connect_to_milvus()
    collection = load_collection(collection_name)

    num, idx = len(test_data), random.randint(0, len(test_data) - 1)
    start_time = time.perf_counter()
    count = 0
    latencies = []

    while time.perf_counter() - start_time < DURATION:
        s = time.perf_counter()
        try:
            condition = conditions[idx]
            vector = test_data[idx]
            if len(condition) == 1:
                conditions_sql = f"{columns[0]} == {condition[0]}"
            else:
                conditions_sql = " && ".join([f"{col} == {condition[i]}" for i, col in enumerate(columns[:len(condition)])])
            # Search in Milvus
            results_count = search_embedding(collection, test_data[idx], k, conditions_sql, j_value)
            if results_count > 0:
                count += 1
        except Exception as e:
            log.warning(f"Milvus search error: {e}")
            traceback.print_exc()

        latencies.append(time.perf_counter() - s)
        idx = idx + 1 if idx < num - 1 else 0

        if count % 500 == 0:
            log.debug(f"Process {mp.current_process().name} search_count: {count}, latest_latency={latencies[-1]}")

    total_dur = round(time.perf_counter() - start_time, 4)
    return (count, total_dur, latencies)


def run_multiprocess_search(
    test_data: List[List[float]],
    i_value: str,
    j_value: int,
    output_file: str,
    columns: List,
    collection_name: str,
    k: int = K,
    conditions: Union[List, None] = None,
    concurrencies: Iterable[int] = CONCURRENCIES,  # Changed to typing.Iterable[int]
    duration: int = DURATION,
) -> Tuple[float, List[int], List[float], List[float], List[float]]:
    with open(output_file, "a") as f:
        for conc in concurrencies:
            with mp.Manager() as manager:
                q = manager.Queue()
                cond = manager.Condition()

                with concurrent.futures.ProcessPoolExecutor(
                    mp_context=mp.get_context("spawn"),
                    max_workers=conc,
                ) as executor:
                    log.info(f"Start search {duration}s with concurrency {conc}")
                    future_iter = [
                        executor.submit(
                            search_worker,
                            collection_name,
                            test_data,
                            q,
                            cond,
                            k,
                            conditions,
                            columns,
                            j_value,
                        )
                        for _ in range(conc)
                    ]

                    # Synchronize all processes
                    while q.qsize() < conc:
                        sleep_time = conc if conc < 10 else 10
                        time.sleep(sleep_time)

                    with cond:
                        cond.notify_all()
                        log.info(f"Syncing all processes and starting concurrency search, concurrency={conc}")

                    start = time.perf_counter()
                    results = [future.result() for future in future_iter]
                    total_count = sum(result[0] for result in results)
                    total_dur = max(r[1] for r in results)
                    latencies = sum((result[2] for result in results), [])
                    total_cost = time.perf_counter() - start

                    # Calculate metrics
                    qps = total_count / total_dur if total_dur > 0 else 0
                    f.write(
                        f"Recall Rate {i_value:<5} and {j_value:<4} : QPS: {float(qps):>8.2f}\n"
                    )
    # Return dummy values since the original function doesn't return anything meaningful yet
    return (0.0, [], [], [], [])


def read_fvecs(file_path: str, num_vectors: int) -> List[List[float]]:
    data = []
    with open(file_path, 'rb') as f:
        for _ in range(num_vectors):
            dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
            vector = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            data.append(vector.tolist())
    return data


def read_conditions(file_path: str) -> List[List[int]]:
    """Read query conditions."""
    conditions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:  # Skip empty lines
                elements = line.split()
                elements = [int(element) for element in elements]
                conditions.append(elements)
    return conditions


def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "data/Experiment"))
    # Test data (replace with your actual test dataset)
    test_data = read_fvecs(os.path.join(ROOT_DIR, "labelfilterData/datasets/gist/gist_query.fvecs"), 100)
    output_file = os.path.join(os.path.dirname(__file__), "result", f"mul_qps.out")

    list_1 = ["1", "3_1", "3_2", "3_3", "3_4", "4", "5_1", "5_2", "5_3", "5_4", "6", "7_1", "7_2", "7_3", "7_4"]
    list_2 = [["col_1"], ["col_16"], ["col_17"], ["col_18"], ["col_19"], ["col_8"], ["col_2"], ["col_3"], ["col_5"], ["col_1"],
              ["col_1", "col_9", "col_10"], ["col_1", "col_9", "col_16"], ["col_1", "col_9", "col_17"],
              ["col_1", "col_9", "col_18"], ["col_1", "col_9", "col_19"]]
    list_3 = [10,15,20,30,40,50,100,150]

    for i, i_value in enumerate(list_1):
        for j_value in list_3:
            conditions = read_conditions(os.path.join(ROOT_DIR, f"labelfilterData/query_label/gist/{i_value}.txt"))
            columns = list_2[i]
            # Ensure data alignment
            vectors = test_data  # Renamed for clarity
            min_len = min(len(vectors), len(conditions))
            vectors = vectors[:min_len]
            conditions = conditions[:min_len]
            try:
                max_qps, conc_num_list, conc_qps_list, conc_latency_p99_list, conc_latency_avg_list = run_multiprocess_search(
                    vectors, i_value, j_value, output_file, columns, MILVUS_COLLECTION_NAME, K, conditions, CONCURRENCIES, DURATION
                )
            except Exception as e:
                log.error(f"Error during multi-process search: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()