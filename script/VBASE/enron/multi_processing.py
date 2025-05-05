import logging
import multiprocessing as mp
import concurrent.futures
import struct
import time
import psycopg2
from psycopg2 import OperationalError
import random
import traceback
from typing import Tuple
from typing import List
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# 配置参数
DURATION = 15          # 单次并发测试持续时间
CONCURRENCIES = [16] # 并发度列表

def create_connection():
    """创建PostgreSQL连接"""
    try:
        conn = psycopg2.connect(
            database="vectordb",
            user="vectordb",
            host="172.17.0.2",  # depend on your own docker container ip
            port="5432"
        )
        return conn
    except OperationalError as e:
        log.error(f"Connection failed: {e}")
        return None

def pg_search_worker(
    test_data: list,
    conditions: list,
    columns: list,
    j_value: int,
    q: mp.Queue,
    cond: mp.Condition,
    duration: int = DURATION
) -> Tuple[int, float, list]:
    q.put(1)  # 准备就绪信号
    with cond:
        cond.wait()  # 等待同步
    
    conn = create_connection()
    if not conn:
        return (0, 0, [])
    
    cursor = conn.cursor()
    num = len(test_data)
    start_time = time.perf_counter()
    count = 0
    latencies = []
    
    try:
        while time.perf_counter() - start_time < duration:
            idx = random.randint(0, num-1)
            condition = conditions[idx]
            vector = test_data[idx]

            if len(condition) == 1:
                conditions_sql = f"{columns[0]} = {condition[0]}"
            else:
                # 使用 zip 将 columns 和 condition 的值一一对应
                conditions_part = [f"{col} = {val}" for col, val in zip(columns, condition)]
                conditions_sql = " AND ".join(conditions_part)
            
            query = f"""
                SELECT id
                FROM enron_label
                WHERE {conditions_sql}
                ORDER BY image_embedding <-> ARRAY[{', '.join(map(str, vector))}]
                LIMIT 10
            """
            # print(query)
            start = time.perf_counter()
            try:
                # 执行参数化查询
                cursor.execute(f"SET enable_indexscan = on; SET enable_seqscan = off; SET hnsw.ef_search = {j_value};")
                cursor.execute(query)
                results = cursor.fetchall()
                if results:
                    count += 1
            except Exception as e:
                log.error(f"Query failed: {e}")
            
            latencies.append(time.perf_counter() - start)
            
            if count % 100 == 0:
                log.debug(f"Process {mp.current_process().name} completed {count} queries")
                
    finally:
        cursor.close()
        conn.close()
    
    total_dur = time.perf_counter() - start_time
    return (count, total_dur, latencies)

def run_multiprocess_search(
    test_data: list,
    conditions: list,
    i_value: str,
    j_value: int,
    output_file: str,
    columns: list,
    concurrencies: List[int] = CONCURRENCIES,  # 使用 List[int] 代替 Iterable[int]
    duration: int = DURATION,
    
) -> Tuple[float, List[int], List[float], List[float], List[float]]:  # 使用 List[int] 和 List[float]
    
    with open(output_file, "a") as f:
        for conc in concurrencies:
            with mp.Manager() as manager:
                q = manager.Queue()
                cond = manager.Condition()

                with concurrent.futures.ProcessPoolExecutor(
                    mp_context=mp.get_context("spawn"),
                    max_workers=conc
                ) as executor:
                    log.info(f"Starting concurrency {conc} for {duration}s")
                    futures = [
                        executor.submit(
                            pg_search_worker,
                            test_data,
                            conditions,
                            columns,
                            j_value,
                            q,
                            cond,
                            duration
                        ) for _ in range(conc)
                    ]

                    # 等待所有worker就绪
                    while q.qsize() < conc:
                        time.sleep(0.1)

                    with cond:
                        cond.notify_all()
                        log.info("All workers started")

                    # 收集结果
                    results = [f.result() for f in futures]
                    total_count = sum(r[0] for r in results)
                    total_dur = max(r[1] for r in results)
                    all_latencies = [lat for r in results for lat in r[2]]

                    # 计算指标
                    qps = total_count / total_dur if total_dur > 0 else 0
                    f.write(
                        f"Recall Rate {i_value:<5} and {j_value:<4} : QPS: {float(qps):>8.2f}\n")
                    
def read_fvecs(file_path):
    """读取.fvecs文件"""
    vectors = []
    with open(file_path, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes: break
            dim = struct.unpack("i", dim_bytes)[0]
            vec = struct.unpack(f"{dim}f", f.read(dim*4))
            vectors.append(vec)
    return vectors

def read_conditions(file_path):
    """读取查询条件"""
    conditions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # 去除首尾空白字符
            if line:  # 跳过空行
                # 将每一行的数据分割成一个子列表
                elements = line.split()
                # 转换为整数并存储为子列表
                elements = [int(element) for element in elements]
                conditions.append(elements)
    return conditions
    
def main():
    # 数据准备
    ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "data/Experiment"))
    vectors = read_fvecs(os.path.join(ROOT_DIR, "labelfilterData/datasets/enron/enron_query.fvecs"))
    output_file = os.path.join(os.path.dirname(__file__), "result", f"mul_qps.out")
    
    list_1 = ["1", "3_1", "3_2", "3_3", "3_4", "4", "5_1", "5_2", "5_3", "5_4", "6", "7_1", "7_2", "7_3", "7_4"]
    list_2 = [["col_1"], ["col_16"], ["col_17"], ["col_18"], ["col_19"], ["col_8"], ["col_2"], ["col_3"], ["col_5"], ["col_1"], ["col_1", "col_9", "col_10"],["col_1", "col_9", "col_16"],["col_1", "col_9", "col_17"],["col_1", "col_9", "col_18"],["col_1", "col_9", "col_19"]]
    list_3 = [86, 150, 250, 400]

    for i, i_value in enumerate(list_1):
        for j_value in list_3:

            conditions = read_conditions(os.path.join(ROOT_DIR, f"labelfilterData/query_label/enron/{i_value}.txt"))
            columns = list_2[i]
            # 确保数据对齐
            min_len = min(len(vectors), len(conditions))
            vectors = vectors[:min_len]
            conditions = conditions[:min_len]
    
            try:
                run_multiprocess_search(vectors, conditions, i_value, j_value, output_file, columns)
            except Exception as e:
                log.error(f"执行失败: {str(e)}")
                traceback.print_exc()

if __name__ == "__main__":
    main()
