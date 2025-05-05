#include <omp.h>
#include <unistd.h>
#include <immintrin.h> // AVX指令集
#include <sys/stat.h>
#include <sys/types.h>
#include <memory>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include <iomanip> 
#include <cstdlib>

#include <faiss/IVFlib.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>
#include <faiss/impl/FaissAssert.h>

#include <sys/resource.h>

#include "FileReader.h"
#include "DataStructures.h"

using idx_t = faiss::idx_t;

class HybridSearchExperiment {
public:
    std::string dataset;
    std::string vectorFileBase;
    std::string labelFileQuery;
    std::string vectorFileQuery;
    std::string groundTruthFile;
    std::string buildIndexResultFile;
    std::string resultFile;
    size_t vectorDim;
    size_t k;
    std::vector<int> thread_nums;
    std::vector<int> nprobe_values;

    std::string index_key;
    std::unique_ptr<faiss::Index> index;
    std::string index_path;

    size_t cycleNum;
    bool isBatch;
    bool isSaveToFile;

    

    HybridSearchExperiment()
        : k(10), 
          cycleNum(3), 
          isSaveToFile(true), 
          isBatch(false), 
          vectorDim(0), 
          index_key("IVF1000,Flat"),
          index_path("../../index/") {}
    void setBaseFilePaths(const std::string& vectorBase) {
        vectorFileBase = vectorBase;  
    }

    void setBulidIndexResultFile(const std::string& buildIndexResult){
        buildIndexResultFile = buildIndexResult;
    }

    void setQueryFilePaths(const std::string& labelQuery, const std::string& vectorQuery, 
                          const std::string& groundTruth, const std::string& result) {
        labelFileQuery = labelQuery;
        vectorFileQuery = vectorQuery;
        groundTruthFile = groundTruth;
        resultFile = result;
    }


    void setDataset(const std::string& _dataset) {
        dataset = _dataset;
    }

    void setIndexKey(const std::string& key) {
        index_key = key;
    }

    void setNprobeValues(const std::vector<int>& values) {
        nprobe_values = values;
    }

    void setThreadNums(const std::vector<int>& values) {
        thread_nums = values;
    }

    void setk(const size_t& _k) {
        k = _k;
    }

    void setIndexPath(const std::string& _index_path) {
        index_path = _index_path;
    }

    void setCycleNum(const size_t& _cycleNum) {
        cycleNum = _cycleNum;
    }

    void setIsBatch(const bool& _isBatch) {
        isBatch = _isBatch;
    }

    void setIsSaveToFile(const bool& _isSaveToFile) {
        isSaveToFile = _isSaveToFile;
    }

    void buildIndex() {
        // 索引文件路径
        std::string stored_name = index_path + dataset + "_" + index_key + ".faissindex";
    
        // 如果索引文件不存在，创建并保存
        
        if (access(stored_name.c_str(), F_OK) != 0) {

            int thread_num = omp_get_max_threads();
            omp_set_num_threads(thread_num);

            auto startIndexTime = std::chrono::high_resolution_clock::now();
            // 读取数据点
            std::vector<float> data = FileReader::readIvecs(vectorFileBase, vectorDim);
            float* xb = data.data();
            size_t nb = data.size() / vectorDim;
            std::cout << "Creating index..." << std::endl;
            index.reset(faiss::index_factory(vectorDim, index_key.c_str()));
            index->train(nb, xb);
            index->add(nb, xb);
            std::cout << "Index created." << std::endl;
    
            // 计算索引构建时间
            auto endIndexTime = std::chrono::high_resolution_clock::now();
            double indexBuildTime = std::chrono::duration_cast<std::chrono::seconds>(endIndexTime - startIndexTime).count();
    
            // 保存索引
            faiss::write_index(index.get(), stored_name.c_str());
    
            // 获取索引大小
            float indexSize;
            getIndexSize(indexSize, stored_name);
    
            // 获取索引构建时的内存占用
            float indexActualMem, indexVirtualMem;
            getMemoryUsage(indexActualMem, indexVirtualMem);
    
            // 保存索引构建结果
            std::ostringstream indexResult;
            indexResult << "query_set,memory_mb,build_time,index_size_mb" << std::endl;
            indexResult  << "1," << indexActualMem << "," << indexBuildTime << "," << indexSize << "," << std::endl;
            
            saveResultsToFile(indexResult.str(), false, false); // 仅第一次保存

            std::cout << "Thread Num: " << thread_num << std::endl;
            indexResult << "Index Key: " << index_key << std::endl;
            std::cout << "Index Build Time: " << indexBuildTime << " s" << std::endl;
            std::cout << "Index size: " << indexSize << " MB" << std::endl;
            std::cout << "Memory (VIRT/RES): " << indexActualMem << " MB / " << indexVirtualMem << " MB" << std::endl;
        } else {
            std::cout << "Loading index from file: " << stored_name << std::endl;
            index.reset(faiss::read_index(stored_name.c_str()));
        }
    }

    void run() {

        // 读取数据点、查询集以及 ground truth
        std::vector<Query> queries = FileReader::readQueries(labelFileQuery, vectorFileQuery, vectorDim);
        std::vector<std::vector<int>> groundTruth = FileReader::readGroundTruth_txt(groundTruthFile);
        
        // 读取向量总数
        size_t nb = index->ntotal;

        // 设置索引参数
        faiss::IndexIVF* index_ivf = static_cast<faiss::IndexIVF*>(index.get());
        index_ivf->verbose = true;
        
    
        // 进行搜索和计算查询时间
        for (int thread_num : thread_nums) {
            
            // 保存线程数 和 csv 表头信息
            std::ostringstream threadNumAndCsvHeader;
            std::cout << "Thread Num: " << thread_num << std::endl;
            std::cout << std::left; // 设置左对齐
            std::cout << std::setw(8) << "nprobe" 
                    << std::setw(16) << "Query Time(ms)" 
                    << std::setw(10) << "QPS" 
                    << std::setw(11) << "Recall@10" 
                    << std::setw(9) << "RES(MB)"
                    << std::setw(10) << "VIRT(MB)"  
                    << std::setw(23) << "total_filter_time(ms)" 
                    << std::setw(21) << "avg_filter_time(ms)" 
                    << std::setw(23) << "total_search_time(ms)" 
                    << std::setw(21) << "avg_search_time(ms)" << std::endl;

            //threadNumAndCsvHeader << "Thread Num: " << thread_num << std::endl;
            threadNumAndCsvHeader << "nprobe,Query Time(ms),QPS,Recall@" << k 
                << ",RES(MB),VIRT(MB),total_filter_time(ms),avg_filter_time(ms),total_search_time(ms),avg_search_time(ms)" << std::endl;
            saveResultsToFile(threadNumAndCsvHeader.str(), true, false);
                
            
            
    
            for (int nprobe : nprobe_values) {
                index_ivf->nprobe = nprobe;  // 设置 nprobe
                
                std::vector<std::vector<idx_t>> results(queries.size());
                double best_total_filter_time = 0;
                double best_total_search_time = 0;
                double best_queryTime = std::numeric_limits<double>::max();
                float best_qps = 0.0f;
                float recall = 0.0f;
                float best_searchActualMem, best_searchVirtualMem;

                // 运行 cycleNum 次测试，取 qps 最低的一次的结果
                for(int i = 0 ; i < cycleNum; i++){
                    
                    // 执行搜索
                    double queryTime, total_filter_time, total_search_time;
                    std::vector<std::vector<idx_t>> searchResults;

                    if(isBatch){
                        index_ivf->parallel_mode = 3;
                        omp_set_num_threads(thread_num);
                        std::tie(queryTime, total_filter_time, total_search_time, searchResults) = 
                            executeGroupedSearch(queries, nb, nprobe, k);
                    }
                    else{
                        omp_set_num_threads(thread_num);
                        std::tie(queryTime, total_filter_time, total_search_time, searchResults) = 
                            executeSearch(queries, nb, nprobe, thread_num, k);
                    
                    
        
                        float qps = queries.size() / (queryTime / 1000000.0);
            
                        // 计算搜索时的内存占用
                        float searchActualMem, searchVirtualMem;
                        getMemoryUsage(searchActualMem, searchVirtualMem);

                        if(queryTime < best_queryTime){
                            best_queryTime = queryTime;
                            best_qps = qps;
                            best_searchActualMem = searchActualMem;
                            best_searchVirtualMem = searchVirtualMem;
                            best_total_filter_time = total_filter_time;
                            best_total_search_time = total_search_time;
                            results = std::move(searchResults);
                        }
                    }
                }
    
                // 计算 Recall
                recall = computeRecall(results, groundTruth, k);
    
                // 保存搜索过程结果
                std::ostringstream searchResult;
                double avg_filter_time = best_total_filter_time / (queries.size() * 1000.0); // 转换为毫秒
                double avg_search_time = best_total_search_time / (queries.size() * 1000.0); // 转换为毫秒

                std::cout << std::setw(8) << nprobe 
                            << std::setw(16) << best_queryTime / 1000.0
                            << std::setw(10) << best_qps 
                            << std::setw(11) << recall 
                            << std::setw(9) << best_searchActualMem 
                            << std::setw(10) << best_searchVirtualMem
                            << std::setw(23) << (best_total_filter_time / 1000.0) 
                            << std::setw(21) << avg_filter_time
                            << std::setw(23) << (best_total_search_time / 1000.0) 
                            << std::setw(21) << avg_search_time << std::endl;

                searchResult << nprobe << "," << best_queryTime / 1000.0 << "," << best_qps << "," << recall 
                             << "," << best_searchActualMem 
                             << "," << best_searchVirtualMem 
                             << "," << best_total_filter_time / 1000.0 << "," << avg_filter_time 
                             << "," << best_total_search_time / 1000.0 << "," << avg_search_time << std::endl;
                saveResultsToFile(searchResult.str());
            }
            //std::cout << "--------------------------------------" << std::endl;
        }
    }
    

private:

    // 生成位图
    // 类成员变量
    static constexpr uint8_t BIT_MASKS[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};

    std::vector<uint8_t> generateBitmap(const Query& query) {
        const size_t nb = index->ntotal;
        const size_t imin = query.imin;
        const size_t imax = query.imax;
        
        std::vector<uint8_t> bitmap((nb + 7) / 8, 0);
        
        // 计算完整字节范围
        size_t start_byte = (imin + 7) / 8;
        size_t end_byte = imax / 8;
        
        // 1. 处理起始的不完整字节
        if (imin / 8 == start_byte - 1) {
            uint8_t mask = 0xFF << (imin % 8);
            bitmap[imin / 8] = mask;
        }
        
        // 2. 设置中间的完整字节
        if (start_byte <= end_byte) {
            std::memset(&bitmap[start_byte], 0xFF, end_byte - start_byte + 1);
        }
        
        // 3. 处理结束的不完整字节
        if (imax / 8 == end_byte && imax % 8 != 7) {
            uint8_t mask = (1 << ((imax % 8) + 1)) - 1;
            bitmap[end_byte] = (bitmap[end_byte] & ~mask) | mask;
        }
        
        return bitmap;
    }

    float computeRecall(const std::vector<std::vector<idx_t>>& results,
                        const std::vector<std::vector<int>>& groundTruth, int k) {
        int correct = 0;
        int total = 0;

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            const auto& truth = groundTruth[i];

            std::unordered_set<int> truthSet;
            for (int j = 0; j < k && truth[j] != -1; ++j) {
                truthSet.insert(truth[j]);
            }

            for (const int& id : result) {
                if (id != -1 && truthSet.count(id)) {
                    correct++;
                }
            }
            total += truthSet.size();
        }
        return total == 0 ? 0.0f : static_cast<float>(correct) / total;
    }

    void getMemoryUsage(float& actualMem, float& virtualMem) {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.find("VmRSS:") == 0) { // 实际内存
                std::istringstream iss(line.substr(7)); // 跳过 "VmRSS:"
                float rss_kb;
                std::string unit;
                iss >> rss_kb >> unit;
                actualMem = rss_kb / 1024.0; // KB to MB
            } else if (line.find("VmSize:") == 0) { // 虚拟内存
                std::istringstream iss(line.substr(7)); // 跳过 "VmSize:"
                float size_kb;
                std::string unit;
                iss >> size_kb >> unit;
                virtualMem = size_kb / 1024.0; // KB to MB
            }
        }
    }

    void getIndexSize(float& indexFileSize, const std::string& indexFilePath) {
        std::ifstream file(indexFilePath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "无法打开索引文件: " << indexFilePath << std::endl;
            return;
        }
        indexFileSize = file.tellg() / (1024.0 * 1024.0);  // 获取文件的大小并转换为 MB
        file.close();
    }
    
    void saveResultsToFile(const std::string& result, bool search = true, bool append = true) {
        if(!isSaveToFile){
            return;
        }
        const std::string& fileName = search ? resultFile : buildIndexResultFile;

        // 确保目录存在
        ensureDirectoryExists(fileName);

        std::ofstream outFile(fileName, append ? std::ios::app : std::ios::out);
        if (!outFile.is_open()) {
            std::cerr << "无法打开文件: " << resultFile << std::endl;
            return;
        }
        outFile << result;
        outFile.close();
    }

    void ensureDirectoryExists(const std::string& filePath) {
        // 提取路径部分
        size_t pos = filePath.find_last_of('/');
        if (pos != std::string::npos) {
            std::string dirPath = filePath.substr(0, pos);
            
            // 检查目录是否存在
            struct stat st;
            bool dirExists = (stat(dirPath.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
            
            if (!dirExists) {
                // 目录不存在，创建它
                std::string command = "mkdir -p " + dirPath;
                std::cout << "创建目录: " << dirPath << std::endl; 
                
                // 执行命令
                int status = system(command.c_str());
                if (status != 0) {
                    std::cerr << "警告: 无法创建目录: " << dirPath << std::endl;
                }
            }
        }
    }
    
    // 执行并行搜索操作并返回性能指标
    std::tuple<double, double, double, std::vector<std::vector<idx_t>>> 
    executeSearch(std::vector<Query>& queries, 
                  size_t nb, int nprobe, int thread_num, size_t k) {
        
        std::vector<std::vector<idx_t>> results(queries.size());
        double total_filter_time = 0;
        double total_search_time = 0;
        
        // 记录查询开始时间
        auto startQueryTime = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for if (thread_num > 1) reduction(+:total_filter_time,total_search_time)
        for (size_t qIndex = 0; qIndex < queries.size(); ++qIndex) {
            Query& query = queries[qIndex];
            float* xq = query.queryVector.data();

            std::vector<idx_t> I(k);
            std::vector<float> D(k);

            // 计时 generateBitmap
            auto start_bitmap = std::chrono::high_resolution_clock::now();
            std::vector<uint8_t> bitmap = generateBitmap(query);
            auto end_bitmap = std::chrono::high_resolution_clock::now();
            total_filter_time += std::chrono::duration_cast<std::chrono::microseconds>(
                end_bitmap - start_bitmap).count();

            faiss::IVFSearchParameters params;
            faiss::IDSelectorBitmap sel((nb + 7) / 8, bitmap.data());
            params.sel = &sel;
            params.nprobe = nprobe;

            // 计时 search_with_parameters
            auto start_search = std::chrono::high_resolution_clock::now();
            faiss::ivflib::search_with_parameters(
                index.get(), 1, xq, k, D.data(), I.data(), &params);
            auto end_search = std::chrono::high_resolution_clock::now();
            total_search_time += std::chrono::duration_cast<std::chrono::microseconds>(
                end_search - start_search).count();

            results[qIndex] = std::move(I);
        }

        // 计算查询时间
        auto endQueryTime = std::chrono::high_resolution_clock::now();
        double queryTime = std::chrono::duration_cast<std::chrono::microseconds>(endQueryTime - startQueryTime).count();
        
        return {queryTime, total_filter_time, total_search_time, results};
    }

    // 执行基于分组的搜索操作并返回性能指标
    std::tuple<double, double, double, std::vector<std::vector<idx_t>>> 
    executeGroupedSearch(std::vector<Query>& queries, 
                        size_t nb, int nprobe, size_t k) {
        
        std::vector<std::vector<idx_t>> results(queries.size());
        double total_filter_time = 0;
        double total_search_time = 0;
        
        // 记录查询开始时间
        auto startQueryTime = std::chrono::high_resolution_clock::now();

        // 查询分组
        std::unordered_map<std::string, std::vector<size_t>> groupedQueries;
        for (size_t i = 0; i < queries.size(); ++i) {
            std::string key = queries[i].generateGroupKey();
            groupedQueries[key].emplace_back(i);
        }
        
        // 转换为 vector 以支持并行处理
        std::vector<std::pair<std::string, std::vector<size_t>>> groupedQueriesVector;
        groupedQueriesVector.reserve(groupedQueries.size());
        for (auto &group : groupedQueries) {
            groupedQueriesVector.push_back(std::move(group));
        }

        for (auto it = groupedQueriesVector.begin(); it != groupedQueriesVector.end(); ++it) {
            const std::vector<size_t> &queryGroup = it->second;
            size_t nq = queryGroup.size();
            std::vector<float> xq(nq * vectorDim);
            
            for (size_t i = 0; i < nq; i++) {
                std::copy(queries[queryGroup[i]].queryVector.begin(),
                        queries[queryGroup[i]].queryVector.end(),
                        xq.begin() + i * vectorDim);
            }

            std::vector<idx_t> I(k * nq);
            std::vector<float> D(k * nq);

            // 计时 generateBitmap
            auto start_bitmap = std::chrono::high_resolution_clock::now();
            std::vector<uint8_t> bitmap = generateBitmap(queries[queryGroup[0]]);
            auto end_bitmap = std::chrono::high_resolution_clock::now();
            total_filter_time += std::chrono::duration_cast<std::chrono::microseconds>(
                end_bitmap - start_bitmap).count();

            faiss::IVFSearchParameters params;
            faiss::IDSelectorBitmap sel((nb + 7) / 8, bitmap.data());
            params.sel = &sel;
            params.nprobe = nprobe;

            // 计时 search_with_parameters
            auto start_search = std::chrono::high_resolution_clock::now();
            faiss::ivflib::search_with_parameters(
                index.get(), nq, xq.data(), k, D.data(), I.data(), &params);
            auto end_search = std::chrono::high_resolution_clock::now();
            total_search_time += std::chrono::duration_cast<std::chrono::microseconds>(
                end_search - start_search).count();

            for (size_t i = 0; i < nq; ++i) {
                size_t queryIndex = queryGroup[i];
                results[queryIndex].assign(I.begin() + i * k, I.begin() + (i + 1) * k);
            }
        }

        // 计算查询时间
        auto endQueryTime = std::chrono::high_resolution_clock::now();
        double queryTime = std::chrono::duration_cast<std::chrono::microseconds>(endQueryTime - startQueryTime).count();
        
        return {queryTime, total_filter_time, total_search_time, results};
    }
    
};
