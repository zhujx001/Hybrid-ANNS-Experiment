#include "HybridSearchExperiment.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cstdlib>
#include <filesystem>

// 获取项目根目录路径
std::string getProjectRoot() {
    // 检查当前目录名称或父目录名称
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path originalPath = currentPath;
    
    // 检查当前目录和向上最多5级父目录
    for (int i = 0; i < 10; ++i) {
        // 直接检查目录名称是否为"HybridANNS"
        if (currentPath.filename() == "HybridANNS") {
            return currentPath.string();
        }
        
        // 如果到达了文件系统根目录，停止查找
        if (currentPath.parent_path() == currentPath) {
            break;
        }
        
        // 向上一级目录
        currentPath = currentPath.parent_path();
    }
    
    // 如果以上方法都失败，返回一个默认值
    return "."; // 返回当前目录
}

// 在运行时打印找到的路径，便于调试
void printPaths() {
    std::string projectRoot = getProjectRoot();
    std::string dataDir = projectRoot + "/data/Experiment/rangefilterData/";
    std::string resultDir = projectRoot + "/data/Experiment/Result/rangefilter/";
    
    std::cout << "Project root: " << projectRoot << std::endl;
    std::cout << "Data directory: " << dataDir << std::endl;
    std::cout << "Result directory: " << resultDir << std::endl;
}

// 主要代码
std::string projectRoot = getProjectRoot();
std::string dataDir = projectRoot + "/data/Experiment/rangefilterData/";
std::string resultDir = projectRoot + "/data/Experiment/Result/rangefilter/";

// 程序开始时打印路径信息（可选）
struct PathPrinter {
    PathPrinter() {
        printPaths();
    }
} pathPrinter;


struct DatasetConfig {
    std::string base_file;
    std::string query_file;
    std::string prefix;
    std::string index_key;
    std::vector<int> nprobes;
    std::vector<int> thread_nums;
};

struct QuerySetConfig {
    std::string name;
    std::string suffix;
    std::string savefile;

};

std::pair<std::map<std::string, DatasetConfig>, std::map<std::string, QuerySetConfig>> get_query_config() {
    std::map<std::string, DatasetConfig> datasets = {

        {"deep", {
            "deep-base-1M.fvecs",
            "deep_query.fvecs",
            "deep-96-euclidean_queries",
            "IVF1000,Flat",
            {
            // 1-10: 密集采样，召回率快速增长区间
            1, 2, 3, 4, 5, 6, 8, 10,
            // 10-50: 中等间隔，召回率中等增长区间
            15, 20, 25, 30, 35, 40, 45,            
            // 50-100: 较大间隔，召回率缓慢增长区间
            60, 70, 80, 90, 100, 150, 200          
            },
            {16, 1}
        }},
        {"wit", {
            "wiki_image-base-1M.fvecs",
            "wiki_image-query.fvecs",
            "wit-2048-euclidean_queries",
            "IVF1000,Flat",
            {
            // 1-10: 密集采样，召回率快速增长区间
            1, 2, 3, 4, 5, 6, 8, 10,
            // 10-50: 中等间隔，召回率中等增长区间
            15, 20, 25, 30, 35, 40, 50,            
            // 50-100: 较大间隔，召回率缓慢增长区间
            60, 80,             
            // 100-200: 最大间隔，召回率极其缓慢增长区间
            120, 200, 300
            },
            {16, 1}
        }},
        {"yt8mAudio", {
            "yt8m_audio-base-1M.fvecs",
            "yt8m_audio-query.fvecs",
            "yt8mAudio-128-euclidean_queries",
            "IVF1000,Flat",
            {
            // 1-10: 密集采样，召回率快速增长区间
            1, 2, 3, 4, 5, 6, 8, 10,
            // 10-50: 中等间隔，召回率中等增长区间
            15, 20, 25, 30, 35, 40, 50,            
            // 50-100: 较大间隔，召回率缓慢增长区间
            60, 70, 80, 90,       
            // 100-200: 最大间隔，召回率极其缓慢增长区间
            120
            },
            {16, 1}
        }},
        {"text2image", {
            "text2image-base.fvecs",
            "text2image-query.fvecs",
            "text2image-200-euclidean_queries",
            "IVF3000,Flat",
            {
            //1-10: 密集采样，召回率快速增长区间
            1, 2, 3, 4, 5, 6, 8, 10,
            // 10-50: 中等间隔，召回率中等增长区间
            15, 20, 25, 30, 35, 40, 45, 50,            
            // 50-100: 较大间隔，召回率缓慢增长区间
            60, 70, 80, 90, 100,            
            // 100-200: 最大间隔，召回率极其缓慢增长区间
            120, 140, 160, 180, 200, 260,
            350, 450, 550, 650, 850, 1000, 
            1200, 1500, 1800
            },  
            {16, 1}
        }}
    };
    

   std::map<std::string, QuerySetConfig> query_sets = {
    {"1", {"2^-2", "_2pow-2", "range_2"}},
    // {"2", {"2^-4", "wit-2048-euclidean_queries_2pow-4"}},
    // {"3", {"2^-6", "wit-2048-euclidean_queries_2pow-6"}},
    {"4", {"2^-8", "_2pow-8", "range_8"}}
};
    
    return {datasets, query_sets};
}

void getIndex(bool is_save_to_file) {
    // 获取数据集和查询集配置
    auto [datasets, query_sets] = get_query_config();

    // 定义自定义的数据集遍历顺序
    std::vector<std::string> dataset_order;

    // 使用传入的线程数，不再硬编码
    // thread_nums = 16; 


    dataset_order = {"deep","yt8mAudio","wit", "text2image"};

    // dataset_order = {"text2image"};
    // query_sets = {
    //     {"1", {"2^-2", "_2pow-2", "range_2"}},
    // };

    // 按照指定顺序遍历
    for (const auto& dataset_name : dataset_order) {
        if (datasets.find(dataset_name) == datasets.end()) {
            continue;  // 跳过不存在的数据集
        }
        
        const auto& dataset_config = datasets[dataset_name];
        std::cout << "====================================================================================================================================================" << std::endl;
        std::cout << "Processing dataset: " << dataset_name << std::endl;
        std::stringstream datasetName;
        datasetName << dataset_name << std::endl;
        
        // 创建一个 HybridSearchExperiment 实例
        HybridSearchExperiment experiment;

        // 设置实验参数，使用命令行传入的参数
        experiment.setIsSaveToFile(is_save_to_file);

        // 设置数据集名称
        experiment.setDataset(dataset_name);
        
        // 设置向量文件路径
        experiment.setBaseFilePaths(
            dataDir + "datasets/" + dataset_name  + "/" + dataset_config.base_file
        );

        // 设置索引的类型和配置
        experiment.setIndexKey(dataset_config.index_key);

        experiment.setBulidIndexResultFile(resultDir + "indexdata/Faiss/" + dataset_name + ".csv");

        // 创建向量索引
        experiment.buildIndex();

    }
}

// 修改main函数，添加参数解析
int main(int argc, char *argv[]) {
    // 默认值
    bool is_save_to_file = false;

    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--save" || arg == "-s") {
            if (i + 1 < argc) {
                std::string value = argv[++i];
                is_save_to_file = (value == "true" || value == "1" || value == "yes");
            }
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "用法: " << argv[0] << " [选项]\n"
                      << "选项:\n"
                      << "  -t, --threads N    设置线程数为N (默认: 16)\n"
                      << "  -c, --cycles N     设置循环次数为N (默认: 1)\n"
                      << "  -s, --save yes/no  设置是否保存结果 (默认: 否)\n"
                      << "  -b, --batch yes/no 设置是否批量 (默认: 否)\n"
                      << "  -h, --help         显示此帮助信息\n";
            return 0;
        }
    }
    std::cout << "是否保存结果: " << (is_save_to_file ? "是" : "否") << std::endl;
    getIndex(is_save_to_file);
    return 0;
}