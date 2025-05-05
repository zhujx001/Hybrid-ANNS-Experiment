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
    std::string dataDir = projectRoot + "/data/Experiment/labelfilterData/";
    std::string resultDir = projectRoot + "/data/Experiment/Result/labelfilter/";
    
    std::cout << "Project root: " << projectRoot << std::endl;
    std::cout << "Data directory: " << dataDir << std::endl;
    std::cout << "Result directory: " << resultDir << std::endl;
}

// 主要代码
std::string projectRoot = getProjectRoot();
std::string dataDir = projectRoot + "/data/Experiment/labelfilterData/";
std::string resultDir = projectRoot + "/data/Experiment/Result/labelfilter/";

// 程序开始时打印路径信息（可选）
struct PathPrinter {
    PathPrinter() {
        printPaths();
    }
} pathPrinter;

struct DatasetConfig {
    std::string base_file;
    std::string query_file;
    std::string index_key;
    std::vector<int> nprobes;
    std::vector<int> thread_nums;
};

struct QuerySetConfig {
    std::string name;
    std::vector<int> attrs;
    std::string suffix;
};

std::pair<std::map<std::string, DatasetConfig>, std::map<std::string, QuerySetConfig>> get_query_config() {
    std::map<std::string, DatasetConfig> datasets = {

        {"sift", {
            "sift_base.fvecs",
            "sift_query.fvecs",
            "IVF1000,Flat",
            {
            // 1-10: 密集采样，召回率快速增长区间
            1, 2, 3, 4, 5, 6, 8, 10,
            // 10-50: 中等间隔，召回率中等增长区间
            15, 20, 25, 30, 35, 40, 45, 50,            
            // 50-100: 较大间隔，召回率缓慢增长区间
            60, 70, 80, 90, 100,            
            // 100-200: 最大间隔，召回率极其缓慢增长区间
            120, 140, 160, 180, 200 
            },
            {16, 1}
        }},
        {"gist", {
            "gist_base.fvecs",
            "gist_query.fvecs",
            "IVF1000,Flat",
            {
            // 1-10: 密集采样，召回率快速增长区间
            1, 2, 3, 4, 5, 6, 8, 10,
            // 10-50: 中等间隔，召回率中等增长区间
            15, 20, 25, 30, 35, 40, 45, 50,            
            // 50-100: 较大间隔，召回率缓慢增长区间
            60, 70, 80, 90, 100,            
            // 100-200: 最大间隔，召回率极其缓慢增长区间
            120, 140, 160, 180, 200 
            },
            {16, 1}
        }},
        {"glove-100", {
            "glove-100_base.fvecs",
            "glove-100_query.fvecs",
            "IVF1000,Flat",
            {
            // 1-10: 密集采样，召回率快速增长区间
            1, 2, 3, 4, 5, 6, 8, 10,
            // 10-50: 中等间隔，召回率中等增长区间
            15, 20, 25, 30, 35, 40, 45, 50,            
            // 50-100: 较大间隔，召回率缓慢增长区间
            60, 70, 80, 90, 100,            
            // 100-200: 最大间隔，召回率极其缓慢增长区间
            120, 140, 160, 180, 200, 230, 260, 300
            },
            {16, 1}
        }},
        {"msong", {
            "msong_base.fvecs",
            "msong_query.fvecs",
            "IVF1000,Flat",
            {
            // 1-10: 密集采样，召回率快速增长区间
            1, 2, 3, 4, 5, 6, 8, 10,
            // 10-50: 中等间隔，召回率中等增长区间
            15, 20, 25, 30, 35, 40, 45, 50,
            // 50-100: 较大间隔，召回率缓慢增长区间
            60
            },
            {16, 1}
        }},
        {"audio", {
            "audio_base.fvecs",
            "audio_query.fvecs",
            "IVF70,Flat",
            {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30
            },
            {16, 1}
        }},
        {"enron", {
            "enron_base.fvecs",
            "enron_query.fvecs",
            "IVF65,Flat",
            {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30
            },
            {16, 1}
        }},
        {"text2image", {
            "text2image_base.fvecs",
            "text2image_query.fvecs",
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
    {"1", {"基本实验单属性", {0}, "query_set_1"}},
    // {"2_1", {"多属性构建单标签搜索", {0}, "query_set_2_1"}},
    // {"2_2", {"多属性构建单标签搜索", {0}, "query_set_2_2"}},
    {"3_1", {"1%选择性实验", {15}, "query_set_3_1"}},
    {"3_2", {"25%选择性实验", {16}, "query_set_3_2"}},
    {"3_3", {"50%选择性实验", {17}, "query_set_3_3"}},
    {"3_4", {"75%选择性实验", {18}, "query_set_3_4"}},
    {"4", {"稀疏属性实验", {7}, "query_set_4"}},
    {"5_1", {"长尾分布标签实验", {1}, "query_set_5_1"}},
    {"5_2", {"正态分布标签实验", {2}, "query_set_5_2"}},
    {"5_3", {"幂律分布标签实验", {4}, "query_set_5_3"}},
    {"5_4", {"均匀分布标签实验", {0}, "query_set_5_4"}},

    // {"6", {"三标签搜索", {0, 8, 9}, "query_set_6"}},
    // {"7_1", {"多标签1%选择性", {0, 8, 15}, "query_set_7_1"}},
    // {"7_2", {"多标签25%选择性", {0, 8, 16}, "query_set_7_2"}},
    // {"7_3", {"多标签50%选择性", {0, 8, 17}, "query_set_7_3"}},
    // {"7_4", {"多标签75%选择性", {0, 8, 18}, "query_set_7_4"}}
};
    
    return {datasets, query_sets};
}


void runAllExperiments(int thread_nums, int cycle_num, bool is_save_to_file, bool isBatch) {
    // 获取数据集和查询集配置
    auto [datasets, query_sets] = get_query_config();

    // 定义自定义的数据集遍历顺序
    std::vector<std::string> dataset_order;


    dataset_order = {"audio", "enron", "glove-100", "sift", "msong", "gist", "text2image"};

    // dataset_order = {"text2image"};

    //dataset_order = {"glove-100"};
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
        experiment.setIsBatch(isBatch);
        experiment.setIsSaveToFile(is_save_to_file);
        experiment.setCycleNum(cycle_num);

        // 设置数据集名称
        experiment.setDataset(dataset_name);
        
        // 设置基础标签和向量文件路径
        std::string labelFileBase = dataDir + "labels/" + dataset_name + "/labels_with_selectivity.txt";
        if(dataset_name == "text2image"){
            labelFileBase = dataDir + "labels/" + dataset_name + "/label_1.txt";
        }
        
        
        experiment.setBaseFilePaths(
            labelFileBase,
            dataDir + "datasets/" + dataset_name + "/" + dataset_config.base_file
        );

        // 设置索引的类型和配置
        experiment.setIndexKey(dataset_config.index_key);

        experiment.setIndexPath(projectRoot + "/algorithm/Faiss/build/index/");

        experiment.setBulidIndexResultFile(resultDir + "indexdata/Faiss/" + dataset_name + ".csv");

        // 创建向量索引
        experiment.buildIndex();

        // 设置 nprobe 的值
        experiment.setNprobeValues(dataset_config.nprobes);

        // 设置线程数
        if(thread_nums != -1){
            experiment.setThreadNums({thread_nums});
        }
        else if(!experiment.isBatch){
            experiment.setThreadNums(dataset_config.thread_nums);
        }
        else{
            experiment.setThreadNums({1});
        }

        // 遍历每个查询集
        for (const auto& [query_set_name, query_set_config] : query_sets) {

            // 设置跳过的查询集
            // text2image 数据集只在单线程下跑查询集 "1"
            if(dataset_name == "text2image"){
                if(query_set_name != "1" || experiment.thread_nums[0] != 1){
                    continue;
                }
            }

            std::cout << "Processing query set: " << query_set_name << " (" << query_set_config.name << " )" << std::endl;

            // 设置文件路径
            std::string result_path;
            if(isBatch){
                result_path = resultDir +  "result/Faiss+HQI_Batch" + "/result/";
            }
            else{
                if(experiment.thread_nums[0] == 1){
                    result_path = resultDir +  "result//Faiss" +  "/result/";
                }
                else{
                    result_path = resultDir +  "result//Faiss-" + std::to_string(experiment.thread_nums[0]) + "/result/";
                }
            }
            
            experiment.setQueryFilePaths(
                dataDir + "query_label/" + dataset_name + "/" + query_set_name + ".txt",
                dataDir + "datasets/" + dataset_name + "/" + dataset_config.query_file,
                dataDir + "gt/" + dataset_name + "/gt-" + query_set_config.suffix + ".ivecs",
                result_path + dataset_name + "_" + query_set_name + "_results.csv"
            );

            
            // 设置查询时使用的属性
            experiment.setAttributes(query_set_config.attrs);

            // 执行实验
            experiment.run();
        }

    }
}

// 修改main函数，添加参数解析
int main(int argc, char *argv[]) {
    // 默认值
    int thread_nums = -1;
    int cycle_num = 1;
    bool is_save_to_file = false;

    // 批量需要结合taskset -c使用
    bool isBatch = 0;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--threads" || arg == "-t") {
            if (i + 1 < argc) {
                thread_nums = std::stoi(argv[++i]);
            }
        }
        else if (arg == "--cycles" || arg == "-c") {
            if (i + 1 < argc) {
                cycle_num = std::stoi(argv[++i]);
            }
        }
        else if (arg == "--save" || arg == "-s") {
            if (i + 1 < argc) {
                std::string value = argv[++i];
                is_save_to_file = (value == "true" || value == "1" || value == "yes");
            }
        }
        else if(arg == "--batch" || arg == "-b"){
            if (i + 1 < argc) {
                std::string value = argv[++i];
                isBatch = (value == "true" || value == "1" || value == "yes");
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
    std::cout << "线程数: " << thread_nums << std::endl;
    std::cout << "循环次数: " << cycle_num << std::endl;
    std::cout << "是否批量: " << (isBatch ? "是" : "否") << std::endl;
    std::cout << "是否保存结果: " << (is_save_to_file ? "是" : "否") << std::endl;
    
    runAllExperiments(thread_nums, cycle_num, is_save_to_file, isBatch);
    return 0;
}