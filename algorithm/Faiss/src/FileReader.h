#ifndef FILE_READER_HPP
#define FILE_READER_HPP

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

// 使用预处理器指令动态切换头文件
#ifdef USE_ONE_ATTR
#include "DataStructures_OneAttr.h"
#elif defined(USE_THREE_ATTR)
#include "DataStructures_ThreeAttr.h"
#else
#include "DataStructures_ThreeAttr.h"
#endif

// FileReader 类用于从文件中读取数据
class FileReader {
public:

    // 判断文件第一行是否需要跳过（如果第一行只有两个数字）
    static bool shouldSkipFirstLine(const std::string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            return false; // 无法打开文件，不做特殊处理
        }

        std::string firstLine;
        if (std::getline(file, firstLine)) {
            std::istringstream iss(firstLine);
            int first, second;
            
            // 检查是否有且只有两个数字
            if (iss >> first >> second) {
                // 检查是否还有其他数据
                return !(iss >> std::ws && !iss.eof());
            }
        }
        
        return false; // 默认不跳过
    }

    // 读取标签文件和向量文件并组合成 DataPoint
    static std::vector<DataPointLabel> readDataPointsLabel(const std::string& labelFile, const std::vector<int>& attributeIndices) {
        std::vector<DataPointLabel> dataPointsLabel;
        std::ifstream labelStream(labelFile);
        if (!labelStream.is_open()) {
            throw std::runtime_error("无法打开标签文件: " + labelFile);
            return dataPointsLabel;
        }

        std::string line;

        // 检查是否需要跳过第一行
        bool skip = shouldSkipFirstLine(labelFile);
        if (skip && std::getline(labelStream, line)) {
            // 跳过第一行，不做任何处理
        }

        while (std::getline(labelStream, line)) {
            std::istringstream iss(line);
            std::vector<uint16_t> attributes(attributeIndices.size());

            // 读取指定的属性
            int j = 0;
            for (size_t i = 0; i < attributeIndices.size(); ++i) {
                uint16_t attr;
                // 跳过不需要的属性
                while (j < attributeIndices[i]) {
                    ++j;
                    if (!(iss >> attr)) {
                        throw std::runtime_error("标签文件格式错误: " + line);
                    }
                }
                // 读取目标位置的属性
                ++j;
                if (!(iss >> attr)) {
                    throw std::runtime_error("标签文件格式错误: " + line);
                }
                attributes[i] = attr;
            }
            dataPointsLabel.emplace_back(attributes);
        }

        return dataPointsLabel;
    }

    static std::vector<Query> readQueries(const std::string& labelFile, const std::string& vectorFile, size_t& vectorDim, const std::vector<int>& attributeIndices) {
        std::vector<Query> queries;
        std::ifstream labelStream(labelFile);
        if (!labelStream.is_open()) {
            throw std::runtime_error("Failed to open file: " + labelFile);
            return queries;
        }

        std::ifstream vectorStream(vectorFile, std::ios::binary);
        if (!vectorStream.is_open()) {
            throw std::runtime_error("Failed to open file: " + vectorFile);
            return queries;
        }

        vectorStream.read(reinterpret_cast<char*>(&vectorDim), sizeof(int));
        vectorStream.seekg(0, std::ios::beg);
        std::string line;
        
        // 检查是否需要跳过第一行
        bool skip = shouldSkipFirstLine(labelFile);
        if (skip && std::getline(labelStream, line)) {
            // 跳过第一行，不做任何处理
        }

        while (std::getline(labelStream, line)) {
            std::istringstream iss(line);
            std::vector<uint16_t> attributes(attributeIndices.size());

            // 读取查询的属性
            for (size_t i = 0; i < attributeIndices.size(); ++i) {
                uint16_t attr;
                
                if (!(iss >> attr)) {
                    throw std::runtime_error("标签文件格式错误: " + line);
                    continue;
                }
                attributes[i] = attr;
            }

            std::vector<float> vec(vectorDim);
            vectorStream.seekg(4, std::ios::cur);
            vectorStream.read(reinterpret_cast<char*>(vec.data()), vectorDim * sizeof(float));
            if (vectorStream.gcount() != vectorDim * sizeof(float)) {
                throw std::runtime_error("向量文件数据不足");
                break;
            }

            queries.emplace_back(attributes, vec);
        }

        return queries;
    }

    static std::vector<float> readIvecs(const std::string& fileName, size_t& vectorDim) {
        std::ifstream file(fileName, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + fileName);
        }

        // 获取文件大小
        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        // 读取向量维度
        file.read(reinterpret_cast<char*>(&vectorDim), sizeof(int));
        file.seekg(0, std::ios::beg);

        // 计算向量数量
        // 每个向量占用: (4 bytes for dim + dim * 4 bytes for data)
        size_t bytesPerVector = sizeof(int) * (1 + vectorDim);
        size_t numVectors = fileSize / bytesPerVector;

        // 预分配连续内存 (总大小 = 向量数 * 维度)
        std::vector<float> data(numVectors * vectorDim);

        // 读取所有向量
        for (size_t i = 0; i < numVectors; ++i) {
            int readDim;
            file.read(reinterpret_cast<char*>(&readDim), sizeof(int));
            
            if (readDim != vectorDim) {
                throw std::runtime_error("Vector dimension mismatch in file");
            }

            // 直接读入到连续内存中的对应位置
            file.read(reinterpret_cast<char*>(&data[i * vectorDim]), vectorDim * sizeof(float));
            
            if (file.fail()) {
                // 如果读取失败
                throw std::runtime_error("Failed to read file: " + fileName);
            }
        }

        return data;
    }

    // 读取 ground truth 文件
    static std::vector<std::vector<int>> readGroundTruth(const std::string& fileName) {
        std::ifstream file(fileName, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + fileName);
        }

        std::vector<std::vector<int>> data;
        while (!file.eof()) {
            int dim;
            file.read((char*)&dim, sizeof(int));
            if (file.eof()) break;

            std::vector<int> vec(dim);
            file.read((char*)vec.data(), dim * sizeof(int));
            data.push_back(std::move(vec));
        }
        return data;
    }
};

#endif // FILE_READER_HPP
