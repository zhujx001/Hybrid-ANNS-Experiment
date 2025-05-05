#ifndef FILE_READER_HPP
#define FILE_READER_HPP

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

#include "DataStructures.h"


// FileReader 类用于从文件中读取数据
class FileReader {
public:

    static std::vector<Query> readQueries(const std::string& labelFile, const std::string& vectorFile, size_t& vectorDim) {
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
        
        //std::getline(labelStream, line);  // 跳过文件头

        while (std::getline(labelStream, line)) {
            std::istringstream iss(line);
            std::vector<uint32_t> ranges(2);

            // 读取查询的范围
            for (size_t i = 0; i < ranges.size(); ++i) {
                uint32_t attr;
                
                if (!(iss >> attr)) {
                    throw std::runtime_error("标签文件格式错误: " + line);
                    continue;
                }
                ranges[i] = attr;
            }

            std::vector<float> vec(vectorDim);
            vectorStream.seekg(4, std::ios::cur);
            vectorStream.read(reinterpret_cast<char*>(vec.data()), vectorDim * sizeof(float));
            if (vectorStream.gcount() != vectorDim * sizeof(float)) {
                throw std::runtime_error("向量文件数据不足");
                break;
            }

            queries.emplace_back(ranges, vec);
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

    static std::vector<std::vector<int>> readGroundTruth_txt(const std::string& fileName) {
        std::ifstream file(fileName);  // 移除 std::ios::binary 标志
        if (!file) {
            throw std::runtime_error("Failed to open file: " + fileName);
        }
    
        std::vector<std::vector<int>> data;
        std::string line;
        
        // 逐行读取文件
        while (std::getline(file, line)) {
            std::vector<int> vec;
            std::istringstream iss(line);
            int value;
            
            // 从当前行读取所有整数
            while (iss >> value) {
                vec.push_back(value);
            }
            
            // 如果需要固定10个值，可以进行检查
            if (vec.size() != 10) {
                std::cerr << "Warning: Line contains " << vec.size() << " values instead of 10" << std::endl;
            }
            
            data.push_back(std::move(vec));
        }
        
        return data;
    }
};

#endif // FILE_READER_HPP
