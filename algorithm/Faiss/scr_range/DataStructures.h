#ifndef DATASTRUCTURES_HPP
#define DATASTRUCTURES_HPP

#include <vector>
#include <string>

// class DataPointLabel
// {
// public:
//     uint16_t id;

//     DataPointLabel(const uint16_t &attr1)
//         : id(attr1)
//     {
//     }
//     DataPointLabel(const std::vector<uint16_t> &attrs)
//         : id(attrs[0])
//     {
//     }

// };

// Class to represent a query with labels and a feature vector
class Query
{
public:
    uint32_t imin;
    uint32_t imax;
    std::vector<float> queryVector;

    Query(const uint32_t &_imin, const uint32_t &_imax, const std::vector<float> &queryVec)
        : imin(_imin), imax(_imax),
          queryVector(queryVec) {}

    Query(const std::vector<uint32_t> &attrs, const std::vector<float> &queryVec)
        : imin(attrs[0]), imax(attrs[1]),
          queryVector(queryVec)
    {
    }




    std::string generateGroupKey()
    {
        return std::to_string(imin) + "_" + std::to_string(imax);
    }


};

#endif // DATASTRUCTURES_HPP
