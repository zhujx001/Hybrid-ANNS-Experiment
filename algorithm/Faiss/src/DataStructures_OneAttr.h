#ifndef DATASTRUCTURES_HPP
#define DATASTRUCTURES_HPP

#include <vector>
#include <string>

// Class to represent a data point with labels and a feature vector
class DataPointLabel
{
public:
    uint16_t attribute1;
    // uint16_t attribute2;
    // uint16_t attribute3;
    // uint16_t attribute4;
    // uint16_t attribute5;
    // uint16_t attribute6;
    // uint16_t attribute7;
    // uint16_t attribute8;
    // uint16_t attribute9;
    // uint16_t attribute10;
    // uint16_t attribute11;
    // uint16_t attribute12;
    // uint16_t attribute13;
    // uint16_t attribute14;
    // uint16_t attribute15;

    DataPointLabel(const uint16_t &attr1
                   //    const uint16_t &attr2,
                   //    const uint16_t &attr3,
                   //    const uint16_t &attr4,
                   //    const uint16_t &attr5,
                   //    const uint16_t &attr6,
                   //    const uint16_t &attr7,
                   //    const uint16_t &attr8,
                   //    const uint16_t &attr9,
                   //    const uint16_t &attr10,
                   //    const uint16_t &attr11,
                   //    const uint16_t &attr12,
                   //    const uint16_t &attr13,
                   //    const uint16_t &attr14,
                   //    const uint16_t &attr15
                   )
        : attribute1(attr1)
    //   attribute2(attr2), attribute3(attr3),
    //   attribute4(attr4), attribute5(attr5), attribute6(attr6),
    //   attribute7(attr7), attribute8(attr8), attribute9(attr9),
    //   attribute10(attr10), attribute11(attr11), attribute12(attr12),
    //   attribute13(attr13), attribute14(attr14), attribute15(attr15)
    {
    }

    DataPointLabel(const std::vector<uint16_t> &attrs)
        : attribute1(attrs[0])
    //   attribute2(attrs[1]),
    //   attribute3(attrs[2]),
    //   attribute4(attrs[3]),
    //   attribute5(attrs[4]),
    //   attribute6(attrs[5]),
    //   attribute7(attrs[6]),
    //   attribute8(attrs[7]),
    //   attribute9(attrs[8]),
    //   attribute10(attrs[9]),
    //   attribute11(attrs[10]),
    //   attribute12(attrs[11]),
    //   attribute13(attrs[12]),
    //   attribute14(attrs[13]),
    //   attribute15(attrs[14])
    {
    }
};

// Class to represent a query with labels and a feature vector
class Query
{
public:
    uint16_t attribute1;
    std::vector<float> queryVector;

    Query(const uint16_t &attr1, const std::vector<float> &queryVec)
        : attribute1(attr1),
          queryVector(queryVec) {}

    Query(const std::vector<uint16_t> &attrs, const std::vector<float> &queryVec)
        : attribute1(attrs[0]),
          queryVector(queryVec)
    {
    }

    bool match(const DataPointLabel &dataPoint) const
    {
        if (attribute1 != dataPoint.attribute1)
        {
            return false;
        }
        return true;
    }

    std::string generateGroupKey()
    {
        return std::to_string(attribute1);
    }


};

#endif // DATASTRUCTURES_HPP
