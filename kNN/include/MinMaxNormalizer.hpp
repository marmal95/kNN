#pragma once

#include "DataRow.hpp"
#include "FlatDataView.hpp"
#include <vector>

class MinMaxNormalizer
{
public:
    void normalize(std::vector<DataRow>&) const;
    void normalize(FlatDataView&) const;

    struct MinMax
    {
        float min;
        float max;

        bool operator==(const MinMax& other)
        {
            constexpr auto epsilon = 10e-4f;
            return (std::abs(min - other.min) < epsilon) && (std::abs(max - other.max) < epsilon);
        }
    };

//private:
    std::vector<MinMax> findFeatureMinMax(const std::vector<DataRow>&) const;
    std::vector<MinMax> findFeatureMinMax(const FlatDataView&) const;
};