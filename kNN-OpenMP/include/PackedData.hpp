#pragma once

#include <vector>

class PackedData
{
public:
    PackedData(std::vector<double> data)
        : data{ std::move(data) }
    {}

private:
    std::vector<double> data;
};