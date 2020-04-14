#pragma once

#include <vector>
#include <iostream>

struct DataRow
{
    std::vector<float> features{};
    uint32_t label{};
    uint32_t predictedLabel{ std::numeric_limits<uint32_t>::max() };
};

std::ostream& operator<<(std::ostream& os, const DataRow& row);
std::ostream& operator<<(std::ostream& os, const std::vector<DataRow>& dataRows);