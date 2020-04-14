#include "DataRow.hpp"
#include <iostream>

std::ostream& operator<<(std::ostream& os, const DataRow& row)
{
    for (const auto feature : row.features)
    {
        std::cout << feature << " ";
    }
    std::cout << " -> " << row.predictedLabel << " / " << row.label << std::endl;
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<DataRow>& dataRows)
{
    for (const auto& elem : dataRows)
    {
        std::cout << elem;
    }
    return os;
}