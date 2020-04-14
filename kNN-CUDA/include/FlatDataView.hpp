#pragma once

#include "DataRow.hpp"
#include "FlatRowView.hpp"
#include <vector>
#include <iostream>

class FlatDataView
{
public:
    FlatDataView() = default;
    FlatDataView(std::vector<float>&&, const std::size_t);
    FlatDataView(std::vector<DataRow>&&);

    std::size_t getNumberOfRows() const;
    std::size_t getRowSize() const;
    FlatRowView operator[](const std::size_t);
    const FlatRowView operator[](const std::size_t) const;
    const std::vector<float>& operator*() const;
    std::vector<float>& operator*();

private:
    std::vector<float> flatData{};
    std::size_t rowSize{};
};

std::ostream& operator<<(std::ostream&, const FlatRowView&);
std::ostream& operator<<(std::ostream&, const FlatDataView&);