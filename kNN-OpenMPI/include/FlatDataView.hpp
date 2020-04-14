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
    FlatDataView(const FlatDataView&) = default;
    FlatDataView(FlatDataView&&) = default;
    FlatDataView& operator=(const FlatDataView&) = default;
    FlatDataView& operator=(FlatDataView&&) = default;

    std::size_t getNumberOfRows() const;
    std::size_t getRowSize() const;
    FlatRowView operator[](const std::size_t);
    const FlatRowView operator[](const std::size_t) const;
    const std::vector<float>& operator*() const;

private:
    std::vector<float> flatData{};
    std::size_t rowSize{};
};

std::ostream& operator<<(std::ostream&, const FlatRowView&);