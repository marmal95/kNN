#include "FlatDataView.hpp"

FlatDataView::FlatDataView(std::vector<float>&& data, const std::size_t rowSize)
    : flatData{ std::move(data) }, rowSize{ rowSize }
{}

FlatDataView::FlatDataView(std::vector<DataRow>&& data)
    : rowSize{ data.front().features.size() + 2 }
{
    for (auto&& row : data)
    {
        flatData.insert(std::end(flatData), std::begin(row.features), std::end(row.features));
        flatData.push_back(static_cast<float>(row.label));
        flatData.push_back(static_cast<float>(row.predictedLabel));
    }
}

std::size_t FlatDataView::getNumberOfRows() const
{
    return flatData.size() / (rowSize);
}

std::size_t FlatDataView::getRowSize() const
{
    return rowSize;
}

FlatRowView FlatDataView::operator[](const std::size_t index)
{
    const auto size = rowSize;
    const auto row = flatData.data() + index * size;
    return { row, size };
}

const FlatRowView FlatDataView::operator[](const std::size_t index) const
{
    const auto size = rowSize;
    const auto row = flatData.data() + index * size;
    return { row, size };
}

const std::vector<float>& FlatDataView::operator*() const
{
    return flatData;
}

std::ostream& operator<<(std::ostream& os, const FlatRowView& row)
{
    for (auto featureIndex = 0u; featureIndex < row.getNumberOfFeatures(); ++featureIndex)
    {
        std::cout << row[featureIndex] << " ";
    }
    std::cout << " -> " << row.getPredictedLabel() << " / " << row.getLabel() << std::endl;
    return os;
}