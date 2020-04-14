#include "FlatRowView.hpp"

FlatRowView::FlatRowView(const float* row, const std::size_t size)
    : row{ const_cast<float*>(row) }, size{ size }
{}

std::size_t FlatRowView::getNumberOfFeatures() const
{
    return size - 2;
}

const float& FlatRowView::getLabel() const
{
    return *(row + (size - 2));
}

const float& FlatRowView::getPredictedLabel() const
{
    return *(row + (size - 1));
}

float & FlatRowView::operator[](const std::size_t index)
{
    return row[index];
}

const float& FlatRowView::operator[](const std::size_t index) const
{
    return row[index];
}

void FlatRowView::setPredictedLabel(const float label)
{
    *(row + (size - 1)) = label;
}