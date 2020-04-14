#pragma once

#include <cstdlib>

class FlatRowView
{
public:
    FlatRowView(const float*, const std::size_t);

    std::size_t getNumberOfFeatures() const;
    const float& getLabel() const;
    const float& getPredictedLabel() const;
    float& operator[](const std::size_t);
    const float& operator[](const std::size_t) const;
    void setPredictedLabel(const float);

private:
    float* row;
    const std::size_t size;
};
