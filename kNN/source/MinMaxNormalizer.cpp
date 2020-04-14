#include "MinMaxNormalizer.hpp"

void MinMaxNormalizer::normalize(std::vector<DataRow>& dataRows) const
{
    const auto numOfFeatures = dataRows.front().features.size();
    const auto minsMaxs = findFeatureMinMax(dataRows);

    for (auto& row : dataRows)
    {
        for (auto featureIndex = 0u; featureIndex < numOfFeatures; ++featureIndex)
        {
            row.features[featureIndex] = (row.features[featureIndex] - minsMaxs[featureIndex].min) /
                (minsMaxs[featureIndex].max - minsMaxs[featureIndex].min);
        }
    }
}

void MinMaxNormalizer::normalize(FlatDataView& dataRows) const
{
    const auto numOfRows = dataRows.getNumberOfRows();
    const auto numOfFeatures = dataRows[0].getNumberOfFeatures();
    const auto minsMaxs = findFeatureMinMax(dataRows);
    int rowIndex{};

    for (rowIndex = 0; rowIndex < numOfRows; ++rowIndex)
    {
        auto row = dataRows[rowIndex];
        for (auto featureIndex = 0u; featureIndex < numOfFeatures; ++featureIndex)
        {
            row[featureIndex] = (row[featureIndex] - minsMaxs[featureIndex].min) /
                (minsMaxs[featureIndex].max - minsMaxs[featureIndex].min);
        }
    }
}

std::vector<MinMaxNormalizer::MinMax> MinMaxNormalizer::findFeatureMinMax(const std::vector<DataRow>& dataRows) const
{
    const auto numOfFeatures = dataRows.front().features.size();
    std::vector<MinMax> featuresMinMax(numOfFeatures, { std::numeric_limits<float>::max(), std::numeric_limits<float>::min() });

    for (const auto& row : dataRows)
    {
        for (auto featureIndex = 0u; featureIndex < numOfFeatures; ++featureIndex)
        {
            const auto feature = row.features[featureIndex];
            auto& featureMinMax = featuresMinMax[featureIndex];

            if (feature < featureMinMax.min)
            {
                featureMinMax.min = feature;
            }

            if (feature > featureMinMax.max)
            {
                featureMinMax.max = feature;
            }
        }
    }

    return featuresMinMax;
}

std::vector<MinMaxNormalizer::MinMax> MinMaxNormalizer::findFeatureMinMax(const FlatDataView& dataRows) const
{
    const auto numOfFeatures = dataRows[0].getNumberOfFeatures();
    std::vector<MinMax> featuresMinMax(numOfFeatures, { std::numeric_limits<float>::max(), std::numeric_limits<float>::min() });

    for (auto rowIndex = 0u; rowIndex < dataRows.getNumberOfRows(); ++rowIndex)
    {
        const auto& row = dataRows[rowIndex];
        for (auto featureIndex = 0u; featureIndex < numOfFeatures; ++featureIndex)
        {
            const auto feature = row[featureIndex];
            auto& featureMinMax = featuresMinMax[featureIndex];

            if (feature < featureMinMax.min)
            {
                featureMinMax.min = feature;
            }

            if (feature > featureMinMax.max)
            {
                featureMinMax.max = feature;
            }
        }
    }

    return featuresMinMax;
}
