#pragma once

#include "DataRow.hpp"
#include "FlatDataView.hpp"
#include <vector>

template <typename T>
class kNNClassifier
{
public:
    kNNClassifier(const T&);
    void predict(std::vector<DataRow>&);
    void predict(FlatDataView&);

private:
    double calculateDistance(const DataRow&, const DataRow&) const;
    double calculateDistance(const FlatRowView&, const FlatRowView&) const;

    const T trainingData;
};

template <typename T>
kNNClassifier<T>::kNNClassifier(const T& trainingData)
    : trainingData{ trainingData }
{}

template <typename T>
void kNNClassifier<T>::predict(std::vector<DataRow>& testingData)
{
    for (auto& testRow : testingData)
    {
        auto smallestDistance = std::numeric_limits<double>::max();
        auto nearestLabel = 0u;

        for (const auto& trainRow : trainingData)
        {
            const auto distance = calculateDistance(trainRow, testRow);
            if (distance < smallestDistance)
            {
                smallestDistance = distance;
                nearestLabel = trainRow.label;
            }
        }

        testRow.predictedLabel = nearestLabel;
    }
}

template <typename T>
void kNNClassifier<T>::predict(FlatDataView& testingData)
{
    const auto testingDataSize = testingData.getNumberOfRows();

    for (auto i = 0u; i < testingDataSize; ++i)
    {
        auto smallestDistance = std::numeric_limits<double>::max();
        float nearestLabel = 0;

        const auto trainingDataSize = trainingData.getNumberOfRows();

        for (auto j = 0u; j < trainingDataSize; ++j)
        {
            const auto distance = calculateDistance(trainingData[j], testingData[i]);
            if (distance < smallestDistance)
            {
                smallestDistance = distance;
                nearestLabel = trainingData[j].getLabel();
            }
        }

        testingData[i].setPredictedLabel(nearestLabel);
    }
}

template <typename T>
double kNNClassifier<T>::calculateDistance(const DataRow& lhs, const DataRow& rhs) const
{
    const auto numOfFeatures = lhs.features.size();
    double sum = 0;

    for (auto i = 0u; i < numOfFeatures; ++i)
    {
        sum += std::pow(lhs.features[i] - rhs.features[i], 2);
    }

    return std::sqrt(sum);
}

template<typename T>
double kNNClassifier<T>::calculateDistance(const FlatRowView& lhs, const FlatRowView& rhs) const
{
    const auto numOfFeatures = lhs.getNumberOfFeatures();
    double sum = 0;

    for (auto i = 0u; i < numOfFeatures; ++i)
    {
        sum += std::pow(lhs[i] - rhs[i], 2);
    }

    return std::sqrt(sum);
}
