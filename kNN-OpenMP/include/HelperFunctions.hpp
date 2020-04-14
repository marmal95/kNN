#pragma once

#include <chrono>
#include <iostream>
#include <algorithm>

template <typename Callable, typename... Args>
auto runWithTimeMeasurementCpu(Callable&& function, Args&&... params)
{
    const auto start = std::chrono::high_resolution_clock::now();
    std::forward<Callable>(function)(std::forward<Args>(params)...);
    const auto stop = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    return duration;
}

template <typename Object, typename Function, typename... Args>
auto runWithTimeMeasurementCpu(Object&& object, Function function, Args&&... params)
{
    const auto start = std::chrono::high_resolution_clock::now();
    (std::forward<Object>(object).*function)(std::forward<Args>(params)...);
    const auto stop = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    return duration;
}

auto splitData(const std::vector<DataRow>& data, const uint32_t ratio)
{
    auto splitPos = static_cast<uint32_t>(data.size() * (ratio / 100.f));
    std::vector<DataRow> trainingData(data.begin(), data.begin() + splitPos);
    std::vector<DataRow> testingData(data.begin() + splitPos, data.end());
    return std::make_pair(std::move(trainingData), std::move(testingData));
}

auto splitData(const FlatDataView& data, const uint32_t ratio)
{
    const auto& rawData = *data;
    const auto splitPos = static_cast<uint32_t>(data.getNumberOfRows() * (ratio / 100.f));
    std::vector<float> trainingData(rawData.begin(), rawData.begin() + splitPos * data.getRowSize());
    std::vector<float> testingData(rawData.begin() + splitPos * data.getRowSize(), rawData.end());
    FlatDataView trainingDataView{ std::move(trainingData), data.getRowSize() };
    FlatDataView testingDataView{ std::move(testingData), data.getRowSize() };
    return std::make_pair(std::move(trainingDataView), std::move(testingDataView));
}

void checkAccuracy(const std::vector<DataRow>& data)
{
    const auto numOfCorrect = std::count_if(data.cbegin(), data.cend(),
        [](const auto& row) { return row.label == row.predictedLabel; });
    std::cout << "Accuracy: " << numOfCorrect << " / " << data.size() << std::endl;
}

void checkAccuracy(const FlatDataView& data)
{
    auto numOfCorrect = 0u;
    for (std::size_t i = 0; i < data.getNumberOfRows(); ++i)
    {
        if (data[i].getLabel() == data[i].getPredictedLabel())
        {
            ++numOfCorrect;
        }
    }
    std::cout << "Accuracy: " << numOfCorrect << " / " << data.getNumberOfRows() << std::endl;
}