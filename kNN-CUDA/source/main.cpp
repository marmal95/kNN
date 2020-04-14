#include "DataReader.hpp"
#include "MinMaxNormalizer.hpp"
#include "kNNClassifier.hpp"
#include "FlatDataView.hpp"
#include "HelperFunctions.hpp"
#include "CudaAlgorithm.hpp"
#include <iostream>
#include <chrono>

int main()
{
    constexpr char delimiter = ',';
    auto reader = DataReader{ "../data/smartphone_activity_dataset.csv", LabelIndex::LAST, delimiter };
    auto flatData = reader.readDataFlat();

    const auto duration = runWithTimeMeasurementCpu([&]() {
        Cuda::minMax(flatData);
        auto[trainingData, testingData] = splitData(flatData, 90);
        Cuda::knn(trainingData, testingData);
        checkAccuracy(testingData);
    });

    std::cout << "[CUDA] Duration: " << duration << " ms" << std::endl;

    return EXIT_SUCCESS;
}