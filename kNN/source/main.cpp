#include "DataReader.hpp"
#include "MinMaxNormalizer.hpp"
#include "kNNClassifier.hpp"
#include "FlatDataView.hpp"
#include "HelperFunctions.hpp"
#include <iostream>
#include <chrono>

int main()
{
    constexpr char delimiter = ',';
    auto reader = DataReader{ "../data/smartphone_activity_dataset.csv", LabelIndex::LAST, delimiter };
    auto flatData = reader.readDataFlat();
    auto objectData = reader.readData();

    {
        const auto duration = runWithTimeMeasurementCpu([&]() {
            MinMaxNormalizer{}.normalize(objectData);
        });
        std::cout << "Object normalization: " << duration << " ms" << std::endl;
    }
    {
        const auto duration = runWithTimeMeasurementCpu([&]() {
            MinMaxNormalizer{}.normalize(flatData);
        });
        std::cout << "Flat normalization: " << duration << " ms" << std::endl;
    }

    {
        auto[trainingData, testingData] = splitData(objectData, 90);

        const auto duration = runWithTimeMeasurementCpu([&]() {
            kNNClassifier classifier{ trainingData };
            classifier.predict(testingData);
        });

        std::cout << "Object: " << duration << " ms" << std::endl;
        checkAccuracy(testingData);
    }

    {
        auto[trainingData, testingData] = splitData(flatData, 90);

        const auto duration = runWithTimeMeasurementCpu([&]() {
            kNNClassifier classifier{ trainingData };
            classifier.predict(testingData);
        });

        std::cout << "Flat: " << duration << " ms" << std::endl;
        checkAccuracy(testingData);
    }

    return EXIT_SUCCESS;
}