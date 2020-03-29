#include "DataReader.hpp"
#include "MinMaxNormalizer.hpp"
#include "kNNClassifier.hpp"
#include "FlatDataView.hpp"
#include "HelperFunctions.hpp"
#include "MpiHelpers.hpp"
#include "DataDistributor.hpp"
#include <iostream>
#include <chrono>
#include <mpi.h>

int main()
{
	MPI_Init(nullptr, nullptr);

	FlatDataView trainingData{}, testingData{};
	FlatDataView wholeData{};
	Accuracy totalAccuracy{};

	if (MPI::isMasterProcess())
	{
		constexpr char delimiter = ',';
		auto reader = DataReader{ "../data/smartphone_activity_dataset.csv", LabelIndex::LAST, delimiter };
		wholeData = reader.readDataFlat();
	}

	auto processWholeData = DataDistributor::distributeTrainingData(wholeData);
	MinMaxNormalizer{}.normalize(processWholeData);
	wholeData = DataDistributor::collectData(processWholeData);

	if (MPI::isMasterProcess())
	{
		auto[tempTrainingData, tempTestingData] = splitData(wholeData, 90);
		trainingData = std::move(tempTrainingData);
		testingData = std::move(tempTestingData);
	}

	const auto duration = runWithTimeMeasurementCpu([&]() {
		auto processTrainingData = DataDistributor::distributeTrainingData(trainingData);
		auto processTestingData = DataDistributor::distributeTestingData(testingData);

		kNNClassifier classifier{ processTrainingData };
		classifier.predict(processTestingData);
		totalAccuracy = DataDistributor::collectAccuracy(checkAccuracy(processTestingData));
	});

	if (MPI::isMasterProcess())
	{
		std::cout << "Total accuracy: " << totalAccuracy << std::endl;
		std::cout << "Time: " << duration << " ms" << std::endl;
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}