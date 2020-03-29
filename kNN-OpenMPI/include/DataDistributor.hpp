#pragma once

#include "FlatDataView.hpp"
#include "Accuracy.hpp"

class DataDistributor
{
public:
	static FlatDataView distributeTrainingData(const FlatDataView&);
	static FlatDataView distributeTestingData(const FlatDataView&);
	static FlatDataView collectData(const FlatDataView&);
	static Accuracy collectAccuracy(const Accuracy&);
};