#include "CudaAlgorithm.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cfloat>

__device__ float distanceBetweenPointsOnCUDA(
	const float* trainingData, float* testingData,
	const std::size_t trainingDataIndex, const std::size_t testingDataIndex,
	const std::size_t rowSize)
{
	float sum = 0.f;
	for (int i = 0; i < rowSize - 2; ++i)
	{
		sum += (trainingData[trainingDataIndex * rowSize + i] - testingData[testingDataIndex * rowSize + i])
			* (trainingData[trainingDataIndex * rowSize + i] - testingData[testingDataIndex * rowSize + i]);
	}

	return sqrtf(sum);
}

__global__ void minMaxNormalizationOnCUDA(float* data, const float* minMax, const std::size_t rowSize, const std::size_t dataSize)
{
	const auto rowIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (rowIdx < dataSize)
	{
		for (auto featureIdx = 0u; featureIdx < rowSize - 2; ++featureIdx)
		{
			data[rowIdx * rowSize + featureIdx] = (data[rowIdx * rowSize + featureIdx] - minMax[2 * featureIdx])
				/ (minMax[2 * featureIdx + 1] - minMax[2 * featureIdx]);
		}
	}
}

__global__ void findMinMaxOnCuda(float* rows, float* minMaxs, const std::size_t rowSize, const std::size_t dataSize)
{
	const auto featureId = threadIdx.x + blockIdx.x * blockDim.x;
	const auto numOfFeatures = rowSize - 2;

	if (featureId < numOfFeatures)
	{
		auto min = FLT_MAX;
		auto max = FLT_MIN;

		for (auto rowIndex = 0u; rowIndex < dataSize; rowIndex++)
		{
			const auto value = rows[rowIndex * rowSize + featureId];
			if (value < min)
			{
				min = value;
			}
			if (value > max)
			{
				max = value;
			}
			minMaxs[featureId * 2] = min;
			minMaxs[featureId * 2 + 1] = max;
		}
	}
}

__global__ void knnOnCuda(
	const float* trainingData, float* testingData,
	const std::size_t trainingDataSize, const std::size_t testingDataSize,
	const std::size_t rowSize)
{
	const auto testRowIdx = threadIdx.x + blockIdx.x * blockDim.x;
	auto smallestDistance = DBL_MAX;
	auto nearestLabel = 0u;

	if (testRowIdx < testingDataSize)
	{
		for (auto trainIndex = 0u; trainIndex < trainingDataSize; ++trainIndex)
		{
			auto distance = distanceBetweenPointsOnCUDA(trainingData, testingData, trainIndex, testRowIdx, rowSize);
			if (distance < smallestDistance)
			{
				smallestDistance = distance;
				nearestLabel = trainingData[trainIndex * rowSize + rowSize - 2];
			}
		}

		testingData[testRowIdx * rowSize + rowSize - 1] = nearestLabel;
	}
}

void Cuda::knn(const FlatDataView& trainingData, FlatDataView& testingData)
{
	constexpr int NUM_OF_THREADS = 1024;
	const auto BLOCK_SIZE = std::ceil(testingData.getNumberOfRows() / NUM_OF_THREADS) + 1;

	const auto& trainingRawData = *trainingData;
	auto& testingRawData = *testingData;

	float* deviceTrainingData = nullptr;
	float* deviceTestingData = nullptr;

	cudaMalloc((void**)&deviceTrainingData, trainingRawData.size() * sizeof(float));
	cudaMalloc((void**)&deviceTestingData, testingRawData.size() * sizeof(float));

	cudaMemcpy(deviceTrainingData, trainingRawData.data(), trainingRawData.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceTestingData, testingRawData.data(), testingRawData.size() * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	knnOnCuda << < BLOCK_SIZE, NUM_OF_THREADS >> > (deviceTrainingData, deviceTestingData, trainingData.getNumberOfRows(), testingData.getNumberOfRows(), testingData.getRowSize());

	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);

	const auto cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		std::cout << "FAIL!! " << cudaGetErrorString(cudaError) << std::endl;
	}

	cudaMemcpy(testingRawData.data(), deviceTestingData, testingRawData.size() * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(deviceTrainingData);
	cudaFree(deviceTestingData);

	float timeMs{};
	cudaEventElapsedTime(&timeMs, start, stop);
	std::cout << "[CUDA] (only CUDA calculations): " << timeMs << " ms" << std::endl;
}

void Cuda::minMax(FlatDataView& data)
{
	constexpr int NUM_OF_THREADS = 1024;
	const auto BLOCK_SIZE = std::ceil(data.getNumberOfRows() / NUM_OF_THREADS) + 1;
	auto& rawData = *data;

	float* dev_rows = nullptr;
	float* dev_minMax = nullptr;

	cudaMalloc((void**)&dev_rows, rawData.size() * sizeof(float));
	cudaMalloc((void**)&dev_minMax, (data.getRowSize() - 2) * 2 * sizeof(float));
	cudaMemcpy(dev_rows, rawData.data(), rawData.size() * sizeof(float), cudaMemcpyHostToDevice);

	findMinMaxOnCuda << < BLOCK_SIZE, NUM_OF_THREADS >> > (dev_rows, dev_minMax, data.getRowSize(), data.getNumberOfRows());
	minMaxNormalizationOnCUDA << < BLOCK_SIZE, NUM_OF_THREADS >> > (dev_rows, dev_minMax, data.getRowSize(), data.getNumberOfRows());

	auto cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		std::cout << "FAIL!! " << cudaGetErrorString(cudaError) << std::endl;
	}

	cudaMemcpy(rawData.data(), dev_rows, rawData.size() * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_rows);
	cudaFree(dev_minMax);
}