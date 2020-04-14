#include "DataDistributor.hpp"
#include "MpiHelpers.hpp"
#include <numeric>
#include <mpi.h>

FlatDataView DataDistributor::distributeTrainingData(const FlatDataView& data)
{
    const auto& rawData = *data;
    std::vector<float> dataBuffer{};
    std::size_t rowSizeBuffer{};
    std::size_t dataSizeBuffer{};

    if (MPI::isMasterProcess())
    {
        dataBuffer = *data;
        rowSizeBuffer = data.getRowSize();
        dataSizeBuffer = dataBuffer.size();
    }

    MPI_Bcast(&rowSizeBuffer, 1, MPI_UNSIGNED_LONG_LONG, MPI::MASTER_PROCESS, MPI_COMM_WORLD);
    MPI_Bcast(&dataSizeBuffer, 1, MPI_UNSIGNED_LONG_LONG, MPI::MASTER_PROCESS, MPI_COMM_WORLD);

    dataBuffer.resize(dataSizeBuffer);
    MPI_Bcast(dataBuffer.data(), static_cast<int>(dataSizeBuffer), MPI_FLOAT, MPI::MASTER_PROCESS, MPI_COMM_WORLD);

    return { std::move(dataBuffer), static_cast<std::size_t>(rowSizeBuffer) };
}

FlatDataView DataDistributor::distributeTestingData(const FlatDataView& data)
{
    const auto& rawData = *data;
    std::vector<float> dataBuffer{};
    std::size_t rowSizeBuffer{};
    std::size_t elementsPerProcBuffer{};

    if (MPI::isMasterProcess())
    {
        rowSizeBuffer = data.getRowSize();
        elementsPerProcBuffer = (data.getNumberOfRows() / MPI::getWorldSize()) * data.getRowSize();
    }

    MPI_Bcast(&rowSizeBuffer, 1, MPI_UNSIGNED_LONG_LONG, MPI::MASTER_PROCESS, MPI_COMM_WORLD);
    MPI_Bcast(&elementsPerProcBuffer, 1, MPI_UNSIGNED_LONG_LONG, MPI::MASTER_PROCESS, MPI_COMM_WORLD);

    dataBuffer.resize(elementsPerProcBuffer);
    MPI_Scatter(rawData.data(), static_cast<int>(elementsPerProcBuffer), MPI_FLOAT, dataBuffer.data(),
        static_cast<int>(elementsPerProcBuffer), MPI_FLOAT, MPI::MASTER_PROCESS, MPI_COMM_WORLD);

    return { std::move(dataBuffer), rowSizeBuffer };
}

FlatDataView DataDistributor::collectData(const FlatDataView& data)
{
    const auto& rawData = *data;

    std::vector<float> dataBuffer{};
    dataBuffer.resize((*data).size());

    const auto rowSize = data.getRowSize();
    const auto numOfRows = data.getNumberOfRows();
    const auto elemsPerProcess = numOfRows / MPI::getWorldSize();
    const auto dataElemsPerProcess = elemsPerProcess * data.getRowSize();
    const auto processBegin = MPI::getRank() * elemsPerProcess;
    const auto processEnd = processBegin + elemsPerProcess;
    const auto procBeginData = processBegin * data.getRowSize();
    const auto procEndData = processEnd * data.getRowSize();

    std::vector<float> processBuffer{};
    processBuffer.reserve(dataElemsPerProcess);

    for (auto i = procBeginData; i < procEndData; i++)
    {
        processBuffer.push_back(rawData[i]);
    }

    MPI_Gather(processBuffer.data(), static_cast<int>(processBuffer.size()), MPI_FLOAT, dataBuffer.data(),
        static_cast<int>(processBuffer.size()), MPI_FLOAT, MPI::MASTER_PROCESS, MPI_COMM_WORLD);

    return { std::move(dataBuffer), rowSize };
}

Accuracy DataDistributor::collectAccuracy(const Accuracy& processAccuracy)
{
    std::vector<Accuracy> processesAccuracy{};
    processesAccuracy.resize(MPI::getWorldSize());

    MPI_Gather(&processAccuracy, sizeof(processAccuracy), MPI_BYTE,
        processesAccuracy.data(), sizeof(processAccuracy), MPI_BYTE, MPI::MASTER_PROCESS, MPI_COMM_WORLD);

    const std::size_t initialSum = 0;
    const auto sumCorrect = std::accumulate(processesAccuracy.begin(), processesAccuracy.end(), initialSum,
        [&](const std::size_t sum, const auto& accuracy) { return sum + accuracy.correct; });
    const auto sumAll = std::accumulate(processesAccuracy.begin(), processesAccuracy.end(), initialSum,
        [&](const std::size_t sum, const auto& accuracy) { return sum + accuracy.all; });
    
    return MPI::isMasterProcess() ? Accuracy{ sumCorrect, sumAll } : processAccuracy;
}
