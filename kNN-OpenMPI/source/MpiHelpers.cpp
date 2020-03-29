#include "MpiHelpers.hpp"

namespace MPI
{
	int getRank()
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		return rank;
	}

	int getWorldSize()
	{
		int size;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		return size;
	}

	int isMasterProcess()
	{
		return getRank() == MASTER_PROCESS;
	}

	void synchronizeProcesses()
	{
		MPI_Barrier(MPI_COMM_WORLD);
	}
}