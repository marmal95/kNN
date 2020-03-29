#pragma once

#include <mpi.h>

namespace MPI
{
	constexpr auto MASTER_PROCESS = 0;

	int getRank();
	int getWorldSize();
	int isMasterProcess();
	void synchronizeProcesses();
}