# kNN 

The project contains sequential and different parallel implementations of **kNN** algorithm.<br />
The project was created for the needs of master's thesis of Computer Science studies.<br/>
The aim of the project was to compare performance of sequential and parallel implementations, depending on threads and processes number used for computing, with the use of different approaches and technologies.

## Technologies

* C++
* [OpenMP](https://en.wikipedia.org/wiki/OpenMP)
* [OpenMPI](https://en.wikipedia.org/wiki/Open_MPI)
* [CUDA](https://en.wikipedia.org/wiki/CUDA)

## Implementations

The projects contains a few **kNN** implementations:
* Sequential implementation - using pure C++
* Multithreaded implementation with the use of **OpenMP**
* Parallel implementation with the use of **OpenMPI**
* Massive parallel implementation with use of **CUDA**

## Installation

### Dependencies
* Visual Studio with C++17 support or higher
* OpenMP support enabled in Visual Studio (for kNN-OpenMP project)
* OpenMPI implementation installed e.g. Microsoft MPI
* CUDA installed (+ CUDA capable graphic card)

### Repository

```sh
$ git clone https://github.com/marmal95/kNN.git
```

### Build

The Visual Studio solution contains four projects inside - responding four implementations mentioned in [Implementations](#Implementations) section.
<br/>
Build whole solution by choosing:
```
Build > Build Solution
```
from top menu, or right-click specific project in **Solution Explorer** and choose:
```
Build
```

### Run
Right-click on chosen project in **Solution Explorer** view and click **Set as Startup Project**.<br/>
Click **F5** or choose **Debug >> Start Debugging** from top menu. 


## Customization

### OpenMP

Preferred number of theads in OpenMP implementation used for computing may be changed with function call:
```
omp_set_num_threads(NUM_THREADS)
```
which is called at the beginning of **main()** function.


### OpenMPI

Preferred number of processes is passed as parameter to **mpiexec** command.<br>
The value may be changed in **Visual Studio** in: **Project > Properties > Configuration Properties > Debugging**.<br>
```
Command             mpiexec.exe
Command Arguments   -n 4 "$(TargetPath)"
```


### CUDA

Preferred size of grid used for computing is specified inside **CudaAlgorithm.cu** file in **kNN-CUDA** project.<br>
```
knnOnCuda<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>
```
Inside that call the number of blocks and number of threads per each block is specified.