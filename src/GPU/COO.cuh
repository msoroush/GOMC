/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.60
Copyright (C) 2018  GOMC Group
A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/
#ifndef COO_KERNEL
#define COO_KERNEL

#ifdef GOMC_CUDA
#include <cuda.h>
#include "CUDAMemoryManager.cuh"

__global__ void allocateCOO(int *gpu_cellStartIndex,
                            int *gpu_neighborList,
                            int *gpu_neighborsPerCell);

#endif /*GOMC_CUDA*/
#endif /*COO_KERNEL*/