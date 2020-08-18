/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.60
Copyright (C) 2018  GOMC Group
A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/
#ifdef GOMC_CUDA

#include "COO.cuh"

#define NUMBER_OF_NEIGHBOR_CELL 27
#define threadsPerBlock 256


__global__ void allocateCOO(int *gpu_cellStartIndex,
                            int *gpu_neighborList,
                            int *gpu_neighborsPerCell
                            ){
                        
  __shared__ int numberOfPairsCache[threadsPerBlock];

  int temp = 0;
  int currentCell = blockIdx.x;
  int nCellIndex = threadIdx.x;
  int neighborCell = gpu_neighborList[nCellIndex];
  int particlesInsideCurrentCell, particlesInsideNeighboringCells, endIndex;

  while(nCellIndex < NUMBER_OF_NEIGHBOR_CELL){
    // calculate number of particles inside neighbor Cell
    endIndex = gpu_cellStartIndex[neighborCell + 1];
    particlesInsideNeighboringCells = endIndex - gpu_cellStartIndex[neighborCell];

    // total number of pairs
    temp += particlesInsideNeighboringCells;
  }

  numberOfPairsCache[nCellIndex] = temp;

  int i = blockDim.x/2;
  while (i != 0) {
    if (nCellIndex < i)
        numberOfPairsCache[nCellIndex] += numberOfPairsCache[nCellIndex + i];
    __syncthreads();
    i /= 2;
  }

  if (nCellIndex == 0){
    // Calculate number of particles inside current Cell
    endIndex = gpu_cellStartIndex[currentCell + 1];
    particlesInsideCurrentCell = endIndex - gpu_cellStartIndex[currentCell];
    gpu_neighborsPerCell[blockIdx.x] = numberOfPairsCache[0] + particlesInsideCurrentCell;
  }
}

#endif