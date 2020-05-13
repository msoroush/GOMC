/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.50
Copyright (C) 2018  GOMC Group
A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/
#ifdef GOMC_CUDA
#include <cuda.h>
#include "cub/cub.cuh"
#include <stdio.h>
#include "ConstantDefinitionsCUDAKernel.cuh"
#include "CalculateMinImageCUDAKernel.cuh"
#include "CalculateForceCUDAKernel.cuh"
#include "CalculateEnergyCUDAKernel.cuh"

#define NUMBER_OF_NEIGHBOR_CELL 27

using namespace cub;

void CallBoxInterGPU(VariablesCUDA *vars,
                     vector<int> cellVector,
                     vector<int> cellStartIndex,
                     std::vector<std::vector<int> > neighborList,
                     vector<int> mapParticleToCell,
                     XYZArray const &coords,
                     BoxDimensions const &boxAxes,
                     bool electrostatic,
                     vector<double> particleCharge,
                     vector<int> particleKind,
                     vector<int> particleMol,
                     double &REn,
                     double &LJEn,
                     bool sc_coul,
                     double sc_sigma_6,
                     double sc_alpha,
                     uint sc_power,
                     uint const box)
{
  int atomNumber = coords.Count();
  int neighborListCount = neighborList.size() * NUMBER_OF_NEIGHBOR_CELL;
  int numberOfCells = neighborList.size();
  int *gpu_particleKind, *gpu_particleMol;
  int *gpu_neighborList, *gpu_cellStartIndex;
  int blocksPerGrid, threadsPerBlock;
  double *gpu_particleCharge;
  double *gpu_REn, *gpu_LJEn;
  double *gpu_final_REn, *gpu_final_LJEn;
  double cpu_final_REn, cpu_final_LJEn;

  // Run the kernel...
  threadsPerBlock = 256;
  blocksPerGrid = (int)(numberOfCells * NUMBER_OF_NEIGHBOR_CELL);

  // Convert neighbor list to 1D array
  std::vector<int> neighborlist1D(neighborListCount);
  for(int i=0; i<neighborList.size(); i++) {
    for(int j=0; j<NUMBER_OF_NEIGHBOR_CELL; j++) {
      neighborlist1D[i*NUMBER_OF_NEIGHBOR_CELL + j] = neighborList[i][j];
    }
  }

  
  gpuErrchk(cudaMalloc((void**) &gpu_neighborList, neighborListCount * sizeof(int)));
  gpuErrchk(cudaMalloc((void**) &gpu_cellStartIndex,
                       cellStartIndex.size() * sizeof(int)));
  gpuErrchk(cudaMalloc((void**) &gpu_particleCharge,
                       particleCharge.size() * sizeof(double)));
  gpuErrchk(cudaMalloc((void**) &gpu_particleKind, particleKind.size() * sizeof(int)));
  gpuErrchk(cudaMalloc((void**) &gpu_particleMol, particleMol.size() * sizeof(int)));
  cudaMalloc((void**) &gpu_REn, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc((void**) &gpu_LJEn, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  gpuErrchk(cudaMalloc((void**) &gpu_final_REn, sizeof(double)));
  gpuErrchk(cudaMalloc((void**) &gpu_final_LJEn, sizeof(double)));

  // Copy necessary data to GPU
  gpuErrchk(cudaMemcpy(vars->gpu_mapParticleToCell, &mapParticleToCell[0],
                       atomNumber * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(gpu_neighborList, &neighborlist1D[0],
                       neighborListCount * sizeof(int),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(gpu_cellStartIndex, &cellStartIndex[0],
                       cellStartIndex.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vars->gpu_cellVector, &cellVector[0],
                       atomNumber * sizeof(int),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(gpu_particleCharge, &particleCharge[0],
                       particleCharge.size() * sizeof(double),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(gpu_particleKind, &particleKind[0],
                       particleKind.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(gpu_particleMol, &particleMol[0], particleMol.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vars->gpu_x, coords.x, atomNumber * sizeof(double),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vars->gpu_y, coords.y, atomNumber * sizeof(double),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vars->gpu_z, coords.z, atomNumber * sizeof(double),
                       cudaMemcpyHostToDevice));
  checkLastErrorCUDA(__FILE__, __LINE__);

  BoxInterGPU <<< blocksPerGrid, threadsPerBlock>>>(gpu_cellStartIndex,
                                                    vars->gpu_cellVector,
                                                    gpu_neighborList,
                                                    numberOfCells,
                                                    atomNumber,
                                                    vars->gpu_mapParticleToCell,
                                                    vars->gpu_x,
                                                    vars->gpu_y,
                                                    vars->gpu_z,
                                                    boxAxes.GetAxis(box).x,
                                                    boxAxes.GetAxis(box).y,
                                                    boxAxes.GetAxis(box).z,
                                                    electrostatic,
                                                    gpu_particleCharge,
                                                    gpu_particleKind,
                                                    gpu_particleMol,
                                                    gpu_REn,
                                                    gpu_LJEn,
                                                    vars->gpu_sigmaSq,
                                                    vars->gpu_epsilon_Cn,
                                                    vars->gpu_n,
                                                    vars->gpu_VDW_Kind,
                                                    vars->gpu_isMartini,
                                                    vars->gpu_count,
                                                    vars->gpu_rCut,
                                                    vars->gpu_rCutCoulomb,
                                                    vars->gpu_rCutLow,
                                                    vars->gpu_rOn,
                                                    vars->gpu_alpha,
                                                    vars->gpu_ewald,
                                                    vars->gpu_diElectric_1,
                                                    vars->gpu_nonOrth,
                                                    vars->gpu_cell_x[box],
                                                    vars->gpu_cell_y[box],
                                                    vars->gpu_cell_z[box],
                                                    vars->gpu_Invcell_x[box],
                                                    vars->gpu_Invcell_y[box],
                                                    vars->gpu_Invcell_z[box],
                                                    sc_coul,
                                                    sc_sigma_6,
                                                    sc_alpha,
                                                    sc_power,
                                                    vars->gpu_rMin,
                                                    vars->gpu_rMaxSq,
                                                    vars->gpu_expConst,
                                                    box);
  checkLastErrorCUDA(__FILE__, __LINE__);

  // ReduceSum
  void * d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_REn,
                    gpu_final_REn, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
  CubDebugExit(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_REn,
                    gpu_final_REn, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
  cudaFree(d_temp_storage);

  // LJ ReduceSum
  d_temp_storage = NULL;
  temp_storage_bytes = 0;
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_LJEn,
                    gpu_final_LJEn, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
  CubDebugExit(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_LJEn,
                    gpu_final_LJEn, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
  cudaFree(d_temp_storage);
  // Copy back the result to CPU ! :)
  CubDebugExit(cudaMemcpy(&cpu_final_REn, gpu_final_REn, sizeof(double),
                          cudaMemcpyDeviceToHost));
  CubDebugExit(cudaMemcpy(&cpu_final_LJEn, gpu_final_LJEn, sizeof(double),
                          cudaMemcpyDeviceToHost));
  REn = cpu_final_REn;
  LJEn = cpu_final_LJEn;

  cudaDeviceSynchronize();

  cudaFree(gpu_particleCharge);
  cudaFree(gpu_particleKind);
  cudaFree(gpu_particleMol);
  cudaFree(gpu_REn);
  cudaFree(gpu_LJEn);
  cudaFree(gpu_final_REn);
  cudaFree(gpu_final_LJEn);
  cudaFree(gpu_neighborList);
  cudaFree(gpu_cellStartIndex);
}

__global__ void BoxInterGPU(int *gpu_cellStartIndex,
                            int *gpu_cellVector,
                            int *gpu_neighborList,
                            int numberOfCells,
                            int atomNumber,
                            int *gpu_mapParticleToCell,
                            double *gpu_x,
                            double *gpu_y,
                            double *gpu_z,
                            double xAxes,
                            double yAxes,
                            double zAxes,
                            bool electrostatic,
                            double *gpu_particleCharge,
                            int *gpu_particleKind,
                            int *gpu_particleMol,
                            double *gpu_REn,
                            double *gpu_LJEn,
                            double *gpu_sigmaSq,
                            double *gpu_epsilon_Cn,
                            double *gpu_n,
                            int *gpu_VDW_Kind,
                            int *gpu_isMartini,
                            int *gpu_count,
                            double *gpu_rCut,
                            double *gpu_rCutCoulomb,
                            double *gpu_rCutLow,
                            double *gpu_rOn,
                            double *gpu_alpha,
                            int *gpu_ewald,
                            double *gpu_diElectric_1,
                            int *gpu_nonOrth,
                            double *gpu_cell_x,
                            double *gpu_cell_y,
                            double *gpu_cell_z,
                            double *gpu_Invcell_x,
                            double *gpu_Invcell_y,
                            double *gpu_Invcell_z,
                            bool sc_coul,
                            double sc_sigma_6,
                            double sc_alpha,
                            uint sc_power,
                            double *gpu_rMin,
                            double *gpu_rMaxSq,
                            double *gpu_expConst,
                            int box)
{
  double distSq;
  double qi_qj_fact;
  double qqFact = 167000.0;
  double virX = 0.0, virY = 0.0, virZ = 0.0;
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  gpu_REn[threadID] = 0.0;
  gpu_LJEn[threadID] = 0.0;
  double cutoff = fmax(gpu_rCut[0], gpu_rCutCoulomb[box]);

  int currentCell = blockIdx.x / 27;
  int nCellIndex = blockIdx.x;
  int neighborCell = gpu_neighborList[nCellIndex];

  // calculate number of particles inside neighbor Cell
  int particlesInsideCurrentCell, particlesInsideNeighboringCells;
  int endIndex = neighborCell != numberOfCells - 1 ?
  gpu_cellStartIndex[neighborCell+1] : atomNumber;
  particlesInsideNeighboringCells = endIndex - gpu_cellStartIndex[neighborCell];

  // Calculate number of particles inside current Cell
  endIndex = currentCell != numberOfCells - 1 ?
  gpu_cellStartIndex[currentCell+1] : atomNumber;
  particlesInsideCurrentCell = endIndex - gpu_cellStartIndex[currentCell];

  // total number of pairs
  int numberOfPairs = particlesInsideCurrentCell * particlesInsideNeighboringCells;

  for(int pairIndex = threadIdx.x; pairIndex < numberOfPairs; pairIndex += blockDim.x) {
    int neighborParticleIndex = pairIndex / particlesInsideCurrentCell;
    int currentParticleIndex = pairIndex % particlesInsideCurrentCell;
    
    int currentParticle = gpu_cellVector[gpu_cellStartIndex[currentCell] + currentParticleIndex];
    int neighborParticle = gpu_cellVector[gpu_cellStartIndex[neighborCell] + neighborParticleIndex];

    if(currentParticle < neighborParticle) {
      // Check if they are within rcut
      distSq = 0;
      double dx = gpu_x[currentParticle] - gpu_x[neighborParticle];
      double dy = gpu_y[currentParticle] - gpu_y[neighborParticle];
      double dz = gpu_z[currentParticle] - gpu_z[neighborParticle];

      tx = min(abs(dx), xAxes - abs(dx));
      ty = min(abs(dy), yAxes - abs(dy));
      tz = min(abs(dz), zAxes - abs(dz));

      distSq = dx * dx + dy * dy + dz * dz;
      
      if((cutoff * cutoff) > distSq) {
        if(electrostatic) {
          qi_qj_fact = gpu_particleCharge[currentParticle] *
                        gpu_particleCharge[neighborParticle] * qqFact;
          gpu_REn[threadID] += CalcCoulombGPU(distSq,
                                              gpu_particleKind[currentParticle],
                                              gpu_particleKind[neighborParticle],
                                              qi_qj_fact, gpu_rCutLow[0],
                                              gpu_ewald[0], gpu_VDW_Kind[0],
                                              gpu_alpha[box],
                                              gpu_rCutCoulomb[box],
                                              gpu_isMartini[0],
                                              gpu_diElectric_1[0],
                                              sc_coul,
                                              sc_sigma_6,
                                              sc_alpha,
                                              sc_power,
                                              gpu_sigmaSq,
                                              gpu_count[0]);
        }
        gpu_LJEn[threadID] += CalcEnGPU(distSq,
                                        gpu_particleKind[currentParticle],
                                        gpu_particleKind[neighborParticle],
                                        gpu_sigmaSq, gpu_n, gpu_epsilon_Cn,
                                        gpu_VDW_Kind[0], gpu_isMartini[0],
                                        gpu_rCut[0], gpu_rOn[0], gpu_count[0],
                                        sc_sigma_6, sc_alpha, sc_power, gpu_rMin,
                                        gpu_rMaxSq, gpu_expConst);
      }
    }
  }
}

__device__ double CalcCoulombGPU(double distSq, int kind1, int kind2,
                                 double qi_qj_fact, double gpu_rCutLow,
                                 int gpu_ewald, int gpu_VDW_Kind,
                                 double gpu_alpha, double gpu_rCutCoulomb,
                                 int gpu_isMartini, double gpu_diElectric_1,
                                 bool sc_coul,
                                 double sc_sigma_6, double sc_alpha,
                                 uint sc_power, double *gpu_sigmaSq,
                                 int gpu_count)
{
  if((gpu_rCutCoulomb * gpu_rCutCoulomb) < distSq) {
    return 0.0;
  }

  int index = FlatIndexGPU(kind1, kind2, gpu_count);
  return CalcCoulombParticleGPU(distSq, qi_qj_fact, gpu_ewald, gpu_alpha,
                                sc_coul, sc_sigma_6,
                                sc_alpha, sc_power, gpu_sigmaSq);
}

__device__ double CalcEnGPU(double distSq, int kind1, int kind2,
                            double *gpu_sigmaSq, double *gpu_n,
                            double *gpu_epsilon_Cn, int gpu_VDW_Kind,
                            int gpu_isMartini, double gpu_rCut, double gpu_rOn,
                            int gpu_count,
                            double sc_sigma_6, double sc_alpha, uint sc_power,
                            double *gpu_rMin, double *gpu_rMaxSq,
                            double *gpu_expConst)
{
  if((gpu_rCut * gpu_rCut) < distSq) {
    return 0.0;
  }

  int index = FlatIndexGPU(kind1, kind2, gpu_count);
  return CalcEnParticleGPU(distSq, index, gpu_sigmaSq, gpu_n, gpu_epsilon_Cn,
                           sc_sigma_6, sc_alpha, sc_power);
}

//ElectroStatic Calculation
//**************************************************************//
__device__ double CalcCoulombParticleGPU(double distSq, double qi_qj_fact,
    double gpu_ewald, double gpu_alpha, bool sc_coul,
    double sc_sigma_6, double sc_alpha,
    uint sc_power, double *gpu_sigmaSq)
{
  if(gpu_ewald) {
    double dist = sqrt(distSq);
    double value = gpu_alpha * dist;
    return qi_qj_fact * erfc(value) / dist;
  } else {
    double dist = sqrt(distSq);
    return qi_qj_fact / dist;
  }
}


//VDW Calculation
//**************************************************************//
__device__ double CalcEnParticleGPU(double distSq, int index,
                                    double *gpu_sigmaSq, double *gpu_n,
                                    double *gpu_epsilon_Cn,
                                    double sc_sigma_6,
                                    double sc_alpha,
                                    uint sc_power)
{
  double rRat2 = gpu_sigmaSq[index] / distSq;
  double rRat4 = rRat2 * rRat2;
  double attract = rRat4 * rRat2;
  double repulse = pow(rRat2, gpu_n[index] / 2.0);
  return gpu_epsilon_Cn[index] * (repulse - attract);
}

#endif
