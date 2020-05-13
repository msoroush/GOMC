/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.50
Copyright (C) 2018  GOMC Group
A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/
#ifdef GOMC_CUDA

#include <cuda.h>
#include "CalculateForceCUDAKernel.cuh"
#include "CalculateEnergyCUDAKernel.cuh"
#include "ConstantDefinitionsCUDAKernel.cuh"
#include "CalculateMinImageCUDAKernel.cuh"
#include "cub/cub.cuh"
#include <stdio.h>

using namespace cub;
#define NUMBER_OF_NEIGHBOR_CELL 27

void CallBoxInterForceGPU(VariablesCUDA *vars,
                          vector<int> &cellVector,
                          vector<int> &cellStartIndex,
                          std::vector<std::vector<int> > &neighborList,
                          vector<int> &mapParticleToCell,
                          XYZArray const &currentCoords,
                          XYZArray const &currentCOM,
                          BoxDimensions const &boxAxes,
                          bool electrostatic,
                          vector<double> &particleCharge,
                          vector<int> &particleKind,
                          vector<int> &particleMol,
                          double &rT11,
                          double &rT12,
                          double &rT13,
                          double &rT22,
                          double &rT23,
                          double &rT33,
                          double &vT11,
                          double &vT12,
                          double &vT13,
                          double &vT22,
                          double &vT23,
                          double &vT33,
                          bool sc_coul,
                          double sc_sigma_6,
                          double sc_alpha,
                          uint sc_power,
                          uint const box)
{
  int atomNumber = currentCoords.Count();
  int molNumber = currentCOM.Count();
  int neighborListCount = neighborList.size() * NUMBER_OF_NEIGHBOR_CELL;
  int numberOfCells = neighborList.size();
  int *gpu_particleKind;
  int *gpu_particleMol;
  int *gpu_neighborList, *gpu_cellStartIndex;
  int blocksPerGrid, threadsPerBlock;
  double *gpu_particleCharge;
  double *gpu_final_value;

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
  cudaMalloc((void**) &gpu_particleCharge,
             particleCharge.size() * sizeof(double));
  cudaMalloc((void**) &gpu_particleKind, particleKind.size() * sizeof(int));
  cudaMalloc((void**) &gpu_particleMol, particleMol.size() * sizeof(int));
  cudaMalloc((void**) &gpu_final_value, sizeof(double));
  cudaMalloc(&vars->gpu_rT11, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc(&vars->gpu_rT12, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc(&vars->gpu_rT13, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc(&vars->gpu_rT22, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc(&vars->gpu_rT23, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc(&vars->gpu_rT33, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc(&vars->gpu_vT11, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc(&vars->gpu_vT12, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc(&vars->gpu_vT13, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc(&vars->gpu_vT22, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc(&vars->gpu_vT23, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc(&vars->gpu_vT33, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));

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
  cudaMemcpy(vars->gpu_x, currentCoords.x, atomNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_y, currentCoords.y, atomNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_z, currentCoords.z, atomNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_comx, currentCOM.x, molNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_comy, currentCOM.y, molNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_comz, currentCOM.z, molNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_particleCharge, &particleCharge[0],
             particleCharge.size() * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_particleKind, &particleKind[0],
             particleKind.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_particleMol, &particleMol[0],
             particleMol.size() * sizeof(int),
             cudaMemcpyHostToDevice);

  BoxInterForceGPU <<< blocksPerGrid, threadsPerBlock>>>(gpu_cellStartIndex,
                                                         vars->gpu_cellVector,
                                                         gpu_neighborList,
                                                         numberOfCells,
                                                         atomNumber,
                                                         vars->gpu_mapParticleToCell,
                                                         vars->gpu_x,
                                                         vars->gpu_y,
                                                         vars->gpu_z,
                                                         vars->gpu_comx,
                                                         vars->gpu_comy,
                                                         vars->gpu_comz,
                                                         boxAxes.GetAxis(box).x,
                                                         boxAxes.GetAxis(box).y,
                                                         boxAxes.GetAxis(box).z,
                                                         electrostatic,
                                                         gpu_particleCharge,
                                                         gpu_particleKind,
                                                         gpu_particleMol,
                                                         vars->gpu_rT11,
                                                         vars->gpu_rT12,
                                                         vars->gpu_rT13,
                                                         vars->gpu_rT22,
                                                         vars->gpu_rT23,
                                                         vars->gpu_rT33,
                                                         vars->gpu_vT11,
                                                         vars->gpu_vT12,
                                                         vars->gpu_vT13,
                                                         vars->gpu_vT22,
                                                         vars->gpu_vT23,
                                                         vars->gpu_vT33,
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
                                                         vars->gpu_cell_x[box],
                                                         vars->gpu_cell_y[box],
                                                         vars->gpu_cell_z[box],
                                                         vars->gpu_Invcell_x[box],
                                                         vars->gpu_Invcell_y[box],
                                                         vars->gpu_Invcell_z[box],
                                                         vars->gpu_nonOrth,
                                                         sc_coul,
                                                         sc_sigma_6,
                                                         sc_alpha,
                                                         sc_power,
                                                         vars->gpu_rMin,
                                                         vars->gpu_rMaxSq,
                                                         vars->gpu_expConst,
                                                         box);
  checkLastErrorCUDA(__FILE__, __LINE__);
  cudaDeviceSynchronize();
  // ReduceSum // Virial of LJ
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT11,
                    gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT11,
                    gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
  cudaMemcpy(&vT11, gpu_final_value, sizeof(double),
             cudaMemcpyDeviceToHost);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT12,
                    gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
  cudaMemcpy(&vT12, gpu_final_value, sizeof(double),
             cudaMemcpyDeviceToHost);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT13,
                    gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
  cudaMemcpy(&vT13, gpu_final_value, sizeof(double),
             cudaMemcpyDeviceToHost);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT22,
                    gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
  cudaMemcpy(&vT22, gpu_final_value, sizeof(double),
             cudaMemcpyDeviceToHost);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT23,
                    gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
  cudaMemcpy(&vT23, gpu_final_value, sizeof(double),
             cudaMemcpyDeviceToHost);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT33,
                    gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
  cudaMemcpy(&vT33, gpu_final_value, sizeof(double),
             cudaMemcpyDeviceToHost);

  if(electrostatic) {
    // ReduceSum // Virial of Coulomb
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT11,
                      gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
    cudaMemcpy(&rT11, gpu_final_value, sizeof(double),
               cudaMemcpyDeviceToHost);
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT12,
                      gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
    cudaMemcpy(&rT12, gpu_final_value, sizeof(double),
               cudaMemcpyDeviceToHost);
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT13,
                      gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
    cudaMemcpy(&rT13, gpu_final_value, sizeof(double),
               cudaMemcpyDeviceToHost);
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT22,
                      gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
    cudaMemcpy(&rT22, gpu_final_value, sizeof(double),
               cudaMemcpyDeviceToHost);
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT23,
                      gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
    cudaMemcpy(&rT23, gpu_final_value, sizeof(double),
               cudaMemcpyDeviceToHost);
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT33,
                      gpu_final_value, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
                    threadsPerBlock);
    cudaMemcpy(&rT33, gpu_final_value, sizeof(double),
               cudaMemcpyDeviceToHost);
  }

  cudaFree(vars->gpu_rT11);
  cudaFree(vars->gpu_rT12);
  cudaFree(vars->gpu_rT13);
  cudaFree(vars->gpu_rT22);
  cudaFree(vars->gpu_rT23);
  cudaFree(vars->gpu_rT33);
  cudaFree(vars->gpu_vT11);
  cudaFree(vars->gpu_vT12);
  cudaFree(vars->gpu_vT13);
  cudaFree(vars->gpu_vT22);
  cudaFree(vars->gpu_vT23);
  cudaFree(vars->gpu_vT33);
  cudaFree(d_temp_storage);
  cudaFree(gpu_neighborList);
  cudaFree(gpu_cellStartIndex);
  cudaFree(gpu_particleKind);
  cudaFree(gpu_particleMol);
  cudaFree(gpu_particleCharge);
  cudaFree(gpu_final_value);
}

void CallBoxForceGPU(VariablesCUDA *vars,
                     vector<int> &cellVector,
                     vector<int> &cellStartIndex,
                     std::vector<std::vector<int> > &neighborList,
                     vector<int> &mapParticleToCell,
                     XYZArray const &coords,
                     BoxDimensions const &boxAxes,
                     bool electrostatic,
                     vector<double> &particleCharge,
                     vector<int> &particleKind,
                     vector<int> &particleMol,
                     double &REn,
                     double &LJEn,
                     double *aForcex,
                     double *aForcey,
                     double *aForcez,
                     double *mForcex,
                     double *mForcey,
                     double *mForcez,
                     int atomCount,
                     int molCount,
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

  cudaMemset(vars->gpu_aForcex, 0, atomCount * sizeof(double));
  cudaMemset(vars->gpu_aForcey, 0, atomCount * sizeof(double));
  cudaMemset(vars->gpu_aForcez, 0, atomCount * sizeof(double));
  cudaMemset(vars->gpu_mForcex, 0, molCount * sizeof(double));
  cudaMemset(vars->gpu_mForcey, 0, molCount * sizeof(double));
  cudaMemset(vars->gpu_mForcez, 0, molCount * sizeof(double));

  gpuErrchk(cudaMemcpy(vars->gpu_mapParticleToCell, &mapParticleToCell[0],
    atomNumber * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**) &gpu_neighborList, neighborListCount * sizeof(int)));
  gpuErrchk(cudaMalloc((void**) &gpu_cellStartIndex,
                       cellStartIndex.size() * sizeof(int)));
  cudaMalloc((void**) &gpu_particleCharge,
             particleCharge.size() * sizeof(double));
  cudaMalloc((void**) &gpu_particleKind, particleKind.size() * sizeof(int));
  cudaMalloc((void**) &gpu_particleMol, particleMol.size() * sizeof(int));
  cudaMalloc((void**) &gpu_REn, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc((void**) &gpu_LJEn, numberOfCells * NUMBER_OF_NEIGHBOR_CELL *
             threadsPerBlock * sizeof(double));
  cudaMalloc((void**) &gpu_final_REn, sizeof(double));
  cudaMalloc((void**) &gpu_final_LJEn, sizeof(double));

  // Copy necessary data to GPU
  gpuErrchk(cudaMemcpy(gpu_neighborList, &neighborlist1D[0],
                       neighborListCount * sizeof(int),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(gpu_cellStartIndex, &cellStartIndex[0],
                       cellStartIndex.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vars->gpu_cellVector, &cellVector[0],
                       atomNumber * sizeof(int),
                       cudaMemcpyHostToDevice));
  cudaMemcpy(gpu_particleCharge, &particleCharge[0],
             particleCharge.size() * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_particleKind, &particleKind[0],
             particleKind.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_particleMol, &particleMol[0], particleMol.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  
  cudaMemcpy(vars->gpu_x, coords.x, atomNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_y, coords.y, atomNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_z, coords.z, atomNumber * sizeof(double),
             cudaMemcpyHostToDevice);

  checkLastErrorCUDA(__FILE__, __LINE__);
  BoxForceRealGPU <<< blocksPerGrid, threadsPerBlock, 0, vars->streams[1]>>>(gpu_cellStartIndex,
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
                                                    vars->gpu_aForcex,
                                                    vars->gpu_aForcey,
                                                    vars->gpu_aForcez,
                                                    vars->gpu_mForcex,
                                                    vars->gpu_mForcey,
                                                    vars->gpu_mForcez,
                                                    sc_coul,
                                                    sc_sigma_6,
                                                    sc_alpha,
                                                    sc_power,
                                                    vars->gpu_rMin,
                                                    vars->gpu_rMaxSq,
                                                    vars->gpu_expConst,
                                                    box);
  BoxForceLJGPU <<< blocksPerGrid, threadsPerBlock, 0, vars->streams[0]>>>(gpu_cellStartIndex,
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
                                                    vars->gpu_aForcex,
                                                    vars->gpu_aForcey,
                                                    vars->gpu_aForcez,
                                                    vars->gpu_mForcex,
                                                    vars->gpu_mForcey,
                                                    vars->gpu_mForcez,
                                                    sc_coul,
                                                    sc_sigma_6,
                                                    sc_alpha,
                                                    sc_power,
                                                    vars->gpu_rMin,
                                                    vars->gpu_rMaxSq,
                                                    vars->gpu_expConst,
                                                    box);


  cudaDeviceSynchronize();
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

  CubDebugExit(cudaMemcpy(aForcex, vars->gpu_aForcex,
                          sizeof(double) * atomCount,
                          cudaMemcpyDeviceToHost));
  CubDebugExit(cudaMemcpy(aForcey, vars->gpu_aForcey,
                          sizeof(double) * atomCount,
                          cudaMemcpyDeviceToHost));
  CubDebugExit(cudaMemcpy(aForcez, vars->gpu_aForcez,
                          sizeof(double) * atomCount,
                          cudaMemcpyDeviceToHost));
  CubDebugExit(cudaMemcpy(mForcex, vars->gpu_mForcex,
                          sizeof(double) * molCount,
                          cudaMemcpyDeviceToHost));
  CubDebugExit(cudaMemcpy(mForcey, vars->gpu_mForcey,
                          sizeof(double) * molCount,
                          cudaMemcpyDeviceToHost));
  CubDebugExit(cudaMemcpy(mForcez, vars->gpu_mForcez,
                          sizeof(double) * molCount,
                          cudaMemcpyDeviceToHost));

  cudaFree(gpu_particleCharge);
  cudaFree(gpu_particleKind);
  cudaFree(gpu_particleMol);
  cudaFree(gpu_REn);
  cudaFree(gpu_LJEn);
  cudaFree(gpu_final_REn);
  cudaFree(gpu_final_LJEn);
  cudaFree(gpu_neighborList);
  cudaFree(gpu_cellStartIndex);
  checkLastErrorCUDA(__FILE__, __LINE__);
}

void CallVirialReciprocalGPU(VariablesCUDA *vars,
                             XYZArray const &currentCoords,
                             XYZArray const &currentCOMDiff,
                             vector<double> &particleCharge,
                             double &rT11,
                             double &rT12,
                             double &rT13,
                             double &rT22,
                             double &rT23,
                             double &rT33,
                             uint imageSize,
                             double constVal,
                             uint box)
{
  int atomNumber = currentCoords.Count();
  int blocksPerGrid, threadsPerBlock;
  double *gpu_particleCharge;
  double *gpu_final_value;

  cudaMalloc((void**) &gpu_particleCharge,
             particleCharge.size() * sizeof(double));
  cudaMalloc((void**) &gpu_final_value, sizeof(double));

  cudaMemcpy(vars->gpu_x, currentCoords.x, atomNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_y, currentCoords.y, atomNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_z, currentCoords.z, atomNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_dx, currentCOMDiff.x, atomNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_dy, currentCOMDiff.y, atomNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_dz, currentCOMDiff.z, atomNumber * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_particleCharge, &particleCharge[0],
             particleCharge.size() * sizeof(double),
             cudaMemcpyHostToDevice);

  // Run the kernel...
  threadsPerBlock = 256;
  blocksPerGrid = (int)(imageSize / threadsPerBlock) + 1;
  VirialReciprocalGPU <<< blocksPerGrid,
                      threadsPerBlock>>>(vars->gpu_x,
                                         vars->gpu_y,
                                         vars->gpu_z,
                                         vars->gpu_dx,
                                         vars->gpu_dy,
                                         vars->gpu_dz,
                                         vars->gpu_kxRef[box],
                                         vars->gpu_kyRef[box],
                                         vars->gpu_kzRef[box],
                                         vars->gpu_prefactRef[box],
                                         vars->gpu_hsqrRef[box],
                                         vars->gpu_sumRref[box],
                                         vars->gpu_sumIref[box],
                                         gpu_particleCharge,
                                         vars->gpu_rT11,
                                         vars->gpu_rT12,
                                         vars->gpu_rT13,
                                         vars->gpu_rT22,
                                         vars->gpu_rT23,
                                         vars->gpu_rT33,
                                         constVal,
                                         imageSize,
                                         atomNumber);
  cudaDeviceSynchronize();
  checkLastErrorCUDA(__FILE__, __LINE__);
  // ReduceSum // Virial of Reciprocal
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT11,
                    gpu_final_value, imageSize);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT11,
                    gpu_final_value, imageSize);
  cudaMemcpy(&rT11, gpu_final_value, sizeof(double),
             cudaMemcpyDeviceToHost);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT12,
                    gpu_final_value, imageSize);
  cudaMemcpy(&rT12, gpu_final_value, sizeof(double),
             cudaMemcpyDeviceToHost);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT13,
                    gpu_final_value, imageSize);
  cudaMemcpy(&rT13, gpu_final_value, sizeof(double),
             cudaMemcpyDeviceToHost);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT22,
                    gpu_final_value, imageSize);
  cudaMemcpy(&rT22, gpu_final_value, sizeof(double),
             cudaMemcpyDeviceToHost);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT23,
                    gpu_final_value, imageSize);
  cudaMemcpy(&rT23, gpu_final_value, sizeof(double),
             cudaMemcpyDeviceToHost);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT33,
                    gpu_final_value, imageSize);
  cudaMemcpy(&rT33, gpu_final_value, sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaFree(gpu_particleCharge);
  cudaFree(gpu_final_value);
  cudaFree(d_temp_storage);
}

__global__ void BoxInterForceGPU(int *gpu_cellStartIndex,
                                 int *gpu_cellVector,
                                 int *gpu_neighborList,
                                 int numberOfCells,
                                 int atomNumber,
                                 int *gpu_mapParticleToCell,
                                 double *gpu_x,
                                 double *gpu_y,
                                 double *gpu_z,
                                 double *gpu_comx,
                                 double *gpu_comy,
                                 double *gpu_comz,
                                 double xAxes,
                                 double yAxes,
                                 double zAxes,
                                 bool electrostatic,
                                 double *gpu_particleCharge,
                                 int *gpu_particleKind,
                                 int *gpu_particleMol,
                                 double *gpu_rT11,
                                 double *gpu_rT12,
                                 double *gpu_rT13,
                                 double *gpu_rT22,
                                 double *gpu_rT23,
                                 double *gpu_rT33,
                                 double *gpu_vT11,
                                 double *gpu_vT12,
                                 double *gpu_vT13,
                                 double *gpu_vT22,
                                 double *gpu_vT23,
                                 double *gpu_vT33,
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
                                 double *gpu_cell_x,
                                 double *gpu_cell_y,
                                 double *gpu_cell_z,
                                 double *gpu_Invcell_x,
                                 double *gpu_Invcell_y,
                                 double *gpu_Invcell_z,
                                 int *gpu_nonOrth,
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
  double virX, virY, virZ;
  double pRF = 0.0, qi_qj, pVF = 0.0;
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  //tensors for VDW and real part of electrostatic
  gpu_vT11[threadID] = 0.0, gpu_vT22[threadID] = 0.0, gpu_vT33[threadID] = 0.0;
  gpu_rT11[threadID] = 0.0, gpu_rT22[threadID] = 0.0, gpu_rT33[threadID] = 0.0;
  // extra tensors reserved for later on
  gpu_vT12[threadID] = 0.0, gpu_vT13[threadID] = 0.0, gpu_vT23[threadID] = 0.0;
  gpu_rT12[threadID] = 0.0, gpu_rT13[threadID] = 0.0, gpu_rT23[threadID] = 0.0;
  double diff_comx, diff_comy, diff_comz;
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
      if(InRcutGPU(distSq, virX, virY, virZ, gpu_x[currentParticle],
                  gpu_y[currentParticle], gpu_z[currentParticle],
                  gpu_x[neighborParticle], gpu_y[neighborParticle],
                  gpu_z[neighborParticle], xAxes, yAxes, zAxes, xAxes / 2.0,
                  yAxes / 2.0, zAxes / 2.0, cutoff, gpu_nonOrth[0],
                  gpu_cell_x, gpu_cell_y, gpu_cell_z, gpu_Invcell_x, gpu_Invcell_y,
                  gpu_Invcell_z)) {
        diff_comx = gpu_comx[gpu_particleMol[currentParticle]] -
                    gpu_comx[gpu_particleMol[neighborParticle]];
        diff_comy = gpu_comy[gpu_particleMol[currentParticle]] -
                    gpu_comy[gpu_particleMol[neighborParticle]];
        diff_comz = gpu_comz[gpu_particleMol[currentParticle]] -
                    gpu_comz[gpu_particleMol[neighborParticle]];

        diff_comx = MinImageSignedGPU(diff_comx, xAxes, xAxes / 2.0);
        diff_comy = MinImageSignedGPU(diff_comy, yAxes, yAxes / 2.0);
        diff_comz = MinImageSignedGPU(diff_comz, zAxes, zAxes / 2.0);

        if(electrostatic) {
          qi_qj = gpu_particleCharge[currentParticle] *
                  gpu_particleCharge[neighborParticle];
          pRF = CalcCoulombForceGPU(distSq, qi_qj, gpu_VDW_Kind[0], gpu_ewald[0],
                                    gpu_isMartini[0], gpu_alpha[box],
                                    gpu_rCutCoulomb[box], gpu_diElectric_1[0],
                                    gpu_sigmaSq, sc_coul, sc_sigma_6, sc_alpha,
                                    sc_power, gpu_count[0],
                                    gpu_particleKind[currentParticle],
                                    gpu_particleKind[neighborParticle]);

          gpu_rT11[threadID] += pRF * (virX * diff_comx);
          gpu_rT22[threadID] += pRF * (virY * diff_comy);
          gpu_rT33[threadID] += pRF * (virZ * diff_comz);

          //extra tensor calculations
          gpu_rT12[threadID] += pRF * (0.5 * (virX * diff_comy + virY * diff_comx));
          gpu_rT13[threadID] += pRF * (0.5 * (virX * diff_comz + virZ * diff_comx));
          gpu_rT23[threadID] += pRF * (0.5 * (virY * diff_comz + virZ * diff_comy));
        }

        pVF = CalcEnForceGPU(distSq, gpu_particleKind[currentParticle],
                            gpu_particleKind[neighborParticle],
                            gpu_sigmaSq, gpu_n, gpu_epsilon_Cn, gpu_rCut[0],
                            gpu_rOn[0], gpu_isMartini[0], gpu_VDW_Kind[0],
                            gpu_count[0], sc_sigma_6,
                            sc_alpha, sc_power, gpu_rMin, gpu_rMaxSq,
                            gpu_expConst);

        gpu_vT11[threadID] += pVF * (virX * diff_comx);
        gpu_vT22[threadID] += pVF * (virY * diff_comy);
        gpu_vT33[threadID] += pVF * (virZ * diff_comz);

        //extra tensor calculations
        gpu_vT12[threadID] += pVF * (0.5 * (virX * diff_comy + virY * diff_comx));
        gpu_vT13[threadID] += pVF * (0.5 * (virX * diff_comz + virZ * diff_comx));
        gpu_vT23[threadID] += pVF * (0.5 * (virY * diff_comz + virZ * diff_comy));
      }
    }
  }
}

__global__ void BoxForceLJGPU(int *gpu_cellStartIndex,
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
                            double *gpu_aForcex,
                            double *gpu_aForcey,
                            double *gpu_aForcez,
                            double *gpu_mForcex,
                            double *gpu_mForcey,
                            double *gpu_mForcez,
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
  double virX = 0.0, virY = 0.0, virZ = 0.0;
  double forceLJx = 0.0, forceLJy = 0.0, forceLJz = 0.0;
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  gpu_REn[threadID] = 0.0;
  gpu_LJEn[threadID] = 0.0;
  double cutoff = fmax(gpu_rCut[0], gpu_rCutCoulomb[box]);

  int currentCell = blockIdx.x / 27;
  int nCellIndex = blockIdx.x;
  int neighborCell = gpu_neighborList[nCellIndex];

  if(currentCell > neighborCell) return;

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

    if(currentParticle != neighborParticle) {
      // Check if they are within rcut
      if(InRcutGPU(distSq, virX, virY, virZ, gpu_x[currentParticle],
        gpu_y[currentParticle], gpu_z[currentParticle],
        gpu_x[neighborParticle], gpu_y[neighborParticle],
        gpu_z[neighborParticle], xAxes, yAxes, zAxes, xAxes / 2.0,
        yAxes / 2.0, zAxes / 2.0, cutoff, gpu_nonOrth[0], gpu_cell_x,
        gpu_cell_y, gpu_cell_z, gpu_Invcell_x, gpu_Invcell_y,
        gpu_Invcell_z)) {
        // gpu_LJEn[threadID] += CalcEnGPU(distSq,
        //                       gpu_particleKind[currentParticle],
        //                       gpu_particleKind[neighborParticle],
        //                       gpu_sigmaSq, gpu_n, gpu_epsilon_Cn,
        //                       gpu_VDW_Kind[0], gpu_isMartini[0],
        //                       gpu_rCut[0], gpu_rOn[0], gpu_count[0],
        //                       sc_sigma_6, sc_alpha, sc_power, gpu_rMin,
        //                       gpu_rMaxSq, gpu_expConst);
        // double pVF = CalcEnForceGPU(distSq, gpu_particleKind[currentParticle],
        //                     gpu_particleKind[neighborParticle],
        //                     gpu_sigmaSq, gpu_n, gpu_epsilon_Cn,
        //                     gpu_rCut[0], gpu_rOn[0], gpu_isMartini[0],
        //                     gpu_VDW_Kind[0], gpu_count[0], sc_sigma_6,
        //                     sc_alpha, sc_power, gpu_rMin, gpu_rMaxSq,
        //                     gpu_expConst);
        int index = FlatIndexGPU(gpu_particleKind[currentParticle], gpu_particleKind[neighborParticle], gpu_count[0]);
        double rRat2 = gpu_sigmaSq[index] / distSq;
        double rRat4 = rRat2 * rRat2;
        double attract = rRat4 * rRat2;
        double repulse = pow(rRat2, gpu_n[index] / 2.0);
        gpu_LJEn[threadID] += gpu_epsilon_Cn[index] * (repulse - attract);
        double pVF = gpu_epsilon_Cn[index] * 6.0 *
          ((gpu_n[index] / 6.0) * repulse - attract) / distSq;

        forceLJx = virX * pVF;
        forceLJy = virY * pVF;
        forceLJz = virZ * pVF;

        atomicAdd(&gpu_aForcex[currentParticle], forceLJx);
        atomicAdd(&gpu_aForcey[currentParticle], forceLJy);
        atomicAdd(&gpu_aForcez[currentParticle], forceLJz);
        atomicAdd(&gpu_aForcex[neighborParticle], -1.0 * (forceLJx));
        atomicAdd(&gpu_aForcey[neighborParticle], -1.0 * (forceLJy));
        atomicAdd(&gpu_aForcez[neighborParticle], -1.0 * (forceLJz));

        atomicAdd(&gpu_mForcex[gpu_particleMol[currentParticle]],
          forceLJx);
        atomicAdd(&gpu_mForcey[gpu_particleMol[currentParticle]],
          forceLJy);
        atomicAdd(&gpu_mForcez[gpu_particleMol[currentParticle]],
          forceLJz);
        atomicAdd(&gpu_mForcex[gpu_particleMol[neighborParticle]],
          -1.0 * (forceLJx));
        atomicAdd(&gpu_mForcey[gpu_particleMol[neighborParticle]],
          -1.0 * (forceLJy));
        atomicAdd(&gpu_mForcez[gpu_particleMol[neighborParticle]],
          -1.0 * (forceLJz));
      }
    }
  }
}

__global__ void BoxForceRealGPU(int *gpu_cellStartIndex,
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
  double *gpu_aForcex,
  double *gpu_aForcey,
  double *gpu_aForcez,
  double *gpu_mForcex,
  double *gpu_mForcey,
  double *gpu_mForcez,
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
  double forceRealx = 0.0, forceRealy = 0.0, forceRealz = 0.0;
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  gpu_REn[threadID] = 0.0;
  gpu_LJEn[threadID] = 0.0;
  double cutoff = fmax(gpu_rCut[0], gpu_rCutCoulomb[box]);

  int currentCell = blockIdx.x / 27;
  int nCellIndex = blockIdx.x;
  int neighborCell = gpu_neighborList[nCellIndex];

  if(currentCell > neighborCell) return;

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

    if(currentParticle != neighborParticle) {
      // Check if they are within rcut
      if(InRcutGPU(distSq, virX, virY, virZ, gpu_x[currentParticle],
      gpu_y[currentParticle], gpu_z[currentParticle],
      gpu_x[neighborParticle], gpu_y[neighborParticle],
      gpu_z[neighborParticle], xAxes, yAxes, zAxes, xAxes / 2.0,
      yAxes / 2.0, zAxes / 2.0, cutoff, gpu_nonOrth[0], gpu_cell_x,
      gpu_cell_y, gpu_cell_z, gpu_Invcell_x, gpu_Invcell_y,
      gpu_Invcell_z)) {
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
      double coulombVir = CalcCoulombForceGPU(distSq, qi_qj_fact,
                        gpu_VDW_Kind[0], gpu_ewald[0],
                        gpu_isMartini[0],
                        gpu_alpha[box],
                        gpu_rCutCoulomb[box],
                        gpu_diElectric_1[0],
                        gpu_sigmaSq, sc_coul, sc_sigma_6,
                        sc_alpha, sc_power,
                        gpu_count[0],
                        gpu_particleKind[currentParticle],
                        gpu_particleKind[neighborParticle]);
      forceRealx = virX * coulombVir;
      forceRealy = virY * coulombVir;
      forceRealz = virZ * coulombVir;

      atomicAdd(&gpu_aForcex[currentParticle], forceRealx);
      atomicAdd(&gpu_aForcey[currentParticle], forceRealy);
      atomicAdd(&gpu_aForcez[currentParticle], forceRealz);
      atomicAdd(&gpu_aForcex[neighborParticle], -1.0 * (forceRealx));
      atomicAdd(&gpu_aForcey[neighborParticle], -1.0 * (forceRealy));
      atomicAdd(&gpu_aForcez[neighborParticle], -1.0 * (forceRealz));

      atomicAdd(&gpu_mForcex[gpu_particleMol[currentParticle]],
      forceRealx);
      atomicAdd(&gpu_mForcey[gpu_particleMol[currentParticle]],
      forceRealy);
      atomicAdd(&gpu_mForcez[gpu_particleMol[currentParticle]],
      forceRealz);
      atomicAdd(&gpu_mForcex[gpu_particleMol[neighborParticle]],
      -1.0 * (forceRealx));
      atomicAdd(&gpu_mForcey[gpu_particleMol[neighborParticle]],
      -1.0 * (forceRealy));
      atomicAdd(&gpu_mForcez[gpu_particleMol[neighborParticle]],
      -1.0 * (forceRealz));
      }
    }
  }
}

__global__ void VirialReciprocalGPU(double *gpu_x,
                                    double *gpu_y,
                                    double *gpu_z,
                                    double *gpu_comDx,
                                    double *gpu_comDy,
                                    double *gpu_comDz,
                                    double *gpu_kxRef,
                                    double *gpu_kyRef,
                                    double *gpu_kzRef,
                                    double *gpu_prefactRef,
                                    double *gpu_hsqrRef,
                                    double *gpu_sumRref,
                                    double *gpu_sumIref,
                                    double *gpu_particleCharge,
                                    double *gpu_rT11,
                                    double *gpu_rT12,
                                    double *gpu_rT13,
                                    double *gpu_rT22,
                                    double *gpu_rT23,
                                    double *gpu_rT33,
                                    double constVal,
                                    uint imageSize,
                                    uint atomNumber)
{
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  if(threadID >= imageSize)
    return;

  double factor, arg;
  int i;
  factor = gpu_prefactRef[threadID] * (gpu_sumRref[threadID] *
                                       gpu_sumRref[threadID] +
                                       gpu_sumIref[threadID] *
                                       gpu_sumIref[threadID]);
  gpu_rT11[threadID] = factor * (1.0 - 2.0 *
                                 (constVal + 1.0 / gpu_hsqrRef[threadID]) *
                                 gpu_kxRef[threadID] * gpu_kxRef[threadID]);
  gpu_rT12[threadID] = factor * (-2.0 *
                                 (constVal + 1.0 / gpu_hsqrRef[threadID]) *
                                 gpu_kxRef[threadID] * gpu_kyRef[threadID]);
  gpu_rT13[threadID] = factor * (-2.0 *
                                 (constVal + 1.0 / gpu_hsqrRef[threadID]) *
                                 gpu_kxRef[threadID] * gpu_kzRef[threadID]);
  gpu_rT22[threadID] = factor * (1.0 - 2.0 *
                                 (constVal + 1.0 / gpu_hsqrRef[threadID]) *
                                 gpu_kyRef[threadID] * gpu_kyRef[threadID]);
  gpu_rT23[threadID] = factor * (-2.0 *
                                 (constVal + 1.0 / gpu_hsqrRef[threadID]) *
                                 gpu_kyRef[threadID] * gpu_kzRef[threadID]);
  gpu_rT33[threadID] = factor * (1.0 - 2.0 *
                                 (constVal + 1.0 / gpu_hsqrRef[threadID]) *
                                 gpu_kzRef[threadID] * gpu_kzRef[threadID]);

  //Intramolecular part
  for(i = 0; i < atomNumber; i++) {
    arg = DotProductGPU(gpu_kxRef[threadID], gpu_kyRef[threadID],
                        gpu_kzRef[threadID], gpu_x[i], gpu_y[i], gpu_z[i]);

    factor = gpu_prefactRef[threadID] * 2.0 *
             (gpu_sumIref[threadID] * cos(arg) - gpu_sumRref[threadID] * sin(arg)) *
             gpu_particleCharge[i];

    gpu_rT11[threadID] += factor * (gpu_kxRef[threadID] * gpu_comDx[i]);
    gpu_rT12[threadID] += factor * 0.5 * (gpu_kxRef[threadID] * gpu_comDy[i] +
                                          gpu_kyRef[threadID] * gpu_comDx[i]);
    gpu_rT13[threadID] += factor * 0.5 * (gpu_kxRef[threadID] * gpu_comDz[i] +
                                          gpu_kzRef[threadID] * gpu_comDx[i]);
    gpu_rT22[threadID] += factor * (gpu_kyRef[threadID] * gpu_comDy[i]);
    gpu_rT13[threadID] += factor * 0.5 * (gpu_kyRef[threadID] * gpu_comDz[i] +
                                          gpu_kzRef[threadID] * gpu_comDy[i]);
    gpu_rT33[threadID] += factor * (gpu_kzRef[threadID] * gpu_comDz[i]);
  }
}

__device__ double CalcEnForceGPU(double distSq, int kind1, int kind2,
                                 double *gpu_sigmaSq, double *gpu_n,
                                 double *gpu_epsilon_Cn, double gpu_rCut,
                                 double gpu_rOn, int gpu_isMartini,
                                 int gpu_VDW_Kind, int gpu_count,
                                 double sc_sigma_6,
                                 double sc_alpha, uint sc_power,
                                 double *gpu_rMin, double *gpu_rMaxSq,
                                 double *gpu_expConst)
{
  if((gpu_rCut * gpu_rCut) < distSq) {
    return 0.0;
  }

  int index = FlatIndexGPU(kind1, kind2, gpu_count);
  return CalcVirParticleGPU(distSq, index, gpu_sigmaSq, gpu_n,
                            gpu_epsilon_Cn, sc_sigma_6,
                            sc_alpha, sc_power);
}

//ElectroStatic Calculation
//**************************************************************//
__device__ double CalcCoulombVirParticleGPU(double distSq, double qi_qj,
    int gpu_ewald, double gpu_alpha,
    int index, double *gpu_sigmaSq,
    bool sc_coul, double sc_sigma_6,
    double sc_alpha, uint sc_power)
{
  if(gpu_ewald) {
    double dist = sqrt(distSq);
    double constValue = 2.0 * gpu_alpha / sqrt(M_PI);
    double expConstValue = exp(-1.0 * gpu_alpha * gpu_alpha * distSq);
    double temp = 1.0 - erf(gpu_alpha * dist);
    return qi_qj * (temp / dist + constValue * expConstValue) / distSq;
  } else {
    double dist = sqrt(distSq);
    double result = qi_qj / (distSq * dist);
    return result;
  }
}

//VDW Calculation
//*****************************************************************//
__device__ double CalcVirParticleGPU(double distSq, int index,
                                     double *gpu_sigmaSq, double *gpu_n,
                                     double *gpu_epsilon_Cn, double sc_sigma_6,
                                     double sc_alpha, uint sc_power)
{
  double rNeg2 = 1.0 / distSq;
  double rRat2 = gpu_sigmaSq[index] * rNeg2;
  double rRat4 = rRat2 * rRat2;
  double attract = rRat4 * rRat2;
  double repulse = pow(rRat2, gpu_n[index] / 2.0);
  return gpu_epsilon_Cn[index] * 6.0 *
         ((gpu_n[index] / 6.0) * repulse - attract) * rNeg2;
}

#endif
