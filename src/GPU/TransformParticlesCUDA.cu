#ifdef GOMC_CUDA
#include "TransformParticlesCUDA.cuh"

#define MIN_FORCE 1E-12
#define MAX_FORCE 30

__device__ inline double randomGPU(unsigned int counter, unsigned int step, unsigned int seed) {
  RNG rng;
  RNG::ctr_type c = {{}};
  RNG::ukey_type uk = {{}};
  uk[0] = step;
  uk[1] = seed;
  RNG::key_type k = uk;
  c[0] = counter;
  RNG::ctr_type r = philox4x32(c, k);
  return (double)r[0] / UINT_MAX;
}

__device__ inline double WrapPBC(double &v, double ax) {
  if(v >= ax)
    v -= ax;
  else if(v < 0)
    v += ax;
  return v;
}

void CallTranslateParticlesGPU(VariablesCUDA *vars,
                               vector<uint> &moleculeIndex,
                               uint moveType,
                               double t_max,
                               double *mForcex,
                               double *mForcey,
                               double *mForcez,
                               unsigned int step,
                               unsigned int seed,
                               vector<int> particleMol,
                               int atomCount,
                               double xAxes,
                               double yAxes,
                               double zAxes)
{
  int numberOfMolecules = moleculeIndex.size();
  int threadsPerBlock = 256;
  int blocksPerGrid = (int)(atomCount / threadsPerBlock) + 1;
  double *gpu_mForcex, *gpu_mForcey, *gpu_mForcez;
  int *gpu_particleMol;

  cudaMalloc((void**) &gpu_particleMol, particleMol.size() * sizeof(int)));

  cudaMemcpy(vars->gpu_mForcex, mForcex, numberOfMolecules * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_mForcey, mForcey, numberOfMolecules * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vars->gpu_mForcez, mForcez, numberOfMolecules * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_particleMol, &particleMol[0], particleMol.size() * sizeof(int),
             cudaMemcpyHostToDevice));

  TranslateParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(numberOfMolecules,
                                                               t_max,
                                                               vars->gpu_mForcex,
                                                               vars->gpu_mForcey,
                                                               vars->gpu_mForcez,
                                                               step,
                                                               seed,
                                                               vars->gpu_x,
                                                               vars->gpu_y,
                                                               vars->gpu_z,
                                                               gpu_particleMol,
                                                               atomCount,
                                                               xAxes,
                                                               yAxes,
                                                               zAxes);
}

__global__ void TranslateParticlesKernel(unsigned int numberOfMolecules,
                                         double t_max,
                                         double *molForcex,
                                         double *molForcey,
                                         double *molForcez,
                                         double *molForceRecx,
                                         double *molForceRecy,
                                         double *molForceRecz
                                         unsigned int step,
                                         unsigned int seed,
                                         double *gpu_x,
                                         double *gpu_y,
                                         double *gpu_z
                                         int *gpu_particleMol,
                                         int atomCount,
                                         double xAxes,
                                         double yAxes,
                                         double zAxes)
{
  int atomNumber = blockIdx.x * blockDim.x + threadIdx.x;
  if(atomNumber >= atomCount) return;

  int molIndex = gpu_particleMol[atomNumber];

  // This section calculates the amount of shift
  double lbfx = molForcex[molIndex];
  double lbfy = molForcey[molIndex];
  double lbfz = molForcez[molIndex];
  double lbmaxx = lbfx * t_max;
  double lbmaxy = lbfy * t_max;
  double lbmaxz = lbfz * t_max;

  double shiftx, shifty, shiftz;

  if(abs(lbmaxx) > MIN_FORCE && abs(lbmaxx) < MAX_FORCE) {
    shiftx = log(exp(-1.0 * lbmaxx) + 2 * randomGPU(molIndex * 3, step, seed) * sinh(lbmaxx)) / lbfx;
  } else {
    double rr = randomGPU(molIndex * 3) * 2.0 - 1.0;
    shiftx = t_max * rr;
  }

  if(abs(lbmaxy) > MIN_FORCE && abs(lbmaxy) < MAX_FORCE) {
    shifty = log(exp(-1.0 * lbmaxy) + 2 * randomGPU(molIndex * 3 + 1, step, seed) * sinh(lbmaxx)) / lbfy;
  } else {
    double rr = randomGPU(molIndex * 3 + 1) * 2.0 - 1.0;
    shifty = t_max * rr;
  }

  if(abs(lbmaxz) > MIN_FORCE && abs(lbmaxz) < MAX_FORCE) {
    shiftz = log(exp(-1.0 * lbmaxz) + 2 * randomGPU(molIndex * 3 + 2, step, seed) * sinh(lbmaxz)) / lbfz;
  } else {
    double rr = randomGPU(molIndex * 3 + 2) * 2.0 - 1.0;
    shiftz = t_max * rr;
  }


  // perform the shift on the coordinates
  gpu_x[atomNumber] += shiftx;
  gpu_y[atomNumber] += shifty;
  gpu_z[atomNumber] += shiftz;

  // rewrapping
  WrapPBC(gpu_x[atomNumber], xAxes);
  WrapPBC(gpu_y[atomNumber], yAxes);
  WrapPBC(gpu_z[atomNumber], zAxes);

  // TODO ======================= shift COM as well
}

#endif