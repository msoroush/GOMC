/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.50
Copyright (C) 2018  GOMC Group
A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/
#ifndef MULTIPARTICLE_H
#define MULTIPARTICLE_H

#include "MoveBase.h"
#include "System.h"
#include "StaticVals.h"
#include "TransformMatrix.h"
#include "Random123Wrapper.h"
#ifdef GOMC_CUDA
#include "TransformParticlesCUDA.cuh"
#include "VariablesCUDA.cuh"
#endif
#include <cmath>

#define MIN_FORCE 1E-12
#define MAX_FORCE 30

class MultiParticle : public MoveBase
{
public:
  MultiParticle(System &sys, StaticVals const& statV);

  virtual uint Prep(const double subDraw, const double movPerc);
  virtual void CalcEn();
  virtual uint Transform();
  virtual void Accept(const uint rejectState, const uint step);
  virtual void PrintAcceptKind();

private:
  uint bPick;
  uint typePick;
  double lambda;
  bool initMol[BOX_TOTAL];
  SystemPotential sysPotNew;
  XYZArray molTorqueRef;
  XYZArray molTorqueNew;
  XYZArray atomForceRecNew;
  XYZArray molForceRecNew;
  XYZArray t_k;
  XYZArray r_k;
  Coordinates newMolsPos;
  COM newCOMs;
  vector<uint> moleculeIndex;
  uint moveType;
  const MoleculeLookup& molLookup;
  Random123Wrapper &r123wrapper;
  const Molecules& mols;
#ifdef GOMC_CUDA
  VariablesCUDA *cudaVars;
  std::vector<int> particleMol;
#endif

  long double GetCoeff();
  void CalculateTrialDistRot();
  void RotateForceBiased(std::vector<uint> molIndeces);
  void TranslateForceBiased(std::vector<uint> molIndeces);
  void SetMolInBox(uint box);
  double CalculateWRatio(XYZ const &lb_new, XYZ const &lb_old, XYZ const &k,
                         double max);
};

inline MultiParticle::MultiParticle(System &sys, StaticVals const &statV) :
  MoveBase(sys, statV),
  newMolsPos(sys.boxDimRef, newCOMs, sys.molLookupRef, sys.prng, statV.mol),
  newCOMs(sys.boxDimRef, newMolsPos, sys.molLookupRef, statV.mol),
  molLookup(sys.molLookup), r123wrapper(sys.r123wrapper), mols(statV.mol)
{
  molTorqueNew.Init(sys.com.Count());
  molTorqueRef.Init(sys.com.Count());
  atomForceRecNew.Init(sys.coordinates.Count());
  molForceRecNew.Init(sys.com.Count());

  t_k.Init(sys.com.Count());
  r_k.Init(sys.com.Count());
  newMolsPos.Init(sys.coordinates.Count());
  newCOMs.Init(sys.com.Count());

  // set default value for r_max, t_max, and lambda
  // the value of lambda is based on the paper
  lambda = 0.5;
  for(uint b = 0; b < BOX_TOTAL; b++) {
    initMol[b] = false;
  }

#ifdef GOMC_CUDA
  cudaVars = sys.statV.forcefield.particles->getCUDAVars();

  uint maxAtomInMol = 0;
  for(uint m = 0; m < mols.count; ++m) {
    const MoleculeKind& molKind = mols.GetKind(m);
    if(molKind.NumAtoms() > maxAtomInMol)
      maxAtomInMol = molKind.NumAtoms();
    for(uint a = 0; a < molKind.NumAtoms(); ++a) {
      particleMol.push_back(m);
    }
  }
#endif
}

inline void MultiParticle::PrintAcceptKind()
{
  printf("%-37s", "% Accepted MultiParticle ");
  for(uint b = 0; b < BOX_TOTAL; b++) {
    printf("%10.5f ", 100.0 * moveSetRef.GetAccept(b, mv::MULTIPARTICLE));
  }
  std::cout << std::endl;
}


inline void MultiParticle::SetMolInBox(uint box)
{
  // NEED to check if atom is not fixed!
#if ENSEMBLE == GCMC || ENSEMBLE == GEMC
  moleculeIndex.clear();
  MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
  MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
  while(thisMol != end) {
    //Make sure this molecule is not fixed in its position
    if(!molLookup.IsFix(*thisMol)) {
      moleculeIndex.push_back(*thisMol);
    }
    thisMol++;
  }
#else
  if(!initMol[box]) {
    moleculeIndex.clear();
    MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
    MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
    while(thisMol != end) {
      //Make sure this molecule is not fixed in its position
      if(!molLookup.IsFix(*thisMol)) {
        moleculeIndex.push_back(*thisMol);
      }
      thisMol++;
    }
  }
#endif
  initMol[box] = true;
}

inline uint MultiParticle::Prep(const double subDraw, const double movPerc)
{
  uint state = mv::fail_state::NO_FAIL;
#if ENSEMBLE == GCMC
  bPick = mv::BOX0;
#else
  prng.PickBox(bPick, subDraw, movPerc);
#endif

  SetMolInBox(bPick);
  // In each step, we perform either:
  // 1- All displacement move.
  // 2- All rotation move.
  uint length = molRef.GetKind(moleculeIndex[0]).NumAtoms();
  if(length == 1) moveType = mp::MPALLDISPLACE;
  else moveType = prng.randIntExc(mp::MPTOTALTYPES);

  if(moveSetRef.GetSingleMoveAccepted()) {
    //Calculate force for long range electrostatic using old position
    calcEwald->BoxForceReciprocal(sysPotRef, coordCurrRef, atomForceRecRef,
                                  molForceRecRef, bPick);

    //calculate short range energy and force for old positions
    calcEnRef.BoxForce(sysPotRef, coordCurrRef, atomForceRef, molForceRef,
                       boxDimRef, bPick);

    if(typePick != mp::MPALLDISPLACE) {
      //Calculate Torque for old positions
      calcEnRef.CalculateTorque(moleculeIndex, coordCurrRef, comCurrRef,
                                atomForceRef, atomForceRecRef, molTorqueRef,
                                moveType, bPick);
    }
  }
#ifndef GOMC_CUDA // if GPU is enabled this part will be combined with Transform section
  CalculateTrialDistRot();
#endif
  coordCurrRef.CopyRange(newMolsPos, 0, 0, coordCurrRef.Count());
  comCurrRef.CopyRange(newCOMs, 0, 0, comCurrRef.Count());
  return state;
}

inline uint MultiParticle::Transform()
{
  // Based on the reference force decided whether to displace or rotate each
  // individual particle.
  uint state = mv::fail_state::NO_FAIL;

#ifdef GOMC_CUDA
  // This kernel will calculate translation/rotation amout + shifting/rotating
  if(moveType == mp::MPALLROTATE) {
    double r_max = moveSetRef.GetRMAX(bPick);
    CallRotateParticlesGPU(cudaVars, moleculeIndex, moveType, r_max,
                           molTorqueRef.x, molTorqueRef.y, molTorqueRef.z,
                           r123wrapper.GetStep(), r123wrapper.GetSeedValue(),
                           particleMol, atomForceRecNew.Count(),
                           molForceRecNew.Count(), boxDimRef.GetAxis(bPick).x,
                           boxDimRef.GetAxis(bPick).y, boxDimRef.GetAxis(bPick).z,
                           newMolsPos, newCOMs);
  } else {
    double t_max = moveSetRef.GetTMAX(bPick);
    CallTranslateParticlesGPU(cudaVars, moleculeIndex, moveType, t_max,
                              molForceRef.x, molForceRef.y, molForceRef.z,
                              r123wrapper.GetStep(), r123wrapper.GetSeedValue(),
                              particleMol, atomForceRecNew.Count(),
                              molForceRecNew.Count(), boxDimRef.GetAxis(bPick).x,
                              boxDimRef.GetAxis(bPick).y, boxDimRef.GetAxis(bPick).z,
                              newMolsPos, newCOMs);
  }
#else
  // move particles according to force and torque and store them in the new pos
  if(moveType == mp::MPALLROTATE) {
    // rotate
    RotateForceBiased(moleculeIndex);
  } else {
    // displacement
    TranslateForceBiased(moleculeIndex);
  }
#endif
  return state;
}

inline void MultiParticle::CalcEn()
{
  // Calculate the new force and energy and we will compare that to the
  // reference values in Accept() function
  cellList.GridAll(boxDimRef, newMolsPos, molLookup);

  //back up cached fourier term
  calcEwald->backupMolCache();
  //setup reciprocate vectors for new positions
  //calcEwald->BoxReciprocalSetup(bPick, newMolsPos);

  sysPotNew = sysPotRef;
  //calculate short range energy and force
  sysPotNew = calcEnRef.BoxForce(sysPotNew, newMolsPos, atomForceNew,
                                 molForceNew, boxDimRef, bPick);
  //calculate long range of new electrostatic energy
  //sysPotNew.boxEnergy[bPick].recip = calcEwald->BoxReciprocal(bPick);
  //Calculate long range of new electrostatic force
  calcEwald->BoxForceReciprocal(sysPotNew, newMolsPos,
                                atomForceRecNew, molForceRecNew,
                                bPick);

  if(typePick != mp::MPALLDISPLACE) {
    //Calculate Torque for new positions
    calcEnRef.CalculateTorque(moleculeIndex, newMolsPos, newCOMs, atomForceNew,
                              atomForceRecNew, molTorqueNew, moveType, bPick);
  }
  sysPotNew.Total();
}

inline double MultiParticle::CalculateWRatio(XYZ const &lb_new, XYZ const &lb_old,
    XYZ const &k, double max)
{
  double w_ratio = 1.0;
  XYZ lbmax = lb_old * max;
  //If we used force to bias the displacement or rotation, we include it
  if(abs(lbmax.x) > MIN_FORCE && abs(lbmax.x) < MAX_FORCE) {
    w_ratio *= lb_new.x * exp(-lb_new.x * k.x) / (2.0 * sinh(lb_new.x * max));
    w_ratio /= lb_old.x * exp(lb_old.x * k.x) / (2.0 * sinh(lb_old.x * max));
  }

  if(abs(lbmax.y) > MIN_FORCE && abs(lbmax.y) < MAX_FORCE) {
    w_ratio *= lb_new.y * exp(-lb_new.y * k.y) / (2.0 * sinh(lb_new.y * max));
    w_ratio /= lb_old.y * exp(lb_old.y * k.y) / (2.0 * sinh(lb_old.y * max));
  }

  if(abs(lbmax.z) > MIN_FORCE && abs(lbmax.z) < MAX_FORCE) {
    w_ratio *= lb_new.z * exp(-lb_new.z * k.z) / (2.0 * sinh(lb_new.z * max));
    w_ratio /= lb_old.z * exp(lb_old.z * k.z) / (2.0 * sinh(lb_old.z * max));
  }

  return w_ratio;
}

inline long double MultiParticle::GetCoeff()
{
  // calculate (w_new->old/w_old->new) and return it.
  XYZ lbf_old, lbf_new; // lambda * BETA * force
  XYZ lbt_old, lbt_new; // lambda * BETA * torque
  long double w_ratio = 1.0;
  double lBeta = lambda * BETA;
  uint m, molNumber;
  double r_max = moveSetRef.GetRMAX(bPick);
  double t_max = moveSetRef.GetTMAX(bPick);
#ifdef _OPENMP
  #pragma omp parallel for default(shared) private(m, molNumber, lbt_old, lbt_new, lbf_old, lbf_new) reduction(*:w_ratio)
#endif
  for(m = 0; m < moleculeIndex.size(); m++) {
    molNumber = moleculeIndex[m];
    if(moveType == mp::MPALLROTATE) {
      // rotate
      lbt_old = molTorqueRef.Get(molNumber) * lBeta;
      lbt_new = molTorqueNew.Get(molNumber) * lBeta;
      w_ratio *= CalculateWRatio(lbt_new, lbt_old, r_k.Get(molNumber), r_max);
    } else {
      // displace
      lbf_old = (molForceRef.Get(molNumber) + molForceRecRef.Get(molNumber)) *
                lBeta;
      lbf_new = (molForceNew.Get(molNumber) + molForceRecNew.Get(molNumber)) *
                lBeta;
      w_ratio *= CalculateWRatio(lbf_new, lbf_old, t_k.Get(molNumber), t_max);
    }
  }

  // In case where force or torque is a large negative number (ex. -800)
  // the exp value becomes inf. In these situations we have to return 0 to
  // reject the move
  // if(!std::isfinite(w_ratio)) {
  //   // This error can be removed later on once we know this part actually works.
  //   std::cout << "w_ratio is not a finite number. Auto-rejecting move.\n";
  //   return 0.0;
  // }
  return w_ratio;
}

inline void MultiParticle::Accept(const uint rejectState, const uint step)
{
  // Here we compare the values of reference and trial and decide whether to
  // accept or reject the move
  long double MPCoeff = GetCoeff();
  double uBoltz = exp(-BETA * (sysPotNew.Total() - sysPotRef.Total()));
  long double accept = MPCoeff * uBoltz;
  cout << "MPCoeff: " << MPCoeff << ", sysPotNew: " << sysPotNew.Total()
       << ", sysPotRef: " << sysPotRef.Total() << ", accept: " << accept <<endl;
  bool result = (rejectState == mv::fail_state::NO_FAIL) && prng() < accept;
  if(result) {
    sysPotRef = sysPotNew;
    swap(coordCurrRef, newMolsPos);
    swap(comCurrRef, newCOMs);
    swap(molForceRef, molForceNew);
    swap(atomForceRef, atomForceNew);
    swap(molForceRecRef, molForceRecNew);
    swap(atomForceRecRef, atomForceRecNew);
    swap(molTorqueRef, molTorqueNew);
    //update reciprocate value
    calcEwald->UpdateRecip(bPick);
  } else {
    cellList.GridAll(boxDimRef, coordCurrRef, molLookup);
    calcEwald->exgMolCache();
  }

  moveSetRef.UpdateMoveSettingMultiParticle(bPick, result, typePick);
  moveSetRef.AdjustMultiParticle(bPick, typePick);

  moveSetRef.Update(mv::MULTIPARTICLE, result, step, bPick);
}

inline void MultiParticle::CalculateTrialDistRot()
{
  uint m, molIndex;
  XYZ num;
  XYZ lbmax;

  if(moveType == mp::MPALLROTATE) { // rotate all
    double r_max = moveSetRef.GetRMAX(bPick);
    XYZ lbt; // lambda * BETA * torque * maxRotation
    for(m = 0; m < moleculeIndex.size(); m++) {
      molIndex = moleculeIndex[m];
      lbt = molTorqueRef.Get(molIndex) * lambda * BETA;
      lbmax = lbt * r_max;

      if(abs(lbmax.x) > MIN_FORCE && abs(lbmax.x) < MAX_FORCE) {
        num.x = log(exp(-1.0 * lbmax.x) + 2 * r123wrapper(m*3+0) * sinh(lbmax.x)) / lbt.x;
      } else {
        double rr = r123wrapper(m*3+0) * 2.0 - 1.0;
        num.x = r_max * rr;
      }

      if(abs(lbmax.y) > MIN_FORCE && abs(lbmax.y) < MAX_FORCE) {
        num.y = log(exp(-1.0 * lbmax.y) + 2 * r123wrapper(m*3+1) * sinh(lbmax.y)) / lbt.y;
      } else {
        double rr = r123wrapper(m*3+1) * 2.0 - 1.0;
        num.y = r_max * rr;
      }

      if(abs(lbmax.z) > MIN_FORCE && abs(lbmax.z) < MAX_FORCE) {
        num.z = log(exp(-1.0 * lbmax.z) + 2 * r123wrapper(m*3+2) * sinh(lbmax.z)) / lbt.z;
      } else {
        double rr = r123wrapper(m*3+2) * 2.0 - 1.0;
        num.z = r_max * rr;
      }

      if(num.Length() >= boxDimRef.axis.Min(bPick)) {
        std::cout << "Trial Displacement exceed half of the box length in Multiparticle move." << endl;
        std::cout << "Trial transform: " << num << endl;
        exit(EXIT_FAILURE);
      } else if (!isfinite(num.Length())) {
        std::cout << "Trial Displacement is not a finite number in Multiparticle move." << endl;
        std::cout << "Trial transform: " << num << endl;
        exit(EXIT_FAILURE);
      }

      r_k.Set(m, num);
    }
  } else if(moveType == mp::MPALLDISPLACE) { // displace all
    double t_max = moveSetRef.GetTMAX(bPick);
    XYZ lbf; // lambda * BETA * force * maxTranslate
    for(m = 0; m < moleculeIndex.size(); m++) {
      molIndex = moleculeIndex[m];
      lbf = (molForceRef.Get(molIndex) + molForceRecRef.Get(molIndex)) *
            lambda * BETA;
      lbmax = lbf * t_max;

      if(abs(lbmax.x) > MIN_FORCE && abs(lbmax.x) < MAX_FORCE) {
        num.x = log(exp(-1.0 * lbmax.x) + 2 * r123wrapper(m*3+0) * sinh(lbmax.x)) / lbf.x;
      } else {
        double rr = r123wrapper(m*3+0) * 2.0 - 1.0;
        num.x = t_max * rr;
      }

      if(abs(lbmax.y) > MIN_FORCE && abs(lbmax.y) < MAX_FORCE) {
        num.y = log(exp(-1.0 * lbmax.y) + 2 * r123wrapper(m*3+1) * sinh(lbmax.y)) / lbf.y;
      } else {
        double rr = r123wrapper(m*3+1) * 2.0 - 1.0;
        num.y = t_max * rr;
      }

      if(abs(lbmax.z) > MIN_FORCE && abs(lbmax.z) < MAX_FORCE) {
        num.z = log(exp(-1.0 * lbmax.z) + 2 * r123wrapper(m*3+2) * sinh(lbmax.z)) / lbf.z;
      } else {
        double rr = r123wrapper(m*3+2) * 2.0 - 1.0;
        num.z = t_max * rr;
      }

      if(num.Length() >= boxDimRef.axis.Min(bPick)) {
        std::cout << "Trial Displacement exceed half of the box length in Multiparticle move." << endl;
        std::cout << "Trial transform: " << num << endl;
        exit(EXIT_FAILURE);
      } else if (!isfinite(num.Length())) {
        std::cout << "Trial Displacement is not a finite number in Multiparticle move." << endl;
        std::cout << "Trial transform: " << num << endl;
        exit(EXIT_FAILURE);
      }

      t_k.Set(m, num);
    }
  }
}

inline void MultiParticle::RotateForceBiased(std::vector<uint> molIndeces)
{
  for(int i=0; i<molIndeces.size(); i++) {
    uint molIndex = molIndeces[i];
    XYZ rot = r_k.Get(molIndex);
    double rotLen = rot.Length();
    RotationMatrix matrix;

    matrix = RotationMatrix::FromAxisAngle(rotLen, rot * (1.0 / rotLen));

    XYZ center = newCOMs.Get(molIndex);
    uint start, stop, len;
    molRef.GetRange(start, stop, len, molIndex);

    // Copy the range into temporary array
    XYZArray temp(len);
    newMolsPos.CopyRange(temp, start, 0, len);
    boxDimRef.UnwrapPBC(temp, bPick, center);

    // Do Rotation
    for(uint p = 0; p < len; p++) {
      temp.Add(p, -center);
      temp.Set(p, matrix.Apply(temp[p]));
      temp.Add(p, center);
    }
    boxDimRef.WrapPBC(temp, bPick);
    // Copy back the result
    temp.CopyRange(newMolsPos, 0, start, len);
  }
}

inline void MultiParticle::TranslateForceBiased(std::vector<uint> molIndeces)
{
  for(int i=0; i<molIndeces.size(); i++) {
    uint molIndex = molIndeces[i];
    XYZ shift = t_k.Get(molIndex);

    XYZ newcom = newCOMs.Get(molIndex);
    newcom += shift;
    newcom = boxDimRef.WrapPBC(newcom, bPick);
    newCOMs.Set(molIndex, newcom);

    uint stop, start, len;
    molRef.GetRange(start, stop, len, molIndex);
    // Copy the range into temporary array
    XYZArray temp(len);
    newMolsPos.CopyRange(temp, start, 0, len);
    //Shift the coordinate and COM
    temp.AddAll(shift);
    //rewrapping
    boxDimRef.WrapPBC(temp, bPick);
    //set the new coordinate
    temp.CopyRange(newMolsPos, 0, start, len);
  }
}

#endif
