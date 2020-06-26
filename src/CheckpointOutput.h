/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.60
Copyright (C) 2018  GOMC Group
A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/
#pragma once

#include "OutputAbstracts.h"
#include "MoveSettings.h"
#include "Coordinates.h"
#include "MoveBase.h"
#include <iostream>
#include "GOMC_Config.h"

class CheckpointOutput : public OutputableBase
{
public:
  CheckpointOutput(System & sys, StaticVals const& statV);

  ~CheckpointOutput()
  {
    if(outputFile)
      fclose(outputFile);
  }

  virtual void DoOutput(const ulong step);
  virtual void Init(pdb_setup::Atoms const& atoms,
                    config_setup::Output const& output);
  virtual void Sample(const ulong step) {}
  virtual void Output(const ulong step)
  {
    if(!enableOutCheckpoint) {
      return;
    }

    if((step + 1) % stepsPerCheckpoint == 0) {
      DoOutput(step);
    }
  }

private:
  MoveSettings & moveSetRef;
  MoleculeLookup & molLookupRef;
  BoxDimensions & boxDimRef;
  PRNG & prngRef;
  Coordinates & coordCurrRef;
  std::string filename;

  bool enableOutCheckpoint;
  FILE* outputFile;
  ulong stepsPerCheckpoint;

  void openOutputFile();
  void printStepNumber(ulong step);
  void printRandomNumbers();
  void printCoordinates();
  void printMoleculeLookupData();
  void printMoveSettingsData();
  void printBoxDimensionsData();

  void printVector3DDouble(std::vector< std::vector< std::vector<double> > > data);
  void printVector3DUint(std::vector< std::vector< std::vector<uint> > > data);
  void printVector2DUint(std::vector< std::vector< uint > > data);
  void printVector1DDouble(std::vector< double > data);
  void outputDoubleIn8Chars(double data);
  void outputUintIn8Chars(uint32_t data);

};
