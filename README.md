# mos-devkit-py

Python model development kit for MOS.

## Contents

* Linear solver wrappers
  * ``mos.devkit.linalg.LinSolverMUMPS``
  * ``mos.devkit.linalg.LinSolverSUPERLU``
  * ``mos.devkit.linalg.LinSolverUMFPACK``
* Optimization problem classes
  * ``mos.devkit.problem.Problem``
  * ``mos.devkit.problem.LinProblem``
  * ``mos.devkit.problem.QuadProblem``
  * ``mos.devkit.problem.MixIntLinProblem``
* Optimization solvers and wrappers
  * ``mos.devkit.solver.SolverClp``
  * ``mos.devkit.solver.SolverClpCMD``
  * ``mos.devkit.solver.SolverCbc``
  * ``mos.devkit.solver.SolverCbcCMD``
  * ``mos.devkit.solver.SolverCplexCMD``
  * ``mos.devkit.solver.SolverIQP``
  * ``mos.devkit.solver.SolverINLP``
  * ``mos.devkit.solver.SolverIpopt``
  * ``mos.devkit.solver.SolverAugL``
  * ``mos.devkit.solver.SolverNR``
* Algebraic modeling system with sparse automatic symbolic differentiation (in C)
  * ``mos.devkit.model.VariableScalar``
  * ``mos.devkit.model.VariableMatrix``
  * ``mos.devkit.model.VariableDict``
  * ``mos.devkit.model.Model``
  * ``mos.devkit.model.maximize``
  * ``mos.devkit.model.minimize``
  * ``mos.devkit.model.sin``
  * ``mos.devkit.model.cos``
  * ``mos.devkit.model.sum``
  
## Prerequisites

```
pip install -r requirements.txt
```

## Environment Variables

The following environment variables can be used to provide library names, and include and library directories for the following solver interfaces:

* MUMPS: ``MOS_DEVKIT_MUMPS_LIB``, ``MOS_DEVKIT_MUMPS_LIB_DIR``, ``MOS_DEVKIT_MUMPS_INC_DIR``
* IPOPT: ``MOS_DEVKIT_IPOPT_LIB``, ``MOS_DEVKIT_IPOPT_LIB_DIR``, ``MOS_DEVKIT_IPOPT_INC_DIR``
* CLP: ``MOS_DEVKIT_CLP_LIB``, ``MOS_DEVKIT_CLP_LIB_DIR``, ``MOS_DEVKIT_CLP_INC_DIR``
* CBC: ``MOS_DEVKIT_CBC_LIB``, ``MOS_DEVKIT_CBC_LIB_DIR``, ``MOS_DEVKIT_CBC_INC_DIR``

**Sample Configuration on Ubuntu**

```
MOS_DEVKIT_IPOPT_LIB=ipopt
MOS_DEVKIT_IPOPT_INC_DIR=/usr/include/coin
MOS_DEVKIT_IPOPT_LIB_DIR=/usr/lib
MOS_DEVKIT_CLP_LIB=Clp
MOS_DEVKIT_CLP_INC_DIR=/usr/include/coin
MOS_DEVKIT_CLP_LIB_DIR=/usr/lib
MOS_DEVKIT_CBC_LIB=CbcSolver
MOS_DEVKIT_CBC_INC_DIR=/usr/include/coin
MOS_DEVKIT_CBC_LIB_DIR=/usr/lib
MOS_DEVKIT_MUMPS_LIB=dmumps_seq
MOS_DEVKIT_MUMPS_INC_DIR=/usr/include/mumps_seq
MOS_DEVKIT_MUMPS_LIB_DIR=/usr/lib
```

## Local Build

```
python setup.py build_ext --inplace
```

To re-build, use the ``clean.sh`` script and execute the ``build_ext`` command again.

## Testing

Install pytest and execute

```
pytest -v --show-capture=no
```

## Installation

```
python setup.py install
```
