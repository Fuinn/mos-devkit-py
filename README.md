# mos-devkit-py

Python model development kit for MOS.

## Contents

* Linear solver wrappers
* Optimizaiton problem classes
* Optimization solvers and wrappers
* Algebraic modeling system

## Prerequisites

```
pip install -r requirements.txt
```

## Environment Variables

The following environment variables can be used to provide libary names, include and library directories for the following wrappers:

* MUMPS: ``MOS_DEVKIT_MUMPS_LIB``, ``MOS_DEVKIT_MUMPS_LIB_DIR``, ``MOS_DEVKIT_MUMPS_INC_DIR``
* IPOPT: ``MOS_DEVKIT_IPOPT_LIB``, ``MOS_DEVKIT_IPOPT_LIB_DIR``, ``MOS_DEVKIT_IPOPT_INC_DIR``
* CLP: ``MOS_DEVKIT_CLP_LIB``, ``MOS_DEVKIT_CLP_LIB_DIR``, ``MOS_DEVKIT_CLP_INC_DIR``
* CBC: ``MOS_DEVKIT_CBC_LIB``, ``MOS_DEVKIT_CBC_LIB_DIR``, ``MOS_DEVKIT_CBC_INC_DIR``

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
