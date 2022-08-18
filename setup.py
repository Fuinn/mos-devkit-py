import os
import sys
import numpy as np
from subprocess import call
from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension

# Extension modules
ext_modules = []

# MUMPS
mumps_lib = os.environ.get('MOS_DEVKIT_MUMPS_LIB')
mumps_inc_dir = os.environ.get('MOS_DEVKIT_MUMPS_INC_DIR')
mumps_lib_dir = os.environ.get('MOS_DEVKIT_MUMPS_LIB_DIR')
if mumps_lib:
    ext_modules += cythonize([Extension(name='mos.devkit.linalg._mumps._dmumps',
                                        sources=['./mos/devkit/linalg/_mumps/_dmumps.pyx'],
                                        libraries=[mumps_lib],
                                        include_dirs=[mumps_inc_dir],
                                        library_dirs=[mumps_lib_dir],
                                        extra_link_args=[])])

# IPOPT
ipopt_lib = os.environ.get('MOS_DEVKIT_IPOPT_LIB')
ipopt_inc_dir = os.environ.get('MOS_DEVKIT_IPOPT_INC_DIR')
ipopt_lib_dir = os.environ.get('MOS_DEVKIT_IPOPT_LIB_DIR')
if ipopt_lib:
    ext_modules += cythonize([Extension(name='mos.devkit.solver._ipopt._ipopt',
                                        sources=['./mos/devkit/solver/_ipopt/_ipopt.pyx'],
                                        libraries=[ipopt_lib],
                                        include_dirs=[np.get_include(), ipopt_inc_dir],
                                        library_dirs=[ipopt_lib_dir],
                                        extra_link_args=[])])
    
# CLP
clp_lib = os.environ.get('MOS_DEVKIT_CLP_LIB')
clp_inc_dir = os.environ.get('MOS_DEVKIT_CLP_INC_DIR')
clp_lib_dir = os.environ.get('MOS_DEVKIT_CLP_LIB_DIR')
if clp_lib:
    ext_modules += cythonize([Extension(name='mos.devkit.solver._clp._clp',
                                        sources=['./mos/devkit/solver/_clp/_clp.pyx'],
                                        libraries=[clp_lib],
                                        include_dirs=[np.get_include(), clp_inc_dir],
                                        library_dirs=[clp_lib_dir],
                                        extra_link_args=[])])

# CBC
cbc_lib = os.environ.get('MOS_DEVKIT_CBC_LIB')
cbc_inc_dir = os.environ.get('MOS_DEVKIT_CBC_INC_DIR')
cbc_lib_dir = os.environ.get('MOS_DEVKIT_CBC_LIB_DIR')
if cbc_lib:
    ext_modules += cythonize([Extension(name='mos.devkit.solver._cbc._cbc',
                                        sources=['./mos/devkit/solver/_cbc/_cbc.pyx'],
                                        libraries=[cbc_lib],
                                        include_dirs=[np.get_include(), cbc_inc_dir],
                                        library_dirs=[cbc_lib_dir],
                                        extra_link_args=[])])

# Expression evaluator
ext_modules += cythonize([Extension(name='mos.devkit.model._eval._eval',
                                   sources=['./mos/devkit/model/_eval/_eval.pyx',
                                            './mos/devkit/model/_eval/evaluator.c',
                                            './mos/devkit/model/_eval/node.c'],
                                   libraries=[],
                                   include_dirs=[np.get_include(), './mos/devkit/model/_eval'],
                                   library_dirs=[],
                                   extra_link_args=[])])

setup(name='mos-devkit',
      zip_safe=False,
      version='0.1.0',
      author='Fuinn',
      url='https://github.com/Fuinn/mos-devkit-py',
      description='Python model development kit for MOS',
      license='BSD 3-Clause License',
      packages=find_packages(),
      include_package_data=True,
      classifiers=['Development Status :: 4 - Beta',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: MacOS',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.6'],
      package_data={'mos.devkit.linalg._mumps' : ['libcoinmumps*'],
                    'mos.devkit.solver._ipopt' : ['libipopt*'],
                    'mos.devkit.solver._clp' : ['libClp*'],
                    'mos.devkit.solver._cbc' : ['libCbc*']},
      install_requires=['cython>=0.20.1',
                        'numpy>=1.11.2',
                        'scipy>=0.18.1'],
      ext_modules=ext_modules)