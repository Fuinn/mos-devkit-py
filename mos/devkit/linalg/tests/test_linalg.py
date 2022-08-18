import unittest
import numpy as np
from mos import devkit
from numpy.linalg import norm
from scipy.sparse import coo_matrix

class TestLinAlg(unittest.TestCase):
   
    def setUp(self):

        np.random.seed(2)

    def test_umfpack(self):

        A = np.random.randn(100,100)
        b = np.random.randn(100)
        
        try:
            umf = devkit.linalg.new_linsolver('umfpack','unsymmetric')
        except ImportError:
            raise unittest.SkipTest('no umfpack')

        self.assertTrue(isinstance(umf, devkit.linalg.LinSolverUMFPACK))
        umf.analyze(A)
        umf.factorize(A)
        x = umf.solve(b)

        self.assertLess(norm(np.dot(A,x)-b),1e-10)

    def test_superlu(self):

        A = np.random.randn(100,100)
        b = np.random.randn(100)

        superlu = devkit.linalg.new_linsolver('superlu','unsymmetric')
        self.assertTrue(isinstance(superlu, devkit.linalg.LinSolverSUPERLU))

        superlu.analyze(A)
        superlu.factorize(A)
        x = superlu.solve(b)

        self.assertLess(norm(np.dot(A,x)-b),1e-10)

    def test_mumps(self):

        A = np.random.randn(100,100)
        b = np.random.randn(100)
        
        try:
            mumps = devkit.linalg.new_linsolver('mumps','unsymmetric')
        except ImportError:
            raise unittest.SkipTest('no mumps')

        self.assertTrue(isinstance(mumps, devkit.linalg.LinSolverMUMPS))
        mumps.analyze(A)
        mumps.factorize(A)
        x = mumps.solve(b)

        self.assertLess(norm(np.dot(A,x)-b),1e-10)
