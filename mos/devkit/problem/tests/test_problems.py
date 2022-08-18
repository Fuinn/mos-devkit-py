import os
import unittest
import numpy as np
from mos import devkit

class TestProblems(unittest.TestCase):

    def test_lin_to_lp_file(self):

        A = np.array([[-2.,2. ,1.,0.],
                      [-8.,10.,0.,1.]])
        b = np.array([1.,13.])
        
        l = np.array([-1e8,-1e8,-1e8,0.])
        u = np.array([1e8,1e8,0.,1e8])
        
        c = np.array([-1.,-1.,0.,0.])
                
        problem = devkit.problem.LinProblem(c,A,b,l,u)

        try:
            
            problem.write_to_lp_file('foo.lp')

        finally:

            if os.path.isfile('foo.lp'):
                os.remove('foo.lp')
   
    def test_mixintlin_to_lp_file(self):

        A = np.array([[-2.,2. ,1.,0.],
                      [-8.,10.,0.,1.]])
        b = np.array([1.,13.])
        
        l = np.array([-1e8,-1e8,-1e8,0.])
        u = np.array([1e8,1e8,0.,1e8])
        
        c = np.array([-1.,-1.,0.,0.])
        
        P = np.array([True,True,False,False])
        
        problem = devkit.problem.MixIntLinProblem(c,A,b,l,u,P)

        try:
            
            problem.write_to_lp_file('foo.lp')

        finally:

            if os.path.isfile('foo.lp'):
                os.remove('foo.lp')
        
