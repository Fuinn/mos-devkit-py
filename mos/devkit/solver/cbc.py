from __future__ import print_function
import numpy as np
from .solver_error import *
from .solver import Solver
from mos.devkit.problem import Problem

class SolverCbc(Solver):

    parameters = {'quiet' : False}

    def __init__(self):
        """
        Mixed integer linear "branch and cut" solver from COIN-OR.
        """

        # Import
        from ._cbc import CbcContext
        
        Solver.__init__(self)
        self.parameters = SolverCbc.parameters.copy()

    def supports_properties(self, properties):

        for p in properties:
            if p not in [Problem.PROP_CURV_LINEAR,
                         Problem.PROP_VAR_CONTINUOUS,
                         Problem.PROP_VAR_INTEGER,
                         Problem.PROP_TYPE_FEASIBILITY,
                         Problem.PROP_TYPE_OPTIMIZATION]:
                return False
        return True
        
    def solve(self, problem):

        # Import
        from ._cbc import CbcContext

        # Local vars
        params = self.parameters

        # Parameters
        quiet = params['quiet']

        # Problem
        try:
            self.problem = problem.to_mixintlin()
        except:
            raise SolverError_BadProblemType(self)

        # Cbc context
        self.cbc_context = CbcContext()
        self.cbc_context.loadProblem(self.problem.get_num_primal_variables(),
                                     self.problem.A,
                                     self.problem.l,
                                     self.problem.u,
                                     self.problem.c,
                                     self.problem.b,
                                     self.problem.b)
        self.cbc_context.setInteger(self.problem.P)
        
        # Reset
        self.reset()

        # Options
        if quiet:
            self.cbc_context.setParameter("loglevel", 0)

        # Solve
        self.cbc_context.solve()
        
        # Save
        self.x = self.cbc_context.getColSolution()
        if self.cbc_context.status() == 0:
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        else:
            raise SolverError_Cbc(self)
