from __future__ import print_function
import numpy as np
from .solver_error import *
from .solver import Solver
from mos.devkit.problem import Problem

class SolverClp(Solver):

    parameters = {'quiet' : False}

    def __init__(self):
        """
        Linear programming solver from COIN-OR.
        """

        # Import
        from ._clp import ClpContext
        
        Solver.__init__(self)
        self.parameters = SolverClp.parameters.copy()
        
    def supports_properties(self, properties):

        for p in properties:
            if p not in [Problem.PROP_CURV_LINEAR,
                         Problem.PROP_VAR_CONTINUOUS,
                         Problem.PROP_TYPE_FEASIBILITY,
                         Problem.PROP_TYPE_OPTIMIZATION]:
                return False
        return True
        
    def solve(self, problem):

        # Import
        from ._clp import ClpContext

        # Local vars
        params = self.parameters

        # Parameters
        quiet = params['quiet']

        # Problem
        try:
            self.problem = problem.to_lin()
        except:
            raise SolverError_BadProblemType(self)

        # Clp context
        self.clp_context = ClpContext()
        self.clp_context.loadProblem(self.problem.get_num_primal_variables(),
                                     self.problem.A,
                                     self.problem.l,
                                     self.problem.u,
                                     self.problem.c,
                                     self.problem.b,
                                     self.problem.b)
        
        # Reset
        self.reset()

        # Options
        if quiet:
            self.clp_context.setlogLevel(0)

        # Solve
        self.clp_context.initialSolve()

        # Save
        self.x = self.clp_context.primalColumnSolution()
        self.lam = self.clp_context.dualRowSolution()
        self.pi = np.maximum(self.clp_context.dualColumnSolution(),0)
        self.mu = -np.minimum(self.clp_context.dualColumnSolution(),0)
        if self.clp_context.status() == 0:
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        else:
            raise SolverError_Clp(self)
            
