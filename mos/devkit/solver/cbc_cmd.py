from __future__ import print_function
import os
import numpy as np
import tempfile
import subprocess
from . import utils
from .solver_error import *
from .solver import Solver
from mos.devkit.problem import Problem

class SolverCbcCMD(Solver):

    parameters = {'quiet' : False, 'debug': False}

    def __init__(self):
        """
        Mixed integer linear "branch and cut" solver from COIN-OR (via command-line interface, version 2.8.5).
        """

        # Check
        if not utils.cmd_exists('cbc'):
            raise ImportError('cbc cmd not available')
        
        Solver.__init__(self)
        self.parameters = SolverCbcCMD.parameters.copy()

    def supports_properties(self, properties):

        for p in properties:
            if p not in [Problem.PROP_CURV_LINEAR,
                         Problem.PROP_VAR_CONTINUOUS,
                         Problem.PROP_VAR_INTEGER,
                         Problem.PROP_TYPE_FEASIBILITY,
                         Problem.PROP_TYPE_OPTIMIZATION]:
                return False
        return True

    def read_solution(self, filename, problem):

        f = open(filename, 'r')
        
        l = f.readline().split()
        status = l[0]    
        
        x = np.zeros(problem.c.size)
        lam = np.zeros(problem.A.shape[0])
        nu = np.zeros(0)
        mu = np.zeros(x.size)
        pi = np.zeros(x.size)
        for l in f:
            l = l.split()
            name = l[1]
            if name[0] == 'x':
                i = int(name.split('_')[1])
                x[i] = float(l[2])
                if float(l[3]) > 0.:
                    pi[i] = float(l[3])
                else:
                    mu[i] = -float(l[3])
            elif name[0] == 'c':
                i = int(name.split('_')[1])
                lam[i] = float(l[3])
        f.close()
        return status, x, lam, nu, mu, pi
        
    def solve(self, problem):

        # Local vars
        params = self.parameters

        # Parameters
        quiet = params['quiet']
        debug = params['debug']

        # Problem
        try:
            self.problem = problem.to_mixintlin()
        except:
            raise SolverError_BadProblemType(self)

        # Solve
        status = ''
        try:
            base_name = next(tempfile._get_candidate_names())
            input_filename = base_name+'.lp'
            output_filename = base_name+'.sol'
            self.problem.write_to_lp_file(input_filename)
            cmd = ['cbc',
                   input_filename,
                   'solve',
                   'printingOptions',
                   'all',
                   'solution',
                   output_filename]
            if not quiet:
                code = subprocess.call(cmd)
            else:
                code = subprocess.call(cmd,
                                       stdout=open(os.devnull, 'w'),
                                       stderr=subprocess.STDOUT)
            assert(code == 0)
            status, self.x, self.lam, self.nu, self.mu, self.pi = self.read_solution(output_filename, self.problem)
        except Exception as e:
            raise SolverError_CbcCMDCall(self)
        finally:
            if os.path.isfile(input_filename) and not debug:
                os.remove(input_filename)
            if os.path.isfile(output_filename) and not debug:
                os.remove(output_filename)

        if status == 'Optimal':
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        else:
            raise SolverError_CbcCMD(self)

        
