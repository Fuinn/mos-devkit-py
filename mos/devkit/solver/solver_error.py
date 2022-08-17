class SolverError(Exception):
    
    def __init__(self, solver, value):
        if solver:
            solver.set_status(solver.STATUS_ERROR)
            solver.set_error_msg(value)
        self.value = value
        
    def __str__(self):
        return str(self.value)
    
class SolverError_Cbc(SolverError):    
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'cbc solver failed')

class SolverError_CbcCMD(SolverError):    
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'cbc command-line solver failed')

class SolverError_CbcCMDCall(SolverError):    
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'error while calling cbc command-line solver')
        
class SolverError_Clp(SolverError):    
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'clp solver failed')

class SolverError_ClpCMD(SolverError):    
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'clp command-line solver failed')

class SolverError_ClpCMDCall(SolverError):    
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'error while calling clp command-line solver')

class SolverError_CplexCMD(SolverError):    
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'cplex command-line solver failed')

class SolverError_CplexCMDCall(SolverError):    
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'error while calling cplex command-line solver')

class SolverError_Ipopt(SolverError):    
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'ipopt solver failed')

class SolverError_NumProblems(SolverError):
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'numerical problems')

class SolverError_LineSearch(SolverError):    
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'line search failed')

class SolverError_BadProblemType(SolverError):
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'invalid problem type')

class SolverError_BadLinSolver(SolverError):
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'invalid linear solver')

class SolverError_BadSearchDir(SolverError_LineSearch):
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'bad search direction')

class SolverError_BadLinSystem(SolverError):
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'bad linear system')

class SolverError_LinFeasLost(SolverError):
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'linear equality constraint feasibility lost')

class SolverError_Infeasibility(SolverError):
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'problem appears infeasible')

class SolverError_NoInterior(SolverError):
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'empty interior')

class SolverError_MaxIters(SolverError):
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'maximum number of iterations')

class SolverError_SmallPenalty(SolverError):
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'penalty parameter too small')

class SolverError_BadInitPoint(SolverError):
    def __init__(self, solver=None):
        SolverError.__init__(self, solver, 'bad initial point')

