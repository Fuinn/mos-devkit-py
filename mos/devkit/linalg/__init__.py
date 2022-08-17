from .lin_solver import LinSolver
from .mumps import LinSolverMUMPS
from .superlu import LinSolverSUPERLU
from .umfpack import LinSolverUMFPACK

def new_linsolver(name='default', prop='unsymmetric'):
    """
    Creates a linear solver.

    Parameters
    ----------
    name : string
    prop : string
    
    Returns
    -------
    solver : LinSolver
    """
    
    if name == 'mumps':
        return LinSolverMUMPS(prop)
    elif name == 'superlu':
        return LinSolverSUPERLU(prop)
    elif name == 'umfpack':
        return LinSolverUMFPACK(prop)
    elif name == 'default':
        try:
            return new_linsolver('mumps', prop)
        except ImportError:
            return new_linsolver('superlu', prop)            
    else:
        raise ValueError('invalid linear solver name')