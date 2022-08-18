import numpy as np
cimport numpy as np

cimport _cbc

np.import_array()

from scipy.sparse import csc_matrix

cdef extern from "numpy/arrayobject.h":
     void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
     void PyArray_CLEARFLAGS(np.ndarray arr, int flags)

cdef ArrayDouble(double* a, int size):
     cdef np.npy_intp shape[1]
     shape[0] = <np.npy_intp> size
     arr = np.PyArray_SimpleNewFromData(1,shape,np.NPY_DOUBLE,a)
     PyArray_CLEARFLAGS(arr,np.NPY_OWNDATA)
     return arr

class CbcContextError(Exception):
    """
    Cbc context error exception.
    """
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

cdef class CbcContext:
    """
    Cbc context class.
    """
    
    cdef _cbc.Cbc_Model* model
    
    def __cinit__(self):

        self.model = _cbc.Cbc_newModel()

    def __dealloc__(self):

        if self.model != NULL:
            _cbc.Cbc_deleteModel(self.model)
        self.model = NULL

    def loadProblem(self, n, A, collb, colub, obj, rowlb, rowub):
        
        A = csc_matrix(A)
        
        cdef np.ndarray[int,mode='c'] _start = A.indptr
        cdef np.ndarray[int,mode='c'] _index = A.indices
        cdef np.ndarray[double,mode='c'] _value = A.data
        cdef np.ndarray[double,mode='c'] _collb = collb
        cdef np.ndarray[double,mode='c'] _colub = colub
        cdef np.ndarray[double,mode='c'] _obj = obj
        cdef np.ndarray[double,mode='c'] _rowlb = rowlb
        cdef np.ndarray[double,mode='c'] _rowub = rowub

        assert(A.shape[1] == n)
        assert(_start.size == (n+1))
        assert(_start[n] == A.nnz)
        assert(_index.size == A.nnz)
        assert(_value.size == A.nnz)
        assert(_collb.size == n)
        assert(_colub.size == n)
        assert(_obj.size == n)
        assert(_rowlb.size == A.shape[0])
        assert(_rowub.size == A.shape[0])

        _cbc.Cbc_loadProblem(self.model,
                             n,
                             A.shape[0],
                             <int*>(_start.data),
                             <int*>(_index.data),
                             <double*>(_value.data),
                             <double*>(_collb.data),
                             <double*>(_colub.data),
                             <double*>(_obj.data),
                             <double*>(_rowlb.data),
                             <double*>(_rowub.data))

    def setInteger(self, flags):
        
        if flags.dtype != 'bool':
            raise CbcContextError('flags must be bool array')
        if flags.size != _cbc.Cbc_getNumCols(self.model):
            raise CbcContextError('flags array must of size numcols')

        for i in range(flags.size):
            if flags[i]:
                _cbc.Cbc_setInteger(self.model, i)

    def setParameter(self, name, value):

        name = str(name).encode('UTF-8')
        value = str(value).encode('UTF-8')
        
        _cbc.Cbc_setParameter(self.model, name, value)

    def isProvenOptimal(self):

        return _cbc.Cbc_isProvenOptimal(self.model)

    def status(self):

        return _cbc.Cbc_status(self.model)

    def solve(self):
        
        return _cbc.Cbc_solve(self.model)

    def getColSolution(self):

        n = _cbc.Cbc_getNumCols(self.model)
        return ArrayDouble(<double*>_cbc.Cbc_getColSolution(self.model),n)

