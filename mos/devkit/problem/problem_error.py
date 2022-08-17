
class ProblemError(Exception):
    
    def __init__(self,value):
        self.value = value
        
    def __str__(self):
        return str(self.value)

class ProblemError_InvalidDataDimensions(ProblemError):    
    def __init__(self):
        ProblemError.__init__(self,'invalid data dimemnesions')


