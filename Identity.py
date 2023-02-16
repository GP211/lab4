from genericSolver.pyOperator import Operator as operator
from genericSolver.pyVector  import vector
from sep_python.hypercube import Hypercube, Axis
import sep_python.modes    #Import SEP python module
io=sep_python.modes.default_io  #Get default IO that expects SEPlib datasets and uses sepVectors

from numba import jit


class identity(operator):
    """
    Identity operator
    """

    def __init__(self, model, data):
        """
        Initialize operator
        
        model - Model space
        data  - Data space
        
        Both must be the same dimension,  derived from pyVector.vectorIC
        """

        if not isinstance(model, vector):
            raise Exception(
                "Model must be vector or derived from it")

        if not isinstance(data, vector):
            raise Exception(
                "Model must be pyVector.vectorIC or derived from it")

        if not model.checkSame(data):
            raise Exception("Model and data must be same domain")

        self.setDomainRange(model, data)

    def forward(self, add, model, data):
        """Apply the forward
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            data.zero()
        data.scaleAdd(model)

    def adjoint(self, add, model, data):
        """Apply the adjoint
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            model.zero()
        model.scaleAdd(data)
