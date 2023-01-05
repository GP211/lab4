import pyOperator
import giee
from numba import jit


class identity(pyOperator.Operator):
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

        if not isinstance(model, giee.vector):
            raise Exception(
                "Model must be giee.vector or derived from it")

        if not isinstance(data, giee.vector):
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
