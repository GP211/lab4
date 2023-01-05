import pyOperator
import giee
from numba import jit


class igrad(pyOperator.Operator):
    """Operator that does causal integration

       b_i=a_i-a_{i-1} in all directions

       """

    def __init__(self, model, data):
        """Initialize operator

             model - Model space
             data  - Data space

             Both must be the same space,  derived from pyVector.vectorIC"""

        if not isinstance(model, giee.vector):
            raise Exception(
                "Model must be giee.vector or derived from it")

        if not isinstance(data, giee.vector):
            raise Exception(
                "Model must be giee.vector or derived from it")


class igrad1(igrad):
    """Operator that does causal integration

       b_i=a_i - a_{i-1}

       """

    def __init__(self, model, data):
        """Initialize operator

             model - Model space
             data  - Data space

             Both must be the same space,  derived from goee.vector, and 1-D"""
        super().__init__(model, data)
        if model.ndims != 1:
            raise Exception("Expecting a 1-D array")
        if not model.checkSame(data):
            raise Exception("Model and data must be the same space")
        self.setDomainRange(model, data)

    def forward(self, add, model, data):
        """Apply the forward
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            data.zero()
        forward1D(model.arr, data.arr)

    def adjoint(self, add, model, data):
        """Apply the adjoint
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            model.zero()
        adjoint1D(model.arr, data.arr)


class igrad2(igrad):
    """Operator that does causal integration

       b_{0,j,i}=a_{j,i} - a_{j,i-1}
       b_{1,j,i}=a_{j,i} - a_{j-1,i}

       """

    def __init__(self, model, data):
        """Initialize operator

             model - Model space
             data  - Data space

             Both must be the same space,  derived from goee.vector, and 1-D"""
        super().__init__(model, data)
        if model.ndims != 2:
            raise Exception("Expecting a 2-D array")
        if model.getHyper().getAxis(1).n != data.getHyper().getAxis(1).n:
            raise Exception("Fast axis of model and data don't match")
        if model.getHyper().getAxis(2).n != data.getHyper().getAxis(2).n:
            raise Exeption(
                "Second axis data and first axis of model not sane size")
        if data.getHyper().getAxis(3).n != 2:
            raise Exception(
                "Model slowest dimension must be size 2 =",
                data.getHyper().getAxis(3).n)
        self.setDomainRange(model, data)

    def forward(self, add, model, data):
        """Apply the forward
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            data.zero()
        forward2D(model.arr, data.arr)

    def adjoint(self, add, model, data):
        """Apply the adjoint
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            model.zero()
        adjoint2D(model.arr, data.arr)


@jit(nopython=True)
def adjoint1D(model, data):
    for i in range(0, model.shape[0] - 1):
        model[i + 1] += data[i]
        model[i] -= data[i]


@jit(nopython=True)
def forward1D(model, data):
    for i in range(0, model.shape[0] - 1):
        data[i] += model[i + 1] - model[i]


@jit(nopython=True)
def adjoint2D(model, data):
    for i2 in range(0, model.shape[0]):
        for i1 in range(0, model.shape[1] - 1):
            model[i2][i1 + 1] += data[0][i2][i1]
            model[i2][i1] -= data[0][i2][i1]

    for i2 in range(0, model.shape[0] - 1):
        for i1 in range(0, model.shape[1]):
            model[i2 + 1][i1] += data[1][i2][i1]
            model[i2][i1] -= data[1][i2][i1]


@jit(nopython=True)
def forward2D(model, data):
    for i2 in range(0, model.shape[0]):
        for i1 in range(0, model.shape[1] - 1):
            data[0][i2][i1] += model[i2][i1 + 1] - model[i2][i1]
    for i2 in range(0, model.shape[0] - 1):
        for i1 in range(0, model.shape[1]):
            data[1][i2][i1] += model[i2 + 1][i1] - model[i2][i1]
