import pyOperator
import giee
from numba import jit


class weight(pyOperator.Operator):
    """Operator
          d= W *m  -Where W is a diagonal matrix

       """

    def __init__(self, w, model, data):
        """Initialize operator

             w - Weight
             model - Model space
             data  - Data space

             Both must be the same space,  derived from giee.vector"""

        if not isinstance(model, giee.vector):
            raise Exception(
                "Model must be giee.vector or derived from it")

        if not isinstance(data, giee.vector):
            raise Exception(
                "Model must be giee.vector or derived from it")

        if not isinstance(w, giee.vector):
            raise Exception(
                "Weight must be giee.vector or derived from it")

        if not model.checkSame(data):
            raise Exception("Model and data must be the same space")

        if not w.checkSame(data):
            raise Exception("Weight and data must be the same space")
        self._weight = w
        self.setDomainRange(model, data)

    def forward(self, add, model, data):
        """Apply the forward
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            data.zero()
        tmp = model.clone()
        tmp.multiply(self._weight)
        data.scaleAdd(tmp)

    def adjoint(self, add, model, data):
        """Apply the adjoint
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            model.zero()
        tmp = data.clone()
        tmp.multiply(self._weight)
        model.scaleAdd(tmp)
