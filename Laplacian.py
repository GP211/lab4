import pyOperator
import giee
import Hypercube
from Igrad import igrad2
from numba import jit

class laplacian(pyOperator.Operator):
    """
    Applies the laplacian
    """

    def __init__(self, model, data):
        """
        Initialize operator

         model - Model space
         data  - Data space

        Both must be the same space,  derived from giee.vector, and 1-D
        """
        super().__init__(model, data)
        if model.ndims != 2:
            raise Exception("Expecting a 2-D array")

        if not model.checkSame(data):
            raise Exception("Model and data must be the same space")
        self.setDomainRange(model, data)

        try:
            h=model.getHyper()
            m=model.getNdArray()
        except:
            raise Exception("Model must have a hypercube and numpy representation")

        # Create a temporary vector
        axis1 = model.getHyper().getAxis(1)
        axis2 = model.getHyper().getAxis(2)
        n1 = axis1.n; o1= axis1.o; d1 = axis1.d
        n2 = axis2.n; o2= axis2.o; d2 = axis2.d
        self._tmp = giee.vector(Hypercube.hypercube(ns=[n1,n2,2], ds=[d1,d2,1.0],os=[o1,o2,0.0]))

        # Create the gradient operator
        self._gop = igrad2(model,self._tmp)

    def forward(self, add, model, data):
        """
        Apply the forward

        Parameters:
          add - Whether or not to add to data
          model - Model
          data - Data 
        """
        self.checkDomainRange(model, data)

        if not add:
            data.zero()
        self._gop.forward(False,model,self._tmp)
        self._gop.adjoint(add,data,self._tmp)

    def adjoint(self, add, model, data):
        """Apply the adjoint
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            model.zero()

        self._gop.forward(False,data,self._tmp)
        self._gop.adjoint(add,model,self._tmp)

