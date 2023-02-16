from genericSolver.pyOperator import Operator as operator
from sep_python.hypercube import Hypercube, Axis
import sep_python.modes    #Import SEP python module
io=sep_python.modes.default_io  #Get default IO that expects SEPlib datasets and uses sepVectors

from Igrad import igrad2
from numba import jit

class laplacian(operator):
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
            h=model.get_hyper()
            m=model.get_nd_array()
        except:
            raise Exception("Model must have a hypercube and numpy representation")

        # Create a temporary vector
        axis1 = model.get_hyper().get_axis(1)
        axis2 = model.get_hyper().get_axis(2)
        n1 = axis1.n; o1= axis1.o; d1 = axis1.d
        n2 = axis2.n; o2= axis2.o; d2 = axis2.d
        self._tmp = io.get_reg_vector(Hypercube.set_with_ns(ns=[n1,n2,2], ds=[d1,d2,1.0],os=[o1,o2,0.0]))

        
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

