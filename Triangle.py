from genericSolver.pyOperator import Operator as operator
from genericSolver.pyVector  import vector
from numba import jit, int32, float32
import numpy as np




class triangle(operator):
    """
       Smooth with a triangle

       """

    def __init__(self, model, data,halfLen):
        """Initialize operator

             model - Model space
             data  - Data space
             halfLen - Half-length of triangle

             Both must be the same space,  derived from pyVector.vectorIC"""

        if not isinstance(model, vector):
            raise Exception(
                "Model must be giee.vector or derived from it")

        if not isinstance(data, vector):
            raise Exception(
                "Model must be giee.vector or derived from it")

        if not model.checkSame(data):
            raise Exception("Model and data must be the same space")

        if not isinstance(halfLen,int):
            raise Exception("Expecting halfLen to be an integer")

        self.filt=np.ndarray(shape=(halfLen*2+1),dtype=np.float32)

        scale=1./(float(halfLen)*float(halfLen+1)+halfLen+1)
        for i in range(halfLen):
            self.filt[i]=(i+1)*scale
            self.filt[2*halfLen-i]=(i+1)*scale
        self.filt[halfLen]=(halfLen+1)*scale


class triangle1D(triangle):
    """Operator that applies a triangle smoother to a 1-D dataset


       """

    def __init__(self, model, data,halfLen):
        """Initialize operator

             model - Model space
             data  - Data space
             halfLen -Half length of triangle

             Both must be the same space,  derived from pyVector.vectorIC, and 1-D"""
        super().__init__(model, data,halfLen)
        if model.ndims != 1:
            raise Exception("Expecting a 1-D array")
        self.setDomainRange(model, data)

    def forward(self, add, model, data):
        """Apply the forward
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            data.zero()
        forward1D(model.get_nd_array(), data.get_nd_array(),self.filt)

    def adjoint(self, add, model, data):
        """Apply the adjoint
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            model.zero()
        adjoint1D(model.get_nd_array(), data.get_nd_array(),self.filt)



class triangle2D_1(triangle):
    """Operator that applies a triangle smoother to a 2-D dataset along axis 1 


       """

    def __init__(self, model, data,halfLen):
        """Initialize operator

             model - Model space
             data  - Data space
             halfLen -Half length of triangle

             Both must be the same space,  derived from pyVector.vectorIC, and 1-D"""
        super().__init__(model, data,halfLen)
        if model.ndims != 2:
            raise Exception("Expecting a 2-D array")
        self.setDomainRange(model, data)

    def forward(self, add, model, data):
        """Apply the forward
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            data.zero()
        forward2D_1(model.get_nd_array(), data.get_nd_array(),self.filt)

    def adjoint(self, add, model, data):
        """Apply the adjoint
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            model.zero()
        adjoint2D_1(model.get_nd_array(), data.get_nd_array(),self.filt)


class triangle2D_2(triangle):
    """Operator that applies a triangle smoother to a 2-D dataset along axis 1 


       """

    def __init__(self, model, data,halfLen):
        """Initialize operator

             model - Model space
             data  - Data space
             halfLen -Half length of triangle

             Both must be the same space,  derived from pyVector.vectorIC, and 1-D"""
        super().__init__(model, data,halfLen)
        if model.ndims != 2:
            raise Exception("Expecting a 2-D array")
        self.setDomainRange(model, data)

    def forward(self, add, model, data):
        """Apply the forward
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            data.zero()
        forward2D_2(model.get_nd_array(), data.get_nd_array(),self.filt)

    def adjoint(self, add, model, data):
        """Apply the adjoint
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            model.zero()
        adjoint2D_2(model.get_nd_array(), data.get_nd_array(),self.filt)


@jit(nopython=True)
def adjoint1D(model, data,filt):
    halfLen=(filt.shape[0]-1)/2
    for i in range(0, model.shape[0]):
        for j in range(filt.shape[0]):
            iy=int(max(min(model.shape[0]-1,i+j-halfLen),0))
            model[iy] += data[i]*filt[j]


@jit(nopython=True)
def forward1D(model, data,filt):
    halfLen=(filt.shape[0]-1)/2
    for i in range(0, model.shape[0]):
        for j in range(filt.shape[0]):
            iy=int(max(min(model.shape[0]-1,i+j-halfLen),0))
            data[i] += model[iy]*filt[j]


@jit(nopython=True)
def adjoint2D_1(model, data,filt):
    halfLen=(filt.shape[0]-1)/2
    for i2 in range(0,model.shape[0]):
        for i in range(0, model.shape[1]):
            for j in range(filt.shape[0]):
                iy=int(max(min(model.shape[0]-1,i+j-halfLen),0))
                model[i2,iy] += data[i2,i]*filt[j]


@jit(nopython=True)
def forward2D_1(model, data,filt):
    halfLen=(filt.shape[0]-1)/2
    for i2 in range(0,model.shape[0]):
        for i in range(0, model.shape[1]):
            for j in range(filt.shape[0]):
                iy=int(max(min(model.shape[1]-1,i+j-halfLen),0))
                data[i2,i] += model[i2,iy]*filt[j]


@jit(nopython=True)
def adjoint2D_2(model, data,filt):
    halfLen=(filt.shape[0]-1)/2
    for i2 in range(0,model.shape[0]):
        for i in range(0, model.shape[1]):
            for j in range(filt.shape[0]):
                iy=int(max(min(model.shape[1]-1,i2+j-halfLen),0))
                model[iy,i] += data[i2,i]*filt[j]


@jit(nopython=True)
def forward2D_2(model, data,filt):
    halfLen=(filt.shape[0]-1)/2
    for i2 in range(0,model.shape[1]):
        for i in range(0, model.shape[0]):
            for j in range(filt.shape[0]):
                iy=int(max(min(model.shape[1]-1,i2+j-halfLen),0))
                data[i2,i] += model[iy,i]*filt[j]
