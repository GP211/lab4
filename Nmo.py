import pyOperator
import pyVector
from numba import jit, int32, float32
import numpy as np
import math

class nmo(pyOperator.Operator):
    """
    Operator that does NMO for 2-D field (time,velocity)
    """

    def __init__(self, slw, model, data):
        """Initialize operator

             vel  - Velocity
             model - Model space
             data  - Data space

             Both must be the same space,  derived from pyVector.vectorIC"""

        if not isinstance(model, pyVector.vector):
            raise Exception(
                "Model must be giee.vectorIC or derived from it")

        if not isinstance(data, pyVector.vector):
            raise Exception(
                "Model must be giee.vector or derived from it")

        if not model.checkSame(data):
            raise Exception("Model and data must be the some space")

        self._slow = slw 

        self.setDomainRange(model, data)

        hyp = model.getHyper()
        if len(hyp.axes) != 2:
            raise Exception("Expecting 2-D field")

        self._nt = hyp.axes[0].n
        self._ot = hyp.axes[0].o
        self._dt = hyp.axes[0].d

        self._no = hyp.axes[1].n
        self._oo = hyp.axes[1].o
        self._do = hyp.axes[1].d

    def forward(self, add, model, data):
        """Apply the forward
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            data.zero()
        forward1DL(
            self._ot,
            self._dt,
            self._oo,
            self._do,
            self._slow.arr,
            model.arr,
            data.arr)

    def adjoint(self, add, model, data):
        """Apply the adjoint
                add - Whether or not to add to data
                model - Model
                data - Data """
        self.checkDomainRange(model, data)

        if not add:
            model.zero()

        adjoint1DL(
            self._ot,
            self._dt,
            self._oo,
            self._do,
            self._slow.arr,
            model.arr,
            data.arr)


@jit(nopython=True)
def forward1DL(ot, dt, oo, do, slw, model, data):
    nz = model.shape[1]; nx = model.shape[0]
    for ix in range(nx):
        x = oo + ix*do
        for iz in range(nz):
            z = ot + dt*iz
            xs = x * slw[iz]
            t = np.sqrt( z * z + xs * xs) + 1e-20
            wt = z/t * ( 1/np.sqrt(t) )
            it = int((t - ot)/dt + 0.5)
            if(it < nz):
                data[ix][it] += model[ix][iz] * wt


@jit(nopython=True)
def adjoint1DL(ot, dt, oo, do, slw, model, data):
    nz = model.shape[1]; nx = model.shape[0]
    for ix in range(nx):
        x = oo + ix*do
        for iz in range(nz):
            z = ot + dt*iz
            xs = x * slw[iz]
            t = np.sqrt( z * z + xs * xs) + 1e-20
            wt = z/t * ( 1/np.sqrt(t) )
            it = int((t - ot)/dt + 0.5)
            if(it < nz):
                model[ix][iz] += data[ix][it] * wt


