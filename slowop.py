import pyOperator
import pyVector
import giee
from numba import jit, int32, float32
import numpy as np

class slow(pyOperator.Operator):
  """
  2D slowness operator that maps spikes in slowness and depth
  to hyperbolas in time and space
  """

  def __init__(self, dom, rng):
    """
    Initialize operator and saves the space of the operator
    """
    if not isinstance(dom,pyVector.vector): 
      raise Exception("Expecting domain to be a python vector")

    if not isinstance(rng,pyVector.vector): 
      raise Exception("Expecting range to be a python vector")

    # Store the vector space of the domain and range
    super().__init__(dom,rng)

    # Get model axes
    zaxis = dom.getHyper().getAxis(1)
    qaxis = dom.getHyper().getAxis(2)
    # Get data axes
    taxis = rng.getHyper().getAxis(1)
    xaxis = rng.getHyper().getAxis(2)

    # Get model dimensions
    self._oq = qaxis.o; self._dq = qaxis.d
    self._oz = zaxis.o; self._dz = zaxis.d
    # Get data dimensions
    self._ox = xaxis.o; self._dx = xaxis.d
    self._ot = taxis.o; self._dt = taxis.d

  def forward(self,add,modl,data):
    """
    Applies the forward operator:
    Spikes in depth-slowness to hyperbolas in time-space

    Parameters:
      add - boolean whether or not add to the data vector or zero it first
      modl - slowness model (s,z)
      data - hyperbolas (t,x)
    """
    self.checkDomainRange(modl,data)

    # Zero the data if add == false
    if not add:
      data.zero()

    forward2D_1(self._oq,self._dq,
                self._oz,self._dz,
                self._ox,self._dx,
                self._ot,self._dt,
                modl.getNdArray(),data.getNdArray())

  def adjoint(self,add,modl,data):
    """
    Applies the adjoint operator:
    Hyperbolas in time-space to spikes in depth-slowness

    Parameters:
      add - boolean whether or not to add the model vector or zero it first
      modl - slowness model (s,z)
      data - hyperbolas (t,x)
    """
    self.checkDomainRange(modl,data)

    if not add:
      modl.zero()

    adjoint2D_1(self._oq,self._dq,
                self._oz,self._dz,
                self._ox,self._dx,
                self._ot,self._dt,
                modl.getNdArray(),data.getNdArray())

@jit(nopython=True)
def forward2D_1(oq,dq,oz,dz,ox,dx,ot,dt,modl,data):
  # Model dimensions
  nq = modl.shape[0]; nz = modl.shape[1]
  # Data dimensions
  nx = data.shape[0]; nt = data.shape[1]
  for iq in range(nq):
    q = oq + iq*dq
    for ix in range(nx):
      x = ox + ix*dx
      sx = np.abs(q * x)
      for iz in range(nz):
        z = oz + iz*dz
        # Given slowness, offset and depth, compute time from NMO equation
        t = np.sqrt( z * z + sx * sx )
        # Compute linear interpolation weights
        f = (t - oz)/dz
        it = int(f + 0.5)
        fx = f - it; gx = 1.0 - fx
        # Linearly interpolate
        if(it >= 0 and it < nt-1):
          data[ix][it+0] += gx*modl[iq][iz]
          data[ix][it+1] += fx*modl[iq][iz]

@jit(nopython=True)
def adjoint2D_1(oq,dq,oz,dz,ox,dx,ot,dt,modl,data):
  # Model dimensions
  nq = modl.shape[0]; nz = modl.shape[1]
  # Data dimensions
  nx = data.shape[0];  nt = data.shape[1]
  for iq in range(nq):
    q = oq + iq*dq
    for ix in range(nx):
      x = ox + ix*dx
      sx = np.abs(q * x)
      for iz in range(nz):
        z = oz + iz*dz
        # Given slowness, offset and depth, compute time from NMO equation
        t = np.sqrt( z * z + sx * sx )
        # Compute linear interpolation weights
        f = (t - oz)/dz
        it = int(f + 0.5)
        fx = f - it; gx = 1.0 - fx
        # Linearly interpolate
        if(it >= 0 and it < nt - 1):
          modl[iq][iz] += gx*data[ix][it] + fx*data[ix][it+1]

