import pyOperator
import pyVector
import giee
from numba import jit, int32, float32
import numpy as np

class tcai(pyOperator.Operator):
  """ 
  1D Transient Convolution operator, Adjoint is Input
  """

  def __init__(self,dom,rng,flt):
    """ 
    Initialize operator and save the space of the operator

    Parameters:
      dom: python vector defining the domain of the operator
      rng: python vector defining the range of the operator
      flt: python vector containing the filter coefficients
    """
    if not isinstance(dom,pyVector.vector):
      raise Exception("Expecting domain to be a python vector")
    
    if not isinstance(rng,pyVector.vector):
      raise Exception("Expecting range to be a python vector")

    if not isinstance(flt,pyVector.vector):
      raise Exception("Expecting filter to be a python vector")
    
    # Store the vector space of the domain and range
    super().__init__(dom,rng)
    
    # Save the filter
    self.__flt = flt 

  def forward(self,add,modl,data):
    """ 
    Applies the forward transient convolution operator

    Parameters:
      add  - boolean whether or not to add the data vector or zero it first
      modl - input signal to be filtered
      data - output filtered signal
    """
    if not add:
      data.zero()

    forward_tcai(self.__flt.getNdArray(),modl.getNdArray(),data.getNdArray())

  def adjoint(self,add,modl,data):
    """ 
    Applies the adjoint transient convolution operator

    Parameters:
      add  - boolean whether or not to add the data vector or zero it first
      modl - output correlated signal
      data - input filtered signal
    """
    if not add:
      modl.zero()

    adjoint_tcai(self.__flt.getNdArray(),modl.getNdArray(),data.getNdArray())

@jit(nopython=True)
def forward_tcai(flt,modl,data):
  # Get dimensions
  nf = flt.shape[0]; nm = modl.shape[0]; nd = data.shape[0]
  if(nf != nd - nm  + 1):
    raise Exception("Size of filter does not match output and input")
  for i in range(nf):
    for im in range(nm):
      j = im + i - 1
      data[j] += modl[im]*flt[i]

@jit(nopython=True)
def adjoint_tcai(flt,modl,data):
  # Get dimensions
  nf = flt.shape[0]; nm = modl.shape[0]; nd = data.shape[0]
  if(nf != nd - nm  + 1):
    raise Exception("Size of filter does not match output and input")
  for i in range(nf):
    for im in range(nm):
      j = im + i - 1
      modl[im] += data[j]*flt[i]

