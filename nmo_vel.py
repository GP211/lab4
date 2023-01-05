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

    def forward(self,add,modl,data,vel):
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
                    modl.getNdarray(),data.getNdArray(),vel)

    def adjoint(self,add,modl,data,vel):
        """
        Applies the adjoint operator:
        Hyperbolas in time-space to spikes in depth-slowness

        Parameters:
          add - boolean whether or not to add the model vector or zero it first
          modl - slowness model (s,z)
          data - hyperbolas (t,x)
        """

        if not add:
            modl.zero()

        adjoint2D_1(self._oq,self._dq,
                    self._oz,self._dz,
                    self._ox,self._dx,
                    self._ot,self._dt,
                    modl.getNdArray(),data.getNdArray(),vel)

####### Please complete the two functions below #######
@jit(nopython=True)
def forward2D_1(oq,dq,oz,dz,ox,dx,ot,dt,modl,data,vel):
    #TODO: implement forward operator
    for iq in np.arange(modl.shape[0]):
        q = oq + dq * iq
        for iz in np.arange(modl.shape[1]):
            for ix in np.arange(data.shape[0]):
                x = ox + dx * ix
                z = oz + dz * iz
                # x = ox + dx * data.shape[1]
                xs = x/vel[iz]
                t = np.sqrt(z*z + xs*xs)
                #it = int(0.5 + (t-ot)/dt) # This is nearest neighbors, we need to do interpolation
                ft = ((t-ot)/dt) #? 
                it = int(ft)
                ft = ft - it
                gt = 1 - ft
                if(it < data.shape[1]-1):
                    data[ix,it] += modl[iq,iz]*gt
                    data[ix,it+1] += modl[iq,iz]*ft
    pass

@jit(nopython=True)
def adjoint2D_1(oq,dq,oz,dz,ox,dx,ot,dt,modl,data,vel):
    #TODO: implement adjoint operator
    for iq in np.arange(modl.shape[0]):
        q = oq + dq * iq
        for iz in np.arange(modl.shape[1]):
            for ix in np.arange(data.shape[0]):
                x = ox + dx * ix
                z = oz + dz * iz
                # x = ox + dx * data.shape[1]
                xs = x/vel[iz]
                t = np.sqrt(z*z + xs*xs)
                #it = int(0.5 + (t-ot)/dt) #? 
                ft = ((t-ot)/dt) # lets say this is 3.2
                it = int(ft) # lets say this is now 3
                ft = ft - it # now this is 0.2
                gt = 1 - ft # now this is 0.8
                if(it < data.shape[1]-1):
                    modl[iq,iz] += data[ix,it]*gt # flipped logic
                    modl[iq,iz] += data[ix,it+1]*ft # remember we are pulling because the variable we are not looping over it, is being added
    pass