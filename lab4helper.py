import numpy as np
import math
from Triangle import triangle2D_1

from sep_plot import Grey
from sep_python.hypercube import Hypercube, Axis
import sep_python.modes    #Import SEP python module
io=sep_python.modes.default_io  #Get default IO that expects SEPlib datasets and uses sepVectors



def mute(t,x,tp=0.15,slope0=1/1.45,slopep=1/1.45):
    """ 
    Designs a mute function for a particular offset.
    Loop over all offsets to get a mute for the whole gather
    This mute function can then be multiplied with the gather to mute
    unwanted events

    Parameters:
      t:      the times for the current trace (float array nt)
      x:      the offset for the current trace (int)
      tp:     controls the smoothness of the taper of the mute function (large is smoother) [0.15] (float)
      slope0: controls the slope of the mute function [1.45 km/s (water velocity)] (float)
      slopep: controls the slope and also the smoothness of the taper [1.45 km/s] (float)
      
    Returns a mute function for one trace
    """
    nt = len(t)
    wt = np.zeros(nt)
    tx = np.abs(x)
    idx = t < tx*slope0
    wt[idx] = 0; wt[~idx] = 1 
    idx2 = (wt == 1) & (t <= tp + tx*slopep)
    wt[idx2] = np.sin((np.pi/2)*(t[idx2]-tx*slope0)/(tp+tx*(slopep-slope0)))**2

    return wt

def envelope(array,r1):
    """ 
    Computes the envelope of an input semblance panel. 

    Parameters:
      array: input semblance panel (array of floats nt,nslow)
      r1   : length of triangle smoothing filter (integer)
      
    Returns the envelope of the input semblance panel
    """
    ar=array.getNdArray()
    cdata=np.ndarray(shape=ar.shape,dtype=np.complex64)
    cdata[:,:]=ar[:,:]
    cdata=np.fft.fft(ar)
    cdata[:,0]=cdata[:,0]/2.
    nhalf=int(ar.shape[1]/2);
    cdata[:,nhalf]=cdata[:,nhalf]/2.
    cdata[:,nhalf+1:]=0
    cdata=np.fft.ifft(cdata)
    botV=array.clone()
    bot=botV.getNdArray()

    for i2 in range(cdata.shape[0]):
        for i1 in range(cdata.shape[1]):
            bot[i2,i1]=np.conj(cdata[i2,i1])*cdata[i2,i1]

    tmp = botV.clone()

    smop = triangle2D_1(botV,tmp,r1)
    smop.forward(False,botV,tmp)

    return tmp

def picker(s0,a,b,error,envIn):
    """
    Find max within a window defined as b,e= s0 + a*t + b*t^.5 +/- error pct
    
    Parameters:
      s0    - intercept of function (float)
      a     - slope of function (float)
      b     - slope of function (float)
      error - percent error of function (float)
      envIn - Input envelope (float array nt,nx)

    Returns 
      envV  - the clipped slowness scan (float array nt,nslow), 
      tm    - times of the picked slowness
      slowV - the picked slowness values (float array nt)
      amp   - the amplitudes of the picked slownesses (nt)
    """
    envV=envIn.clone()
    axis1 = envV.get_hyper().get_axis(1);
    axis2 = envV.get_hyper().get_axis(2);
    n1 = axis1.n; o1 = axis1.o; d1 = axis1.d
    n2 = axis2.n; o2 = axis2.o; d2 = axis2.d
    #Create a 1-D vector of picked RMS
    slowV = io.get_reg_vector(Hypercube.set_with_ns(ns=[n1], ds=[d1],os=[o1]))
    
    #numpy view into arrays
    env=envV.get_nd_array()
    rms=slowV.get_nd_array()
    #amp
    ampV=slowV.clone()
    ampV.scale(0.)
    amp=ampV.get_nd_array()
    tm=[]
    for it in range(n1):
        t=o1+d1*it
        tm.append(t)
        s=s0+t*a+math.sqrt(t)*b
        ib=int(min(n2-1,max(0,(s*(1.-error/100.)-o2)/d2)))
        ie=int(max(0,min(n2-1,(s*(1.+error/100.)-o2)/d2)))
        amp[it]=0
        mx=ib/2+ie/2
        for islow in range(n2):
            if islow< ib:
                env[islow,it]=0
            elif islow<ie:
                if env[islow,it]> amp[it]:
                    amp[it]=env[islow,it]
                    mx=islow
            else:
                env[islow,it]=0.

        rms[it]=o2+d2*mx

    return envV,tm,slowV,amp

