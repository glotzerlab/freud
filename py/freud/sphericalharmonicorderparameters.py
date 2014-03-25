## \package freud.sphericalharmonicorderparameters
#
# Methods to compute spherical harmonic based order parameters including Local Ql, Local Wl, and Solid-Liquid.
#

# bring related c++ classes into the sphericalharmonicorderparameters module
from _freud import LocalQl
from _freud import LocalWl as LWl
from _freud import SolLiq
import scipy.io as sio
import numpy as np

class LocalWl(LWl):
    
    def __init__(self,box,rmax,l):
        #Initialize those wigner3j values as mx1 numpy arra
        # type of numpy should nd.type=float64??? check.
        super(LocalWl,self).__init__(box,rmax,l)
        #pywig3j = this.wigner3j(l);
        if l < 22:
            super(LocalWl,self).setWigner3j(self.getwigner(l))
        else:
            print('l too big, need sympy library')
            super(LocalWl,self).setWigner3j(self.wigner3j(l))
    
    def getwigner(self,l):
        allwig = sio.loadmat('wigner3j.mat')
        W_l = np.array(allwig['l'+str(l)][0],dtype=np.float64)
        return W_l

    def wigner3j(self,l):
        from sympy.physics.wigner import wigner_3j
        counter = 0
        for u1 in range(2*l+1):
            for u2 in range(max(l-u1,0),min(2*l+1,3*l-u1+1)):
                counter += 1
        W_l = np.zeros (counter,dtype=np.float64)

        i = 0
        for u1 in range(2*l+1):
            for u2 in range(max(l-u1,0),min(2*l+1,3*l-u1+1)):
                u3 = 3*l-u2-u1
                W_l[i] = float(wigner_3j(l,l,l,u1-l,u2-l,u3-l))
                i += 1
        return W_l


#How to set up wigner3j.mat file

''' import scipy.io as sio
 from sympy.physics.wigner import wigner_3j
 import numpy as np
 
 wigner={}
 for i in range(10):
     counter = 0
     l = (i+1)*2
     for u1 in range(2*l+1):
         for u2 in range(max(l-u1,0),min(2*l+1,3*l-u1+1)):
             counter += 1
     W_l = np.zeros(counter,dtype=np.float64)
 
     j = 0
     for u1 in range(2*l+1):
         for u2 in range(max(l-u1,0),min(2*l+1,3*l-u1+1)):
             u3 = 3*l-u2-u1
             W_l[j] = float(wigner_3j(l,l,l,u1-l,u2-l,u3-l))
             j += 1
     wigner['l'+str(l)]=W_l
 
sio.savemat('wigner3j.mat',wigner)'''
                                 
