## \package freud.order
#
# Methods to compute order parameters
#

# __all__ = ['HexOrderParameter']

# not sure if broken
from ._freud import BondOrder;
from ._freud import EntropicBonding;
from ._freud import HexOrderParameter;
from ._freud import TransOrderParameter;
from ._freud import LocalDescriptors;
from ._freud import Pairing2D;

# everything below is sphericalharmonic stuff
from ._freud import LocalQl
from ._freud import LocalQlNear
from ._freud import LocalWl
from ._freud import LocalWlNear
from ._freud import MatchEnv
# from ._freud import SolLiq
# from ._freud import SolLiqNear
#import scipy.io as sio
import numpy as np

'''class LocalWl(LWl):

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

    #read of wigner3j coefficients from wigner3j.mat
    def getwigner(self,l):
        allwig = sio.loadmat('wigner3j.mat')
        W_l = np.array(allwig['l'+str(l)][0],dtype=np.float64)
        return W_l

    #calculate wigner3j coefficients from sympy python library
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
        return W_l'''


#How to set up wigner3j.mat file
#The structure of this .mat file is a list of float numbers in the order of l
# 1st row: l=2 wigner3j, 2nd row: l=4 wigner3j...
#The coefficients are in the order of how the loop is written

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
