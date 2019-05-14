// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef WIGNER3J_H
#define WIGNER3J_H

#include <vector>

/*! \file wigner3j.h
 *  \brief Stores wigner3j coefficients for l ranging from 2 to 20
 */

using namespace std;

vector<float> getWigner3j(unsigned int l);

// All wigner3j coefficients created using sympy
/*
from sympy.physics.wigner import wigner_3j
import scipy.io as sio
import numpy as np
import json

wigner = {}
for i in range(10):
    counter = 0
    l = (i+1)*2
    print('Computing l={}'.format(l))
    for u1 in range(2*l+1):
        for u2 in range(max(l-u1,0),min(2*l+1,3*l-u1+1)):
            counter += 1
    W_l = np.zeros(counter, dtype=np.float64)

    j = 0
    for u1 in range(2*l+1):
        for u2 in range(max(l-u1,0),min(2*l+1,3*l-u1+1)):
            u3 = 3*l-u2-u1
            W_l[j] = float(wigner_3j(l,l,l,u1-l,u2-l,u3-l))
            j += 1
    wigner['l'+str(l)] = W_l.tolist()

with open('wigner3j.json', 'w') as jsonfile:
    json.dump(wigner, jsonfile, sort_keys=True, indent=4)
sio.savemat('wigner3j.mat',wigner)'''
*/

#endif // WIGNER3J_H
