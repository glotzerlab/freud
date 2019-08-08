// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef WIGNER3J_H
#define WIGNER3J_H

#include <vector>

/*! \file wigner3j.h
 *  \brief Stores wigner3j coefficients for l ranging from 2 to 20
 */

using namespace std;

vector<double> getWigner3j(unsigned int l);

// All wigner3j coefficients created using sympy
/*

from sympy.physics.wigner import wigner_3j
import numpy as np
import json

wigner = {}
for l in range(21):
    wigner[l] = []
    for m1 in range(-l, l+1):
        for m2 in range(max(-l-m1, -l), min(l-m1, l)+1):
            m3 = -m1-m2
            wigner[l].append(float(wigner_3j(l, l, l, m1, m2, m3)))

with open('wigner3j.json', 'w') as jsonfile:
    json.dump(wigner, jsonfile, sort_keys=True, indent=4)

WIGNER_COEFF = """
    case {l}:
    {{
        return {{
{values}
        }};
    }}"""

def format_values(values, per_line=4, spaces=12):
    lines = []
    for line in range(int(np.ceil(len(values)/per_line))):
        lines.append(', '.join(
            map(str, values[line*per_line:min(len(values), (line+1)*per_line)])))
    return ' '*spaces + (',\n' + ' '*spaces).join(lines)

for l in sorted(wigner.keys()):
    print(WIGNER_COEFF.format(l=l, values=format_values(wigner[l])))

*/

#endif // WIGNER3J_H
