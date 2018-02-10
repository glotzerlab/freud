// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <vector>

#ifndef _WIGNER3J_H
#define _WIGNER3J_H

/*! \file wigner3j.h
 *  \brief Stores wigner3j coefficients for l ranging from 2 to 20
 */

using namespace std;

// All wigner3j coefficients created using sympy
// http://www.sympy.org/
vector<float> getWigner3j(unsigned int l);

#endif
