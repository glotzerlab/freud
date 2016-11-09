// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

//header for function storing wigher3j coefficients

//All wigner3j coefficient created using sympy
//http://www.sympy.org/

#ifndef _WIGNER3J_H
#define _WIGNER3J_H
#include <vector>

using namespace std;

vector<float> getWigner3j(unsigned int l);

#endif
