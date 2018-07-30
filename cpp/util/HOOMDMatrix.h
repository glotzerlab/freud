/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*! \file HOOMDMatrix.h
    \brief matrix code stolen from HOOMD for eigenvector-related calculations
*/

#ifndef HOOMD_MATRIX_H
#define HOOMD_MATRIX_H

#include "HOOMDMath.h"

namespace hoomd{namespace matrix{

// Maximum number of iterations for Jacobi rotations
#define MAXJACOBI 50

/*! Perform a single Jacobi rotation
    \param matrix Matrix to be diagonalized
    \param i
    \param j
    \param k
    \param l
    \param s
    \param tau
*/

void rotate(float matrix[][3], int i, int j, int k, int l, float s, float tau);

/*! Compute eigenvalues and eigenvectors of 3x3 real symmetric matrix based on Jacobi rotations adapted from Numerical Recipes jacobi() function (LAMMPS)
    \param matrix Matrix to be diagonalized
    \param evalues Eigen-values obtained after diagonalized
    \param evectors Eigen-vectors obtained after diagonalized in columns

*/

int diagonalize(float matrix[][3], float *evalues, float evectors[][3]);

/*! Calculate the quaternion from three axes
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector
    \param quat returned quaternion
*/

void quaternionFromExyz(float4 &ex_space, float4 &ey_space, float4 &ez_space, float4 &quat);

};}; // end namespace hoomd::matrix

#endif // HOOMD_MATRIX_H
