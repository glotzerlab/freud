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

/*! \file HOOMDMatrix.cc
    \brief matrix code stolen from HOOMD for eigenvector-related calculations
*/

#include "HOOMDMatrix.h"

namespace hoomd{namespace matrix{

/*! Perform a single Jacobi rotation
    \param matrix Matrix to be diagonalized
    \param i
    \param j
    \param k
    \param l
    \param s
    \param tau
*/

void rotate(float matrix[][3], int i, int j, int k, int l, float s, float tau)
    {
    float g = matrix[i][j];
    float h = matrix[k][l];
    matrix[i][j] = g - s * (h + g * tau);
    matrix[k][l] = h + s * (g - h * tau);
    }

/*! Compute eigenvalues and eigenvectors of 3x3 real symmetric matrix based on Jacobi rotations adapted from Numerical Recipes jacobi() function (LAMMPS)
    \param matrix Matrix to be diagonalized
    \param evalues Eigen-values obtained after diagonalized
    \param evectors Eigen-vectors obtained after diagonalized in columns

*/

int diagonalize(float matrix[][3], float *evalues, float evectors[][3])
    {
    int i,j,k;
    float tresh, theta, tau, t, sm, s, h, g, c, b[3], z[3];

    for (i = 0; i < 3; i++)
        {
        for (j = 0; j < 3; j++) evectors[i][j] = 0.0;
        evectors[i][i] = 1.0;
        }

    for (i = 0; i < 3; i++)
        {
        b[i] = evalues[i] = matrix[i][i];
        z[i] = 0.0;
        }

    for (int iter = 1; iter <= MAXJACOBI; iter++)
        {
        sm = 0.0;
        for (i = 0; i < 2; i++)
            for (j = i+1; j < 3; j++)
                sm += fabs(matrix[i][j]);

        if (sm == 0.0) return 0;

        if (iter < 4) tresh = 0.2*sm/(3*3);
        else tresh = 0.0;

        for (i = 0; i < 2; i++)
            {
            for (j = i+1; j < 3; j++)
                {
                g = 100.0 * fabs(matrix[i][j]);
                if (iter > 4 && fabs(evalues[i]) + g == fabs(evalues[i])
                        && fabs(evalues[j]) + g == fabs(evalues[j]))
                    matrix[i][j] = 0.0;
                else if (fabs(matrix[i][j]) > tresh)
                    {
                    h = evalues[j]-evalues[i];
                    if (fabs(h)+g == fabs(h)) t = (matrix[i][j])/h;
                    else
                        {
                        theta = 0.5 * h / (matrix[i][j]);
                        t = 1.0/(fabs(theta)+sqrt(1.0+theta*theta));
                        if (theta < 0.0) t = -t;
                        }

                    c = 1.0/sqrt(1.0+t*t);
                    s = t*c;
                    tau = s/(1.0+c);
                    h = t*matrix[i][j];
                    z[i] -= h;
                    z[j] += h;
                    evalues[i] -= h;
                    evalues[j] += h;
                    matrix[i][j] = 0.0;
                    for (k = 0; k < i; k++) rotate(matrix,k,i,k,j,s,tau);
                    for (k = i+1; k < j; k++) rotate(matrix,i,k,k,j,s,tau);
                    for (k = j+1; k < 3; k++) rotate(matrix,i,k,j,k,s,tau);
                    for (k = 0; k < 3; k++) rotate(evectors,k,i,k,j,s,tau);
                    }
                }
            }

        for (i = 0; i < 3; i++)
            {
            evalues[i] = b[i] += z[i];
            z[i] = 0.0;
            }
        }

    return 1;
    }

/*! Calculate the quaternion from three axes
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector
    \param quat returned quaternion
*/

void quaternionFromExyz(float4 &ex_space, float4 &ey_space, float4 &ez_space, float4 &quat)
    {

    // enforce 3 evectors as a right-handed coordinate system
    // flip 3rd evector if needed
    float ez0, ez1, ez2; // Cross product of first two vectors
    ez0 = ex_space.y * ey_space.z - ex_space.z * ey_space.y;
    ez1 = ex_space.z * ey_space.x - ex_space.x * ey_space.z;
    ez2 = ex_space.x * ey_space.y - ex_space.y * ey_space.x;

    // then dot product with the third one
    if (ez0 * ez_space.x + ez1 * ez_space.y + ez2 * ez_space.z < 0.0)
        {
        ez_space.x = -ez_space.x;
        ez_space.y = -ez_space.y;
        ez_space.z = -ez_space.z;
        }

    // squares of quaternion components
    float q0sq = 0.25 * (ex_space.x + ey_space.y + ez_space.z + 1.0);
    float q1sq = q0sq - 0.5 * (ey_space.y + ez_space.z);
    float q2sq = q0sq - 0.5 * (ex_space.x + ez_space.z);
    float q3sq = q0sq - 0.5 * (ex_space.x + ey_space.y);

    // some component must be greater than 1/4 since they sum to 1
    // compute other components from it
    if (q0sq >= 0.25)
        {
        quat.x = sqrt(q0sq);
        quat.y = (ey_space.z - ez_space.y) / (4.0 * quat.x);
        quat.z = (ez_space.x - ex_space.z) / (4.0 * quat.x);
        quat.w = (ex_space.y - ey_space.x) / (4.0 * quat.x);
        }
    else if (q1sq >= 0.25)
        {
        quat.y = sqrt(q1sq);
        quat.x = (ey_space.z - ez_space.y) / (4.0 * quat.y);
        quat.z = (ey_space.x + ex_space.y) / (4.0 * quat.y);
        quat.w = (ex_space.z + ez_space.x) / (4.0 * quat.y);
        }
    else if (q2sq >= 0.25)
        {
        quat.z = sqrt(q2sq);
        quat.x = (ez_space.x - ex_space.z) / (4.0 * quat.z);
        quat.y = (ey_space.x + ex_space.y) / (4.0 * quat.z);
        quat.w = (ez_space.y + ey_space.z) / (4.0 * quat.z);
        }
    else if (q3sq >= 0.25)
        {
        quat.w = sqrt(q3sq);
        quat.x = (ex_space.y - ey_space.x) / (4.0 * quat.w);
        quat.y = (ez_space.x + ex_space.z) / (4.0 * quat.w);
        quat.z = (ez_space.y + ey_space.z) / (4.0 * quat.w);
        }

    // Normalize
    float norm = 1.0 / sqrt(quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w);
    quat.x *= norm;
    quat.y *= norm;
    quat.z *= norm;
    quat.w *= norm;

    }

}; }; // end namespace hoomd::matrix
