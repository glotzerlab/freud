// Copyright (c) 2010-2023 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef DIAGONALIZE_H
#define DIAGONALIZE_H

#include "Eigen/Eigen/Dense"
#include "ManagedArray.h"

namespace freud { namespace util {
//! Compute eigenvalues and eigenvectors of a self-adjoint 3x3 matrix.
/*! The eigen_vals and eigen_vecs arguments should be references to
 * ManagedArrays that will be updated by reference. The eigenvectors are placed
 * in rows of eigen_vecs, so e.g. the first eigenvector is [eigen_vecs(0, 0),
 * eigen_vecs(0, 1), eigen_vecs(0, 2)]. The eigenvalues are returned in
 * increasing order, with the eigenvectors in the corresponding order.
 *
 * Note that no checks are performed to check if the matrix is symmetric. It is
 * the responsibility of calling code to only use this function for symmetric
 * matrices.
 *
 *  \param mat The matrix to diagonalize.
 *  \param eigen_vals The eigenvalues (set to 0 if the solver fails).
 *  \param eigen_vecs Matrix with eigenvectors as the rows (set to the identity if the solver fails).
 */
void diagonalize33SymmetricMatrix(const util::ManagedArray<float>& mat, util::ManagedArray<float>& eigen_vals,
                                  util::ManagedArray<float>& eigen_vecs);

}; }; // namespace freud::util
#endif
