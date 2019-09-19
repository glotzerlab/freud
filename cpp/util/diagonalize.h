#ifndef DIAGONALIZE_H
#define DIAGONALIZE_H

#include "Eigen/Eigen/Dense"
#include "ManagedArray.h"

namespace freud { namespace util {
// Sets eigen_vals and eigen_vecs to be the
// eigenvalues and eigenvectors of mat in increasing order
void diagonalize33SymmetricMatrix(const util::ManagedArray<float> &mat, util::ManagedArray<float> &eigen_vals, util::ManagedArray<float> &eigen_vecs);

}; }; // namespace freud::util
#endif
