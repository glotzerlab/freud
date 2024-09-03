// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "diagonalize.h"
#include "ManagedArray.h"
#include <Eigen/Eigen/src/Core/Matrix.h>
#include <Eigen/Eigen/src/Core/Map.h>
#include <Eigen/Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include <Eigen/Eigen/src/Core/util/Constants.h>

namespace freud { namespace util {

void diagonalize33SymmetricMatrix(const util::ManagedArray<float>& mat, util::ManagedArray<float>& eigen_vals,
                                  util::ManagedArray<float>& eigen_vecs)
{
    const Eigen::Matrix3f m = Eigen::Map<const Eigen::Matrix3f>(mat.data());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;
    es.compute(m);

    if (es.info() != Eigen::Success)
    {
        // numerical issue, return identity matrix
        Eigen::Matrix3f const id = Eigen::Matrix3f::Identity();
        Eigen::Map<Eigen::Matrix3f>(eigen_vecs.data(), 3, 3) = id;
        // set eigenvalues to zero so it's easily detectable
        eigen_vals[0] = eigen_vals[1] = eigen_vals[2] = 0.0;
    }
    else
    {
        // Note that Eigen by default stores matrices in column-major order,
        // whereas everything we do in freud uses row major ordering. As a
        // result, this operation here transposes the matrix, which is why
        // eigenvectors are returned as rows rather than columns of the
        // outputmatrix eigen_vecs.
        // See here for information:
        // https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html
        Eigen::Map<Eigen::Matrix3f>(eigen_vecs.data(), 3, 3) = es.eigenvectors();
        Eigen::Map<Eigen::Vector3f>(eigen_vals.data(), 3) = es.eigenvalues();
    }
}

}; }; // namespace freud::util
