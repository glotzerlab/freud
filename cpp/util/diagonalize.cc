#include "diagonalize.h"

namespace freud { namespace util {

void diagonalize33SymmetricMatrix(float mat[9], float eigen_vals[3], float eigen_vecs[9])
{
    Eigen::Matrix3f m = Eigen::Map<Eigen::Matrix3f>(mat);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;
    es.compute(m);

    if (es.info() != Eigen::Success)
    {
        // numerical issue, return identity matrix
        Eigen::Matrix3f id = Eigen::Matrix3f::Identity();
        Eigen::Map<Eigen::Matrix3f>(eigen_vecs, 3, 3) = id;
        // set eigenvalues to zero so it's easily detectable
        eigen_vals[0] = eigen_vals[1] = eigen_vals[2] = 0.0;
    }
    else
    {
        Eigen::Map<Eigen::Matrix3f>(eigen_vecs, 3, 3) = es.eigenvectors();
        Eigen::Map<Eigen::Vector3f>(eigen_vals, 3) = es.eigenvalues();
    }
}

}; }; // namespace freud::util
