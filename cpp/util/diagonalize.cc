#include "diagonalize.h"

namespace freud { namespace util {

void diagonalize33SymmetricMatrix(float mat[9], float eigen_vals[3], float eigen_vecs[9])
{
    Eigen::Matrix3f m;
    Index2D a_i = Index2D(3);
    for (unsigned int i = 0; i < 3; ++i)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            m(i, j) = mat[a_i(i, j)];
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es;
    es.compute(m);

    if (es.info() != Eigen::Success)
    {
        // numerical issue, set r to identity matrix
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                if (i == j)
                {
                    eigen_vecs[a_i(i, j)] = 1.0;
                }
                else
                {
                    eigen_vecs[a_i(j, j)] = 0.0;
                }
        // set order parameter to zero so it's easily detectable
        eigen_vals[0] = eigen_vals[1] = eigen_vals[2] = 0.0;
    }
    else
    {
        // columns are eigenvectors
        Eigen::Matrix3f eigen_vec = es.eigenvectors();
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                eigen_vecs[a_i(i, j)] = eigen_vec(i, j);
        auto eigen_val = es.eigenvalues();
        eigen_vals[0] = eigen_val(0);
        eigen_vals[1] = eigen_val(1);
        eigen_vals[2] = eigen_val(2);
    }
}

}; };
