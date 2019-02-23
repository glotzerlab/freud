// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef BRUTE_FORCE_H
#define BRUTE_FORCE_H

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include "Eigen/Eigen/Dense"
#include "Eigen/Eigen/Sparse"

#include "BiMap.h"

namespace freud { namespace registration {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix;

inline matrix makeEigenMatrix(const std::vector<vec3<float> >& vecs)
    {
    // build the Eigen matrix
    matrix mat;
    unsigned int size = vecs.size();
    // we know the dimension is 3 bc we're dealing with a vector of vec3's.
    mat.resize(size, 3);
    for (unsigned int i=0; i<size; i++)
        {
        mat(i,0) = vecs[i].x;
        mat(i,1) = vecs[i].y;
        mat(i,2) = vecs[i].z;
        }

    return mat;
    }

inline std::vector<vec3<float> > makeVec3Matrix(const matrix& m)
    {
    // Assume the matrix m is an Nx3 matrix.
    // If it isn't, just throw an error to be safe, rather than trying to
    // take the transpose on your own.
    // Force the user to put this in correctly.
    if (m.cols() != 3)
        {
        fprintf(stderr, "Number of columns in the input matrix is %ld\n", m.cols());
        throw std::invalid_argument("makeVec3Matrix requires the input matrix to be Nx3!");
        }
    std::vector<vec3<float> > vecs;
    for (unsigned int i=0; i<m.rows(); i++)
        {
        vec3<float> v;
        v.x = m(i, 0);
        v.y = m(i, 1);
        v.z = m(i, 2);
        vecs.push_back(v);
        }
    return vecs;
    }

inline matrix CenterOfMass(const matrix& P)
    {
    // Assumes that P = (v^T) if v is a column vector.
    // In other notation, P = [x1, y1, z1; ...]
    // p.size = (N rows, 3 cols)
    matrix cm(1, P.cols());
    for(int i =0; i < P.cols(); i++)
        cm(0,i) = P.col(i).sum()/double(P.rows());

    return cm;
    }

inline matrix Translate(const matrix& vec, const matrix& P)
    {
    matrix trans = matrix::Zero(P.rows(), P.cols());
    for(int i = 0; i < P.rows(); i++)
        trans.row(i) = P.row(i)+vec;
    return trans;
    }

inline matrix Rotate(const matrix& R, const matrix& P)
    {
    // Assume the matrix P is a 3xN matrix.
    // Then make sure that matrix R is ready to act on it
    if (R.cols() != P.rows())
        {
        fprintf(stderr, "Number of columns in the rotation matrix is %ld\n", R.cols());
        fprintf(stderr, "Number of rows in the point matrix is %ld\n", P.rows());
        throw std::invalid_argument("These values must be equal to perform the rotation!");
        }
    matrix rotated = matrix::Zero(P.rows(), P.cols());
    // Apply the rotation R.
    rotated = R*P;
    return rotated;
    }

// some helpful references:
// http://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures
// http://btk.sourceforge.net/html_docs/0.8.1/rmsd_theory.html
inline void KabschAlgorithm(const matrix& P, const matrix& Q, matrix& Rotation)
    {
    // Preconditions: P and Q have been translated to have the same center of mass.
    matrix A = P.transpose()*Q;
    // singular value decomposition (~ eigen decomposition)
    Eigen::JacobiSVD<matrix> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV );
    // A = USV^T
    matrix U = svd.matrixU();
    matrix V = svd.matrixV();

    double det = (V*U.transpose()).determinant();

    // if the rotation as we've found it, rot=VU^T, is IMPROPER, find the next best
    // (proper) rotation by reflecting the smallest principal axis in rot:
    if(det < 0)
        {
        V.col(V.cols()-1)*= -1.0;
        }
    // This is the rotation matrix that minimizes the MSD between all pairs of points P and Q.
    Rotation= V*U.transpose();
    }

inline void AlignVectorSets(matrix& P,matrix& Q, matrix* pRotation = NULL)
    {
    // Aligns p with q.
    // both p and q will be changed in this operation.

    matrix rotation;
    //Translate both p,q to origin.
    P = Translate(-CenterOfMass(P), P);
    Q = Translate(-CenterOfMass(Q), Q);
    KabschAlgorithm(P, Q, rotation); // Find the rotation.
    // Apply the rotation.
    // The rotation that we've found from the KabschAlgorithm actually acts
    // on P^T. Then we have to take the transpose again to get our matrix
    // back to its original dimensionality.
    P = (rotation*P.transpose()).transpose();  // Apply the transformation.

    if(pRotation) // optionally copy the rotation.
        *pRotation = rotation;
    }

class RegisterBruteForce
    {
    public:
        RegisterBruteForce(std::vector<vec3<float> > vecs)
            : m_rmsd(0.0), m_tol(1e-6), m_shuffles(1)
            {
            // make the Eigen matrix from vecs
            m_ref_points = makeEigenMatrix(vecs);
            }

        ~RegisterBruteForce(){}

        void Fit(std::vector<vec3<float> >& pts)
            {
            matrix points;
            matrix p, q, r;
            // make the Eigen matrix from pts
            points = makeEigenMatrix(pts);
            int num_pts;

            unsigned int N = points.rows();
            if (N != m_ref_points.rows())
                {
                fprintf(stderr, "Number of vecs to which we are matching is %ld\n", m_ref_points.rows());
                fprintf(stderr, "Number of vecs we are trying to match is %d\n", N);
                throw std::invalid_argument("Brute force matching requires the same number of points!");
                }

            RandomNumber<std::mt19937_64> rng;
            double rmsd_min = -1.0;
            for (size_t shuffles = 0; shuffles < m_shuffles; shuffles++)
                {
                int p0 = 0, p1 = 0, p2 = 0;
                while (p0 == p1 || p0 == p2 || p1 == p2)
                    {
                    p0 = rng.random_int(0, N-1);
                    if (N == int(1))
                        {
                        p1 = int(-2);
                        }
                    else
                        {
                        p1 = rng.random_int(0, N-1);
                        }

                    if (N == int(2) || N == int(1))
                        {
                        p2 = int(-1);
                        }
                    else
                        {
                        p2 = rng.random_int(0, N-1);
                        }
                    }

                size_t comb[3] = {0, 1, 2};
                if (N == int(2))
                    {
                    num_pts = 2;
                    p.resize(num_pts, m_ref_points.cols());
                    p.row(0) = m_ref_points.row(p0);
                    p.row(1) = m_ref_points.row(p1);
                    q.resize(num_pts, m_ref_points.cols());
                    }
                else if (N == int(1))
                    {
                    num_pts = 1;
                    p.resize(num_pts, m_ref_points.cols());
                    p.row(0) = m_ref_points.row(p0);
                    q.resize(num_pts, m_ref_points.cols());
                    }
                else
                    {
                    num_pts = 3;
                    p.resize(num_pts, m_ref_points.cols());
                    p.row(0) = m_ref_points.row(p0);
                    p.row(1) = m_ref_points.row(p1);
                    p.row(2) = m_ref_points.row(p2);
                    q.resize(num_pts, m_ref_points.cols());
                    }
                do
                    {
                    do
                        {
                        if (N == int(2))
                            {
                            q.row(0) = points.row(comb[0]);
                            q.row(1) = points.row(comb[1]);
                            }
                        else if (N == int(1))
                            {
                            q.row(0) = points.row(comb[0]);
                            }
                        else
                            {
                            q.row(0) = points.row(comb[0]);
                            q.row(1) = points.row(comb[1]);
                            q.row(2) = points.row(comb[2]);
                            }

                        // finds the optimal rotation of the FIRST input set
                        // of points such that they match the SECOND input
                        // set of points
                        KabschAlgorithm(q, p, r);

                        // The rotation that we've found from the
                        // KabschAlgorithm actually acts on P^T.
                        matrix rot_points = Rotate(r, points.transpose());

                        // feed back in the TRANSPOSE of rot_points such that
                        // the input matrix is (Nx3).
                        BiMap<unsigned int, unsigned int> vec_map;
                        double rmsd = AlignedRMSDTree(rot_points.transpose(), vec_map);
                        if (rmsd < rmsd_min || rmsd_min < 0.0)
                            {
                            m_rmsd = rmsd;
                            m_rotation = r;
                            m_vec_map = vec_map;
                            rmsd_min = m_rmsd;
                            if (rmsd_min < m_tol)
                                {
                                // The rotation that we've found from the KabschAlgorithm actually acts on P^T.
                                matrix ptsT = Rotate(m_rotation, points.transpose());
                                // Then we have to take the transpose again to get our matrix back to its original dimensionality.
                                pts = makeVec3Matrix(ptsT.transpose());
                                return;
                                }
                            }
                        }
                    while (std::next_permutation(comb, comb+num_pts));
                    }
                while (NextCombination(comb, N, num_pts));
                } // end for loop over shuffles
            // The rotation that we've found from the KabschAlgorithm
            // actually acts on P^T.
            matrix ptsT = Rotate(m_rotation, points.transpose());
            // Then we have to take the transpose again to get our matrix
            // back to its original dimensionality.
            pts = makeVec3Matrix(ptsT.transpose());
            return;
            }

        std::vector<vec3<float> > getRotation()
            {
            matrix R = m_rotation;
            return makeVec3Matrix(R);
            }

        std::vector<vec3<float> > getTranslation()
            {
            matrix T = m_translation;
            return makeVec3Matrix(T);
            }

        double getRMSD()
            {
            return m_rmsd;
            }

        BiMap<unsigned int, unsigned int> getVecMap()
            {
            return m_vec_map;
            }

        void setNumShuffles(size_t s)
            {
            m_shuffles = s;
            }

        void setTol(double tol)
            {
            m_tol = tol;
            }

        // This uses an R-tree to efficiently determine pairs of points that
        // are closest, next closest, etc to each other. NOTE that this does
        // not guarantee an absolutely minimal RMSD. It doesn't figure out the
        // optimal permutation of BOTH sets of vectors to minimize the RMSD.
        // Rather, it just figures out the optimal permutation of the second
        // set, the vector set used in the argument below.
        // To fully solve this, we need to use the Hungarian algorithm or some
        // other way of solving the so-called assignment problem.
        double AlignedRMSDTree(const matrix& points, BiMap<unsigned int, unsigned int>& m)
            {
            // Also brute force.
            assert(points.rows() == m_ref_points.rows());
            double rmsd = 0.0;

            // a mapping between the vectors of m_ref_points and the vectors of points
            BiMap<unsigned int, unsigned int> vec_map;

            // keeps track of whether m_ref_points have been matched to any point in points
            // guarantees 1-1 mapping
            std::set<unsigned int> unused_indices;
            for (int i = 0; i < m_ref_points.rows(); i++)
                {
                unused_indices.insert(i);
                }

            // loop through all the points
            for(int r = 0; r < points.rows(); r++)
                {
                // get the rotated point
                vec3<float> pfit = make_point(points.row(r));
                // compute squared distances to all unused reference points
                std::vector<std::pair<unsigned int, double> > ref_distances;
                for (auto ref_index : unused_indices)
                    {
                    vec3<float> ref_point = make_point(m_ref_points.row(ref_index));
                    vec3<float> delta = ref_point - pfit;
                    double rsq = dot(delta, delta);
                    ref_distances.push_back(std::pair<unsigned int, double>(ref_index, rsq));
                    }
                // sort the ref_distances from nearest to farthest
                sort(ref_distances.begin(), ref_distances.end(), compare_ref_distances);
                // take the first (nearest) ref_point found and mark it as used
                unused_indices.erase(ref_distances[0].first);
                // add this pairing to the mapping between vectors
                vec_map.emplace(ref_distances[0].first, r);
                // add this squared distance to the rmsd
                rmsd += ref_distances[0].second;
                }

            m = vec_map;
            return sqrt(rmsd/double(points.rows()));
            }

    private:
        vec3<float> make_point(const Eigen::VectorXd& row)
            {
            if(row.rows() == 2)
                return vec3<float>(row[0], row[1], 0.0);
            else if(row.rows() == 3)
                return vec3<float>(row[0], row[1], row[2]);
            else
                throw(std::runtime_error("points must 2 or 3 dimensions"));
            }

        static bool compare_ref_distances(const std::pair<unsigned int, float> &a,
                const std::pair<unsigned int, float> &b)
            {
            return (a.second < b.second);
            }

        inline bool NextCombination(size_t* comb, int N, int k)
            {
            // returns next combination.
            if (k == 0 || N == 0 || !comb)
                return false;

            bool bRetVal = false;

            for (int i = k-1; i >= 0; i--)
                {
                if (comb[i] + 1 < size_t(N+i-k+1))
                    {
                    comb[i]++;
                    for (int j = i+1; j < k; j++)
                        {
                        comb[j] = comb[j-1]+1;
                        }
                    bRetVal = true;
                    break;
                    }
                }

            return bRetVal;
            }

        template<class RNG>
        class RandomNumber
            {
            public:
                RandomNumber()
                    {
                    seed_generator();
                    }
                int random_int(int a, int b)
                    {
                    std::uniform_int_distribution<int> distribution(a, b);
                    return distribution(m_generator);
                    }
            private:
                inline void seed_generator(const size_t& n = 100)
                    {
                    std::vector<size_t> seeds;
                    try
                        {
                        std::random_device rd;
                        for (size_t i = 0; i < n; i++)
                            seeds.push_back(rd());
                        }
                    catch (...)
                        {
                        std::cout << "random_device is not available..." << std::endl;
                        seeds.push_back(size_t(
                            std::chrono::system_clock::now().time_since_epoch().count()));
                        seeds.push_back(size_t(getpid()));
                        }
                    std::seed_seq seq(seeds.begin(), seeds.end());
                    m_generator.seed(seq);
                    }
                RNG m_generator;
            };

    private:
        matrix m_ref_points;
        matrix m_rotation;
        matrix m_translation;
        double m_rmsd;
        double m_tol;
        size_t m_shuffles;
        BiMap<unsigned int, unsigned int> m_vec_map;
};

}; }; // end namespace freud::registration

#endif // BRUTE_FORCE_H
