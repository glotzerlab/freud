/*
The MIT License (MIT)

Copyright (c) 2015 Paul Dodd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// stdlib include
#include <iostream>
#include <vector>
#include <random>
// boost include
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>

// eigen include
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
// procrustes includes
#include "boost_utils.h"


#ifndef BRUTE_FORCE_H
#define BRUTE_FORCE_H

namespace procrustes{
    namespace registration{

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix;
inline matrix CenterOfMass(const matrix& P)
{
    // Assumes that P = (v**T) if v is a column vector.  or in other notation  P = [x1, y1, z1; ...]
    matrix cm(1, P.cols());
    for(int i =0; i < P.cols(); i++)
        cm(0,i) = P.col(i).sum()/double(P.rows());
    //cout << "cm = \n"<< cm << endl;

    return cm;
}

inline matrix Translate(const matrix& vec, const matrix& P)
{
    matrix trans = matrix::Zero(P.rows(), P.cols());
    for(int i = 0; i < P.rows(); i++)
        trans.row(i) = P.row(i)+vec;
    return trans;
}

inline double RMSD(const matrix& P, const matrix& Q)
{
    matrix pmq = P-Q;
    return pmq.norm()/double(P.rows());
}

inline void KabschAlgorithm(const matrix& P, const matrix& Q, matrix& Rotation)
{
    // Preconditions: P and Q have been translated to have the same center of mass.
    matrix A = P.transpose()*Q;
    Eigen::JacobiSVD<matrix> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV );
    // A = USV**T
    matrix U = svd.matrixU();
    matrix V = svd.matrixV();

    double det = (V*U.transpose()).determinant();

    if(det < 0)
    {
        //std::cout << "Vb = \n" << V << std::endl<< std::endl;
        V.col(V.cols()-1)*= -1.0;
        //std::cout << "Va = \n" << V << std::endl<< std::endl;
    }
    //Solve for the rotation matrix
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
    P = (rotation*P.transpose()).transpose();  // Apply the transform

    if(pRotation) // optionally copy the roation.
        *pRotation = rotation;
}
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

class RegisterBruteForce  // : public Register
{
    using point = bg::model::point<double, 3, bg::cs::cartesian>;
    using value = std::pair<point, unsigned int>;

    public:
        RegisterBruteForce(boost::python::list pts) : m_rmsd(0.0), m_tol(1e-6), m_shuffles(1)
        {
            utils::_2d_python_list_eigen_matrix(pts, m_data);
            for(int r = 0; r < m_data.rows(); r++)
                m_rtree.insert(std::make_pair(make_point<matrix>(m_data.row(r)), r));
            // m_data = Translate(-CenterOfMass(m_data), m_data);
        }
        ~RegisterBruteForce(){}

        bool Fit(boost::python::list pts)
        {
            matrix points;
            matrix p, q, r;
            utils::_2d_python_list_eigen_matrix(pts, points);
            // m_translation = -CenterOfMass(points);
            // points = Translate(m_translation, points);
            int N = points.rows();
            if(N != m_data.rows())
            {
                std::cout << "brute force matchign requires the same number of points" << std::endl;
                return false;
            }
            RandomNumber<std::mt19937_64> rng;
            double rmsd_min = -1.0;
            for(size_t shuffles = 0; shuffles < m_shuffles; shuffles++)
            {
                int p0 = 0, p1 = 0, p2 = 0;
                while( p0 == p1 || p0 == p2 || p1 == p2)
                {
                    p0 = rng.random_int(0,N-1);
                    p1 = rng.random_int(0,N-1);
                    p2 = rng.random_int(0,N-1);
                }

                size_t comb[3] = {0, 1, 2};
                p.resize(3,m_data.cols());
                p.row(0) = m_data.row(p0);
                p.row(1) = m_data.row(p1);
                p.row(2) = m_data.row(p2);
                q.resize(3,m_data.cols());
                do
                {
                    do {
                        q.row(0) = points.row(comb[0]);
                        q.row(1) = points.row(comb[1]);
                        q.row(2) = points.row(comb[2]);

                        KabschAlgorithm(p, q, r);
                        // matrix qfit(q.rows(), q.cols());
                        // for(int i = 0; i < qfit.rows(); i++)
                        // {
                        //     qfit.row(i) = r*(q.row(i).transpose());
                        // }
                        double rmsd = AlignedRSMDTree(points, r);
                        if(rmsd < rmsd_min || rmsd_min < 0.0)
                        {
                            rmsd_min = rmsd;
                            m_rotation = r;
                            m_rmsd = rmsd_min;
                            if(rmsd_min < m_tol)
                            {
                                return true;
                            }
                        }
                    } while ( std::next_permutation(comb,comb+3) );
                }while(NextCombination(comb, N, 3));
            }
            return true;
        }

        boost::python::list getRotation()
        {
            boost::python::list ret;
            utils::eigen_matrix_to_2d_python_list(m_rotation, ret);
            return ret;
        }

        boost::python::list getTranslation()
        {
            boost::python::list ret;
            utils::eigen_matrix_to_2d_python_list(m_translation, ret);
            return ret;
        }

        double getRMSD() { return m_rmsd; }

        void setNumShuffles(size_t s) { m_shuffles = s; }

        double AlignedRSMD(const matrix& points, const matrix& rot)
        {
            // As named we will do the brute force algorithm here as well.
            // we can optimize it later.
            assert(points.rows() == m_data.rows());
            double rmsd = 0.0;
            for(int r = 0; r < m_data.rows(); r++)
            {
                double dist = -1.0;
                for(int pr = 0; pr < points.rows(); pr++)
                {
                    Eigen::VectorXd pfit = rot*(points.row(pr).transpose());
                    Eigen::VectorXd delta = pfit - m_data.row(r).transpose();

                    // here we should really resolve the mapping and use that.
                    // this can map 2 points to one.
                    if(delta.norm() < dist || dist < 0.0)
                        dist = delta.norm();
                }
                rmsd += dist*dist;
            }

            return sqrt(rmsd/double(points.rows()));
        }

        double AlignedRSMDTree(const matrix& points, const matrix& rot)
        {
            // As named we will do the brute force algorithm here as well.
            // we can optimize it later.
            assert(points.rows() == m_data.rows());
            double rmsd = 0.0;
            std::vector<bool> found(m_data.rows(), false);
            for(int r = 0; r < points.rows(); r++)
            {
                double dist = -1.0;
                Eigen::VectorXd pfit = rot*(points.row(r).transpose());
                point query = make_point<Eigen::VectorXd>(pfit);
                for ( bgi::rtree< value, bgi::rstar<16> >::const_query_iterator it = m_rtree.qbegin(bgi::nearest(query, m_data.rows())); it != m_rtree.qend(); ++it )
                {
                    if(!found[it->second])
                    {

                        dist = bg::distance(query, it->first);
                        found[it->second] = true;
                        break;
                    }
                }


                if(dist < 0.0)
                {
                    std::cout << "nearest neighbor not found!" << std::endl;
                    throw(0x000);
                }
                rmsd += dist*dist;
            }

            return sqrt(rmsd/double(points.rows()));
        }


    private:
        template<class MatrixType>
        point make_point(const Eigen::VectorXd& row) {
            if(row.rows() == 2)
                return point(row[0], row[1], 0.0);
            else if(row.rows() == 3)
                return point(row[0], row[1], row[2]);
            else
                throw(std::runtime_error("points must 2 or 3 dimensions"));

        }
        inline bool NextCombination(size_t* comb, int N, int k)
        {
            //    returns next combination.
            if(k == 0 || N == 0 || !comb)
                return false;

            bool bRetVal = false;

            for(int i = k-1; i >= 0; i--) {
                if(comb[i] + 1 < size_t(N+i-k+1)) {
                    comb[i]++;
                    for (int j = i+1; j < k; j++) {
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
            RandomNumber() { seed_generator(); }
            int random_int(int a, int b)
            {
                std::uniform_int_distribution<int> distribution(a,b);
                return distribution(m_generator);
            }
        private:
            inline void seed_generator(const size_t& n = 100)
            {
                std::vector<size_t> seeds;
                try {
                    std::random_device rd;
                    for(size_t i = 0; i < n; i++)
                        seeds.push_back(rd());
                } catch (...) {
                    std::cout << "random_device is not available..." << std::endl;
                    seeds.push_back(size_t(std::chrono::system_clock::now().time_since_epoch().count()));
                    seeds.push_back(size_t(getpid()));
                }
                std::seed_seq seq(seeds.begin(), seeds.end());
                m_generator.seed(seq);
            }
            RNG m_generator;
        };

    private:
        matrix m_data;
        matrix m_rotation;
        matrix m_translation;
        double m_rmsd;
        double m_tol;
        size_t m_shuffles;
        bgi::rtree< value, bgi::rstar<16> > m_rtree;
};

void export_RegisterBruteForce()
{
    boost::python::class_<RegisterBruteForce, boost::shared_ptr<RegisterBruteForce>, boost::noncopyable>
    ("BruteForce", boost::python::init< boost::python::list
                                 /* other initializers here */>())
    .def("fit", &RegisterBruteForce::Fit)
    .def("getRotation", &RegisterBruteForce::getRotation)
    .def("getTranslation", &RegisterBruteForce::getTranslation)
    .def("getRMSD", &RegisterBruteForce::getRMSD)
    .def("setNumShuffles", &RegisterBruteForce::setNumShuffles)
    ;
}


}}


#endif
