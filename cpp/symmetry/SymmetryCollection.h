// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <memory>
#include <ostream>
#include <tbb/tbb.h>
#include <cmath>

#include "VectorMath.h"
#include "fsph/src/spherical_harmonics.hpp"
#include "LinkCell.h"

using namespace std;

#ifndef _SYMMETRY_COLLECTION_H__
#define _SYMMETRY_COLLECTION_H__

/*! \file SymmetricOrientation.h
    \brief Compute the symmetric orientation
*/

namespace freud { namespace symmetry {

struct FoundSymmetry {
    int n; //type
    vec3<float> v;
    quat<float> q;
    float measured_order;
};

struct Symmetry {
    int type;
    float threshold;
};

//! Compute the symmetry axes
/*!
*/
class SymmetryCollection {
    public:
        //! Constructor, set default value maxL to 30
        SymmetryCollection(unsigned int maxL = 30);

        //! Destructor
        ~SymmetryCollection();

        //! Compute symmetry axes
        void compute(const box::Box& box,
                     const vec3<float> *points,
                     const freud::locality::NeighborList *nlist,
                     unsigned int Np);

        //! fill in Mlm table
        float measure(int type);

        //! rotate the axis and search for higher order
        void optimize(bool perpendicular, FoundSymmetry *symm);



        //! rotate Mlm array by a certain quat
        void rotate(const quat<float> &q);


        //! Returns quaternions for all detected symmetry axes
        quat<float>* getOrderQuats();


        //! a getter to Mlm
        std::shared_ptr<float> getMlm() {
            return m_Mlm;
        }

        //! return rotated Mlm
        std::shared_ptr<float> getMlm_rotated() {
            return m_Mlm_rotated;
        }

        //! return the number of particles in a box
        unsigned int getNP() {
            return m_Np;
        }

        //! return angular quantum number
        unsigned int getMaxL() {
            return m_maxL;
        }

        vector<FoundSymmetry> getSymmetries() {
            return m_found_symmetries;
        }

        //identify Laue group
        string getLaueGroup();

        //determine the crystal system
        string getCrystalSystem();



    private:
        //! Compute spherical harmonics from bond array. private function
        //! Coordinate Descent: https://en.wikipedia.org/wiki/Coordinate_descent
        void computeMlm(const box::Box& box,
                        const vec3<float> *points,
                        const freud::locality::NeighborList *nlist,
                        unsigned int Np);

        //! search for the best symmetry
        void searchSymmetry(FoundSymmetry *foundsym = nullptr);

        // take in a vector and do a mirror reflection around z-direction
        quat<float> initMirrorZ(const vec3<float> &p);

        //a helper indexing function
        //http://web.cmb.usc.edu/people/alber/Software/tomominer/docs/cpp/group__wigner.html
        int WDindex(int j, int mprime, int m);

        //! convert a quaternion(r, v.x, v.y, v.z) to eulerAngle (phi, theta, psi)
        //! https://en.wikiversity.org/wiki/PlanetPhysics/Direction_Cosine_Matrix_to_Euler_323_Angles
        //! http://mathworld.wolfram.com/EulerAngles.html
        vector<float> toEulerAngles323(const quat<float> &q);

        //! modifies q
        //! nomalizes a quaternion.
        //! https://www.mathworks.com/help/aeroblks/quaternionnormalize.html
        quat<float> normalize(quat<float> &q) {
            float norm = 1.0f / sqrt(norm2(q));
            q.s *= norm;
            q.v.x *= norm;
            q.v.y *= norm;
            q.v.z *= norm;
            return q;
        }



    private:
        box::Box m_box;
        unsigned int m_maxL;
        quat<float> m_symmetric_orientation;
        shared_ptr<float> m_Mlm; // a 1D matrix of float
        shared_ptr<float> m_Mlm_rotated;//hold the rotated Mlm
        unsigned int m_Np;
        const int TOTAL = -1;
        const int AXIAL = 0;
        const int MIRROR = 1;
        const int TWOFold = 2;
        const int THREEFold = 3;
        const int FOURFold = 4;
        const int FIVEFold = 5;
        const int SIXFold = 6;
        const int EIGHTFold = 8;
        const int TENFold = 10;
        const int TWELVEFold = 12;
        const int MAXL = 30;
        vector<float> WDTable;

        const float PI = atan(1.0) * 4.0;
         //<type, vertex, quat, measured_order>
        vector<FoundSymmetry> m_found_symmetries; //saves the axis of symmetry that was found,

        const int LENGTH = 11; // numbers of types of symmetry
        Symmetry SYMMETRIES[11];

        const float OPTIMIZESTART = 0.5;
        const float OPTIMIZEEND   = 1e-6;
        const float OPTIMIZESCALE = 0.9;




};








}; }; // end namespace freud::symmetry

#endif // _SYMMETRY_COLLECTION_H__
