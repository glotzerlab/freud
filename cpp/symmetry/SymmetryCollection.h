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

//! Compute the symmetric orientation
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


        struct Symmetry {
            int type;
            float threshold;
        };

        //! fill in Mlm table
        float measure(int type);

        //! rotate the axis ans search for higher order
        float optimize(bool perpendicular, Symmetry *symm);

        //! search for the best symmetry
        int searchSymmetry(bool perpendicular);

        //! detect symmetries
        void symmetrize(bool onlyLocal = false);
       
        //! rotate Mlm array by a certain quat
        void rotate(const quat<float> &q);//private function

        //! Returns quaternion corresponding to the highest-symmetry axis
        quat<float> getHighestOrderQuat();

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

        //from symmetrize(), identify Laue group
        void toLaueGroup(); 


       

        quat<float> getHighestSymmetryQuat() {
            return m_highest_symm_quat;
        }

        

        string getType(int type);

        
    private:
        //! Compute spherical harmonics from bond array. private function
        void computeMlm(const box::Box& box,
                        const vec3<float> *points,
                        const freud::locality::NeighborList *nlist,
                        unsigned int Np);

        // take in a vector and do a mirror reflection around z-direction
        quat<float> initMirrorZ(const vec3<float> &p);

        //a helper indexing function
        int WDindex(int j, int mprime, int m);//private function

        //! convert a quaternion(r, v.x, v.y, v.z) to eulerAngle (phi, theta, psi)
        //! https://en.wikiversity.org/wiki/PlanetPhysics/Direction_Cosine_Matrix_to_Euler_323_Angles
        //! http://mathworld.wolfram.com/EulerAngles.html
        vector<float> toEulerAngles323(const quat<float> &q);//private function

        //! returns a pointer to a symmetry
        Symmetry *findSymmetry(int type);

        //! modifies q
        //! nomalizes a quaternion.
        //! https://www.mathworks.com/help/aeroblks/quaternionnormalize.html
        quat<float> normalize(quat<float> &q) { //private helper function, normalize a quaternion.
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
        std::shared_ptr<float> m_Mlm; // a 1D matrix of float
        std::shared_ptr<float> m_Mlm_rotated;//hold the rotated Mlm
        unsigned int m_Np;
        quat<float> m_highest_symm_quat; // rotate the view after the global detecton == gData.rotation
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
       
        const bool USETABLES = true;
        const int TABLESIZE = 1024;
        const float PI = atan(1.0) * 4.0;
        vector<pair<int, vec3<float> > > m_found_symmetries; //saves the axis of symmetry that was found

        const int LENGTH = 11; // numbers of types of symmetry
        Symmetry SYMMETRIES[11];

        const float OPTIMIZESTART = 0.5;
        const float OPTIMIZEEND   = 1e-6;
        const float OPTIMIZESCALE = 0.9;

        vector<int> NAxis;
        vector<quat<float> > m_found_quats;
        vector<vec3<float> > vertex;



};


       





}; }; // end namespace freud::symmetry

#endif // _SYMMETRY_COLLECTION_H__
