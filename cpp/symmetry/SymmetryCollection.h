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
class SymmetryCollection
    {
    public:
        //! Constructor, set default value maxL to 30
        SymmetryCollection(unsigned int maxL = 30);

        //! Destructor
        ~SymmetryCollection();

        //! Get the symmetric orientation
        quat<float> getSymmetricOrientation();

        //! Compute spherical harmonics from bond array. private function
        void computeMlm(const box::Box& box,
                        const vec3<float> *points,
                        const freud::locality::NeighborList *nlist,
                        unsigned int Np);

        //! Compute symmetry axes
        void compute(const box::Box& box,
                     const vec3<float> *points,
                     const freud::locality::NeighborList *nlist,
                     unsigned int Np);
       

        quat<float> initMirrorZ(const vec3<float> &p);


        

        int WDindex(int j, int mprime, int m);//private function

        vector<float> toEulerAngles323(const quat<float> &q);//private function

       






        //! Returns quaternion corresponding to the highest-symmetry axis
        quat<float> getHighestOrderQuat();

        //! Returns quaternions for all detected symmetry axes
        quat<float>* getOrderQuats();



        std::shared_ptr<float> getMlm() {
            // cout << "getMlm() Test starts" << endl;
            // int a = 0;
            // for (int l = 0; l < (m_maxL + 1) * (m_maxL + 1); ++l) {
            //     cout << m_Mlm.get()[l] << " ";
            //     if(l == a*a+2*a) {
            //         ++a;
            //         cout << endl;
            //     }
            // }
            // cout <<endl;
            // cout << "getMlm() Test ends" << endl;

            return m_Mlm;
        }

        std::shared_ptr<float> getMlm_rotated() {
        //     cout << "getMlm_rotated() Test starts" << endl;
        // int a = 0;
        // for (int l = 0; l < (m_maxL + 1) * (m_maxL + 1); ++l) {
        //     cout << m_Mlm_rotated.get()[l] << " ";
        //     if( l == a*a+2*a) {
        //         ++a;
        //         cout << endl;
        //     }
        // }
        // cout <<endl;
        // cout << "getMlm_rotated() Test ends" << endl;

            return m_Mlm_rotated;
        }
        
        unsigned int getNP() {
            return m_Np;
        }

        unsigned int getMaxL() {
            return m_maxL;
        }

        void toLaueGroup(); //from symmetrize(), identify Laue group


        struct Symmetry {
            int type;
            float threshold;
        };

        //! fill in Mlm table
        float measure(std::shared_ptr<float> Mlm, int type);

        float optimize(bool perpendicular, Symmetry *symm);

        Symmetry *findSymmetry(int type);//need to be private at last

        //helper functions
        //rotate Mlm array by certain quat
        void rotate(const quat<float> &q);//private function

        int searchSymmetry(bool perpendicular);

        void symmetrize(bool onlyLocal);

        quat<float> getHighestSymmetryQuat() {
            return m_highest_symm_quat;
        }

        quat<float> normalize(quat<float> &q) { //private helper function, normalize a quaternion.
            float norm = 1.0f / sqrt(norm2(q));
            //float norm = 1.0f / sqrt(q.s * q.s + q.v.i * q.v.i + q.v.j * q.v.j + q.v.k * q.v.k);
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
        std::shared_ptr<float> m_Mlm;
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
        //float WDTable[(1/6) * (1 + 30) * (2 + 30) * (3 + 2 * 30)];
        const bool USETABLES = true;
        const int TABLESIZE = 1024;
        const float PI = atan(1.0) * 4.0;
        vector<pair<int, vec3<float> > > m_found_symmetries; //saves the axis of symmetry that was found

        const int LENGTH = 11; // numbers of types of symmetry
        Symmetry SYMMETRIES[11];

        const float OPTIMIZESTART = 0.5;
        const float OPTIMIZEEND   = 1e-6;
        const float OPTIMIZESCALE = 0.9;
        bool debug = false;


    };


       





}; }; // end namespace freud::symmetry

#endif // _SYMMETRY_COLLECTION_H__
