// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <complex>
#include <cstring>
#include <stdexcept>

#include "SymmetryCollection.h"
#include "VectorMath.h"

using namespace std;

/*! \file SymmetricOrientation.cc
    \brief Compute the symmetric orientation.
*/

namespace freud { namespace symmetry {

    SymmetryCollection::SymmetryCollection(unsigned int maxL) {
        m_symmetric_orientation = quat<float>();
        m_maxL = maxL;
    }

    SymmetryCollection::~SymmetryCollection() {
    }

    shared_ptr<float > SymmetryCollection::getMlm() {
        return m_Mlm;
    }

    float SymmetryCollection::measure(shared_ptr<float> Mlm, unsigned int type) { // 1. possible that Mlm past in does not fit our Mlm format? or make it private later?
                                                                                  // 2. memory leak: new and delete? 3. lots of casting
        float select = 0.0;
        float all = 0.0;
        int numSelect = 0;
        int numAll = 0;
     
        if (Mlm.get()[0] == 0.0f) {
            return 0.0;
        }

        for (int l = 2; l < (m_maxL + 1); l += 2) {
           int count = -l - 1;
            for (int m = l * l; m < (l + 1) * (l + 1); ++m) {
                ++numAll;
                all += Mlm.get()[m] * Mlm.get()[m];
                ++count;
                if ((type == TOTAL) || (type == AXIAL && m == l * (l + 1)) || 
                   (type == MIRROR && m >= l * (l + 1)) || 
                   (type >= 2 && count % type == 0)) {
                    ++numSelect;
                    select += Mlm.get()[m] * Mlm.get()[m];
                }


            }
        }


        if (type == TOTAL || type == AXIAL) {
            return (select / Mlm.get()[0] / (float)numSelect - (float)1.0);
        }

        if (type == MIRROR || type >= 2) {
            return (select / all - (float)numSelect / (float)numAll) / ((float)1.0 - (float)numSelect / (float)numAll);
        }

        return 0.0;
        
    }

    void SymmetryCollection::computeMlm(const box::Box& box,
                                        const vec3<float> *points,
                                        const freud::locality::NeighborList *nlist,
                                        unsigned int Np) {

        m_box = box;
        unsigned int Nbonds = nlist->getNumBonds();
        const size_t *neighbor_list = nlist->getNeighbors();
        m_Np = Np;

        // Initialize Mlm as a vector with all 0s
        m_Mlm = shared_ptr<float >(new float[(m_maxL + 1) * (m_maxL + 1)],
                                           default_delete<float[]>());

        memset((void*)m_Mlm.get(), 0, sizeof(float)*((m_maxL + 1) * (m_maxL + 1))); // better to put in constructor since this is the initialization?

        vector<vec3<float> > delta;
        vector<float> phi;
        vector<float> theta;

        // Resize delta, phi, theta to Nbonds
        delta.resize(Nbonds);
        phi.resize(Nbonds);
        theta.resize(Nbonds);

        // Fill in delta vector
        for (unsigned int i = 0; i < m_Np; ++i) {
            for (size_t bond = 0; bond < Nbonds && neighbor_list[2 * bond] == i; ++bond) {
                const unsigned int j = neighbor_list[2 * bond + 1];

                if (i != j) {
                    delta[i] = m_box.wrap(points[j] - points[i]);
                    phi[i] = atan2(delta[i].y, delta[i].x);
                    theta[i] = acos(delta[i].z / sqrt(dot(delta[i], delta[i])));
                }

            }
        }

        fsph::PointSPHEvaluator<float> eval(m_maxL);
        for(unsigned int i = 0; i < Nbonds; ++i) {
            unsigned int l0_index = 0;
            unsigned int l = 0;
            unsigned int m = 0;
            eval.compute(phi[i], theta[i]);
            for(typename fsph::PointSPHEvaluator<float>::iterator iter(eval.begin(true)); iter != eval.end(); ++iter) {
                if (m > l) {
                    l++;
                    l0_index += 2*l;
                    m = 0;
                }
                if (m == 0) {
                    m_Mlm.get()[l0_index + m] += (*iter).real();
                } else {
                    m_Mlm.get()[l0_index + m] += sqrt(2) * (*iter).real();
                    m_Mlm.get()[l0_index - m] += sqrt(2) * (*iter).imag();
                }
                m++;
            }
        }
    }


    quat<float> SymmetryCollection::getSymmetricOrientation() {
        return m_symmetric_orientation;
    }

    void SymmetryCollection::compute(const box::Box& box,
                                     const vec3<float> *points,
                                     const freud::locality::NeighborList *nlist,
                                     unsigned int Np) {
        assert(points);
        assert(Np > 0);
        nlist->validate(Np, Np);
        //const size_t *neighbor_list(nlist->getNeighbors());

        computeMlm(box, points, nlist, Np);
    }

}; }; // end namespace freud::symmetry
