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

    SymmetryCollection::SymmetryCollection() {
        m_symmetric_orientation = quat<float>();

    }

    SymmetryCollection::~SymmetryCollection() {
    }

    shared_ptr<complex<float> > SymmetryCollection::getMlm() {
        return m_Mlm;
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
        m_Mlm = shared_ptr<complex<float> >(new complex<float>[(MAXL + 1) * (MAXL + 1)],
                                           default_delete<complex<float>[]>());

        memset((void*)m_Mlm.get(), 0, sizeof(complex<float>)*((MAXL + 1) * (MAXL + 1)));

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

        fsph::PointSPHEvaluator<float> eval(MAXL);
        for(unsigned int i = 0; i < Nbonds; ++i) {
            unsigned int j = 0;
            eval.compute(phi[i], theta[i]);
            for(typename fsph::PointSPHEvaluator<float>::iterator iter(eval.begin(true)); iter != eval.end(); ++iter) {
                m_Mlm.get()[j] += *iter;
                ++j;
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
