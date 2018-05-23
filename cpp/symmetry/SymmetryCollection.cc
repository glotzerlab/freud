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

    SymmetryCollection::SymmetryCollection(unsigned int maxL) :
    m_maxL(maxL), m_symmetric_orientation(quat<float>())
    {
        m_Mlm = shared_ptr<float>(new float[(m_maxL + 1) * (m_maxL + 1)],
                                           default_delete<float[]>());

        memset((void*)m_Mlm.get(), 0, sizeof(float)*((m_maxL + 1) * (m_maxL + 1)));

    }

    SymmetryCollection::~SymmetryCollection() {
    }

    shared_ptr<float > SymmetryCollection::getMlm() {
        
        cout << endl << "getMlm() starts" << endl;
        int c = 0;
        for (int l = 0; l < (m_maxL + 1) * (m_maxL + 1); ++l) {
            cout << m_Mlm.get()[l] << " ";
            if( l == c*c+2*c) {
                ++c;
                cout << endl;
            }
        }
        cout <<endl;
        cout << endl << "getMlm() ends" << endl << endl;

        return m_Mlm;
    }

    float SymmetryCollection::measure(shared_ptr<float> Mlm, int type) { // 1. possible that Mlm past in does not fit our Mlm format? or make it private later?
                                                                        
        float select = 0.0;
        float all = 0.0;
        int numSelect = 0;
        int numAll = 0;

        //cout << "measure() starts" << endl;
        //cout << "only for test part11" << endl;
        // int a = 0;
        // for (int l = 0; l < (m_maxL + 1) * (m_maxL + 1); ++l) {
        //     cout << m_Mlm.get()[l] << " ";
        //     if( l == a*a+2*a) {
        //         ++a;
        //         cout << endl;
        //     }
        // }
        // cout <<endl;
        // cout << "only for test part ends11" << endl;

        if (m_Mlm.get()[0] == 0.0f) {
            //cout << "measure test ends 0 " << endl;
            return 0.0;
        }
        //cout << "measure test starts" << endl;
        for (int l = 2; l < (m_maxL + 1); l += 2) {
            //cout << "l is: " << l << endl;
           int count = -l - 1;
            for (int m = l * l; m < (l + 1) * (l + 1); ++m) {
                ++numAll;
               // cout << "numAll is: " << numAll << "   m is: " << m << "  ";
                all += m_Mlm.get()[m] * m_Mlm.get()[m];
              //  cout << "all: " << all << "  ";
                ++count;
                if ((type == TOTAL) || 
                    (type == AXIAL && (m == l * (l + 1))) || 
                    (type == MIRROR && (m >= l * (l + 1))) || 
                    (type >= 2 && (count % type == 0))) {
                    ++numSelect;
                  //  cout << "numSelect is: " << numSelect << "   m is: " << m << "  ";
                    select += m_Mlm.get()[m] * m_Mlm.get()[m];
                  //  cout << "select is: " << select << "  ";
                }
                //cout << endl;
            }
            //cout << endl;
        }
        // cout << endl;
        // cout << "only for test part" << endl;
        // for (int l = 2; l < (m_maxL + 1); l+=2) {
        //     for (int m = l * l; m < (l+1)*(l+1); ++m) {
        //         cout << m_Mlm.get()[m] << " ";
        //     }
        //     cout << endl;
        // }
        // int b = 0;
        // for (int l = 0; l < (m_maxL + 1) * (m_maxL + 1); ++l) {
        //     cout << m_Mlm.get()[l] << " ";
        //     if( l == b*b+2*b) {
        //         ++b;
        //         cout << endl;
        //     }
        // }
        // cout <<endl;
        // cout << "only for test part ends" << endl;

        if (type == TOTAL || type == AXIAL) {
            //cout << "measure test ends 1 " << endl;
            //cout << "return value: " << (select / m_Mlm.get()[0] / (float)numSelect - 1.0f) << endl;
            return (select / m_Mlm.get()[0] / (float)numSelect - 1.0f);
        }

        if (type == MIRROR || type >= 2) {
            //cout << "measure test ends 2 " << endl;
            //cout << "return value: " << (select / all - (float)numSelect / (float)numAll) / (1.0f - (float)numSelect / (float)numAll) << endl;
            return (select / all - (float)numSelect / (float)numAll) / (1.0f - (float)numSelect / (float)numAll);
        }
        //cout << "measure test ends 3 " << endl;
        return 0.0;
        
    }

    void SymmetryCollection::computeMlm(const box::Box& box,
                                        const vec3<float> *points,
                                        const freud::locality::NeighborList *nlist,
                                        unsigned int Np) {

        m_box = box;
        unsigned int Nbonds = nlist->getNumBonds();
        const size_t *neighbor_list = nlist->getNeighbors();

        vector<vec3<float> > delta;
        vector<float> phi;
        vector<float> theta;

        // Resize delta, phi, theta to Nbonds
        delta.resize(Nbonds);
        phi.resize(Nbonds);
        theta.resize(Nbonds);

        // Fill in delta vector
        size_t bond = 0;
        size_t valid_bonds = 0;
        for (unsigned int i = 0; i < Np; ++i) {
            for (; bond < Nbonds && neighbor_list[2 * bond] == i; ++bond) {
                const unsigned int j = neighbor_list[2 * bond + 1];

                if (i != j) {
                    delta[valid_bonds] = m_box.wrap(points[j] - points[i]);
                    phi[valid_bonds] = atan2(delta[valid_bonds].y, delta[valid_bonds].x);
                    theta[valid_bonds] = acos(delta[valid_bonds].z / sqrt(
                        dot(delta[valid_bonds], delta[valid_bonds])));
                    valid_bonds++;
                }
            }
        }


        // cout << "only for test part in computermlm" << endl;
        // cout << "before anything starts.. " << endl;
        // int a = 0;
        // for (int l = 0; l < (m_maxL + 1) * (m_maxL + 1); ++l) {
        //     cout << m_Mlm.get()[l] << " ";
        //     if( l == a*a+2*a) {
        //         ++a;
        //         cout << endl;
        //     }
        // }
        // cout <<endl;
        // cout << "only for test part ends in computerMlm" << endl;

        //cout << "fsph test starts" << endl;
        fsph::PointSPHEvaluator<float> eval(m_maxL);
        for(unsigned int i = 0; i < valid_bonds; ++i) {
            unsigned int l0_index = 0;
            unsigned int l = 0;
            unsigned int m = 0;
            eval.compute(phi[i], theta[i]);
            for(typename fsph::PointSPHEvaluator<float>::iterator iter(eval.begin(false));
                iter != eval.end(); ++iter) {
                if (m > l) {
                    l++;
                    l0_index += 2*l;
                    m = 0;
                }
                if (m == 0) {
                    m_Mlm.get()[l0_index] += (*iter).real();
                    //cout <<"fsph eval2: " << m_Mlm.get()[l0_index] << " " << endl;
                } else {
                    
                    m_Mlm.get()[l0_index + m] += sqrt(2) * (*iter).real();
                    //cout <<"fsph l0_index + m: " << m_Mlm.get()[l0_index + m]<< " " << endl;
                    m_Mlm.get()[l0_index - m] += sqrt(2) * (*iter).imag();
                    //cout <<"fsph l0_index - m: " << m_Mlm.get()[l0_index - m] << " " << endl;
                }
                m++;
            }
        }
        // cout << "only for test part in computermlm2" << endl;
        // cout << "after fsph ends.. " << endl;
        // int b = 0;
        // for (int l = 0; l < (m_maxL + 1) * (m_maxL + 1); ++l) {
        //     cout << m_Mlm.get()[l] << " ";
        //     if( l == b*b+2*b) {
        //         ++b;
        //         cout << endl;
        //     }
        // }
        // cout <<endl;
        // cout << "only for test part ends in computerMlm2" << endl;
        //  cout << "fsph test ends" << endl;
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


    // quat<float> initMirrorZ(vec3<float> p) {
    //     float x = p.x;
    //     float y = p.y;
    //     float z = p.z + 1.0;
    //     float n = Math.sqrt(x * x + y * y + z * z);
    //     if (n == 0.0) return null;
    //     quat<float> temp = 
    //     return temp;

    //     return m_symmetric_orientation;


    //     return m_symmetric_orientation
    // }

    // int searchSymmetry(bool perpendicular);






}; }; // end namespace freud::symmetry
