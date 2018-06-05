// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <complex>
#include <cstring>
#include <stdexcept>
#include <cmath>

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

        m_Mlm_rotated = shared_ptr<float>(new float[(m_maxL + 1) * (m_maxL + 1)],
                                           default_delete<float[]>());

        memset((void*)m_Mlm_rotated.get(), 0, sizeof(float)*((m_maxL + 1) * (m_maxL + 1)));
        int a = (1/6) * (1 + m_maxL) * (2 + m_maxL) * (3 + 2*m_maxL);

        cout << "constructor starts" << endl;
        for (int l = 0; l < (m_maxL + 1) * (m_maxL + 1); ++l) {
            cout << m_Mlm_rotated.get()[l] << " ";
            if( l == a*a+2*a) {
                ++a;
                cout << endl;
            }
        }
        cout <<endl;
        cout << "constructor ends" << endl;


        WDTable.resize(10416);
        rot.s = 1.0f;
        rot.v = {0.0, 0.0, 0.0};

    }

    SymmetryCollection::~SymmetryCollection() {
    }

    // shared_ptr<float> SymmetryCollection::getMlm() {
        
    //    // cout << endl << "getMlm() starts" << endl;
    //     int c = 0;
    //     for (int l = 0; l < (m_maxL + 1) * (m_maxL + 1); ++l) {
    //         //cout << m_Mlm.get()[l] << " ";
    //         if( l == c*c+2*c) {
    //             ++c;
    //             //cout << endl;
    //         }
    //     }
    //     //cout <<endl;
    //    // cout << endl << "getMlm() ends" << endl << endl;

    //     return m_Mlm;
    // }

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
            return 0.0f;
        }
        //cout << "measure test starts" << endl;
        for (int l = 2; l < (m_maxL + 1); l += 2) {
            //cout << "l is: " << l << endl;
           int count = -l - 1;
            for (int m = l * l; m < (l + 1) * (l + 1); ++m) {
                ++numAll;
                all += m_Mlm.get()[m] * m_Mlm.get()[m];
                ++count;
                if ((type == TOTAL) || 
                    (type == AXIAL && (m == l * (l + 1))) || 
                    (type == MIRROR && (m >= l * (l + 1))) || 
                    (type >= 2 && (count % type == 0))) {
                    ++numSelect;
                    select += m_Mlm.get()[m] * m_Mlm.get()[m];
                }
            }
        }

        if (type == TOTAL || type == AXIAL) {
            return (select / m_Mlm.get()[0] / (float)numSelect - 1.0f);
        }

        if (type == MIRROR || type >= 2) {
            return (select / all - (float)numSelect / (float)numAll) / (1.0f - (float)numSelect / (float)numAll);
        }
        return 0.0f;
        
    }

    void SymmetryCollection::computeMlm(const box::Box& box,
                                        const vec3<float> *points,
                                        const freud::locality::NeighborList *nlist,
                                        unsigned int Np) {
        //cout << endl << "beginning" << endl;
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
        //cout << "ending" << endl;


        // cout << "1.check computermlm starts1" << endl;
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
        // cout << "1.check computerMlm ends" << endl;

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
                    l0_index += 2 * l;
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
        // cout << "fsph test ends" << endl;
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

        computeMlm(box, points, nlist, Np);
    }

    //utility function
    int SymmetryCollection::WDindex(int j, int mprime, int m) {
        int jpart = j * (j + 1) * (2 * j + 1) / 6;
        int mprimepart = mprime * (mprime + 1);
        return jpart + mprimepart + m;
    }

    // pass in an alias to q. 
    vector<float> SymmetryCollection::toEulerAngles323(const quat<float> &q) {
        float q00 = q.s * q.s;
        float q11 = q.v.x * q.v.x;
        float q22 = q.v.y * q.v.y;
        float q33 = q.v.z * q.v.z;
        float q01 = q.s * q.v.x;
        float q02 = q.s * q.v.y;
        float q03 = q.s * q.v.z;
        float q12 = q.v.x * q.v.y;
        float q13 = q.v.x * q.v.z;
        float q23 = q.v.y * q.v.z;
        float angle1 = atan2(2.0 * (q23 - q01), 2.0 * (q13 + q02));
        float angle2 = acos (q00 - q11 - q22 + q33);
        float angle3 = atan2(2.0 * (q01 + q23), 2.0 * (q02 - q13));
        cout << "pre-angle is: " << angle1 << " "  << angle2 << " " << angle3 << endl;
        if (q01 == 0.0 && q23 == 0.0 && q13 == 0.0 && q02 == 0.0) {
            // degenerate case, rotation around z-axis only
            angle1 = atan2(2.0 * (q03 - q12), q00 + q11 - q22 - q33);
            angle2 = 0.0;
            angle3 = 0.0;
        }
        cout << "Pi is: " << PI << endl;
        cout << "mid-angle is: " << angle1 << " "  << angle2 << " " << angle3 << endl;
        if (isnan(angle1)) angle1 = 0.5 * PI;
        if (isnan(angle2)) angle2 = 0.5 * PI;
        if (isnan(angle3)) angle3 = 0.5 * PI;
        cout << "after-angle is: " << angle1 << " "  << angle2 << " " << angle3 << endl;
        vector<float> angles{angle1, angle2, angle3};
        cout << "angle is: " << angles[0] << " "  << angles[1] << " " << angles[2] << endl;
        return angles;
    }


    void SymmetryCollection::rotate(const quat<float> &q) {

        cout << endl << endl << "rotation starts" << endl;
        vector<float> eulerAngles = toEulerAngles323(q);

       for (int l = 0; l < (m_maxL + 1) * (m_maxL + 1); ++l) {
            m_Mlm_rotated.get()[l] = m_Mlm.get()[l];
        }


    cout << "before anything starts.. " << endl;
        int a = 0;
        for (int l = 0; l < (m_maxL + 1) * (m_maxL + 1); ++l) {
            cout << m_Mlm_rotated.get()[l] << " ";
            if( l == a*a+2*a) {
                ++a;
                cout << endl;
            }
        }
        cout <<endl;
        cout << "1.check computerMlm ends" << endl;






        cout << "WDTable.size(): " << WDTable.size() << endl;
        // generate Wigner-D table
        float c = cos(eulerAngles[1]);
        float s = sin(eulerAngles[1]);
        float sH = -s * 0.5;
        float cc = (1.0 + c) * 0.5;
        float ss = (1.0 - c) * 0.5;
        cout << "c is: " << eulerAngles[1] << " " << cos(eulerAngles[1]) << endl;
        cout << "s is: " << eulerAngles[1] << " " << sin(eulerAngles[1]) << endl;

        // initial values
        WDTable[WDindex(0, 0, 0)] = 1.0;                  // l = 0, m2 = 0, m1 = 0
        WDTable[WDindex(1, 0, 0)] = c;                    // l = 1, m2 = 0, m1 = 0
        WDTable[WDindex(1, 1, -1)] = ss;                   // l = 1, m2 = 1, m1 = -1
        WDTable[WDindex(1, 1, 0)] = -s * sqrt(0.5);       // l = 1, m2 = 1, m1 = 0
        WDTable[WDindex(1, 1, 1)] = cc;                   // l = 1, m2 = 1, m1 = 1

        cout <<  "WDTable[WDindex(0, 0, 0)]: " << WDindex(0, 0, 0) << " "  <<WDindex(1, 0, 0)<<WDindex(1, 1, -1) 
             << " " <<WDindex(1, 1, 0) << " " <<WDindex(1, 1, 1) << " " <<endl;

        // recursion for other values
        for (int j = 2; j <= m_maxL; ++j) {
 
            // mprime = 0:
            {
                WDTable[WDindex(j, 0, 0)] = (c * WDTable[WDindex(j - 1, 0, 0)] +
                                             s * WDTable[WDindex(j - 1, 1, 0)] * sqrt((float)(j - 1) / j));//no change?
            }

            // 0 < mprime < j:
            for (int mprime = 1; mprime < j - 1; ++mprime) {
                float Npp = sqrt((j + mprime) * (j + mprime - 1));
                float Npn = sqrt((j + mprime) * (j - mprime    ));
                float Nnn = sqrt((j - mprime) * (j - mprime - 1));
                // m = -mprime:
                {
                    WDTable[WDindex(j, mprime, -mprime)] = (ss * WDTable[WDindex(j - 1, mprime - 1, -mprime)] -
                                                            s  * WDTable[WDindex(j - 1, mprime    , -mprime + 1)] * Npn / Npp +
                                                            cc * WDTable[WDindex(j - 1, mprime + 1, -mprime + 2)] * Nnn / Npp);
                }
                // -mprime < m < mprime:
                for (int m = -mprime + 1; m < mprime; ++m) {
                    WDTable[WDindex(j, mprime, m)] = (sH * WDTable[WDindex(j - 1, mprime - 1, m - 1)] * Npp +
                                                      c  * WDTable[WDindex(j - 1, mprime,     m)] * Npn -
                                                      sH * WDTable[WDindex(j - 1, mprime + 1, m + 1)] * Nnn) / sqrt((j + m) * (j - m));
                }
                // m = mprime:
                {
                    WDTable[WDindex(j, mprime, mprime)] = (cc * WDTable[WDindex(j - 1, mprime - 1, mprime - 2)] +
                                                           s  * WDTable[WDindex(j - 1, mprime, mprime - 1)] * Npn / Npp +
                                                           ss * WDTable[WDindex(j - 1, mprime + 1, mprime)] * Nnn / Npp);
                }
            }

            // mprime = j - 1:
            {
                float Npp = sqrt(((j * 2) - 1) * ((j * 2) - 2));
                float Npn = sqrt((j * 2) - 1);
                // m = -mprime:
                {
                    WDTable[WDindex(j, j - 1, 1 - j)] = (ss * WDTable[WDindex(j - 1, j - 2, 1 - j)] -
                                                         s  * WDTable[WDindex(j - 1, j - 1, -j)] * Npn / Npp);//change?
                }
                // -mprime < m < mprime:
                for (int m = -j + 2; m < j - 1; ++m) {
                    WDTable[WDindex(j, j - 1, m)] = (sH * WDTable[WDindex(j - 1, j - 2, m - 1)] * Npp +
                                                     c  * WDTable[WDindex(j - 1, j - 1, m)] * Npn) / sqrt((j + m) * (j - m));
                }
                // m = mprime:
                {
                    WDTable[WDindex(j, j - 1, j - 1)] = (cc * WDTable[WDindex(j - 1, j - 2, j - 3)] +
                                                         s  * WDTable[WDindex(j - 1, j - 1, j - 2)] * Npn / Npp);
                }
            }

            // mprime = j:
            {
                float Npp = sqrt((j * 2) * ((j * 2) - 1));
                // m = -mprime:
                {
                    WDTable[WDindex(j, j, -j)] = ss * WDTable[WDindex(j - 1, j - 1, -j)];
                }
                // -mprime < m < mprime:
                for (int m = -j + 1; m < j; ++m) {
                    WDTable[WDindex(j, j, m)] = sH * WDTable[WDindex(j - 1, j - 1, m - 1)] * Npp / sqrt((j + m) * (j - m));
                }
                // m = mprime:
                {
                    WDTable[WDindex(j, j, j)] = cc * WDTable[WDindex(j - 1, j - 1, j - 2)];
                }
            }
        }

        // rotate spherical harmonics expansion
        for (int j = 1; j <= m_maxL; ++j) {

            // rotate around z-axis
            for (int k = 1; k <= j; ++k) {
                float arg = k * eulerAngles[2];
                float cosarg = cos(arg);
                float sinarg = sin(arg);
                float Mml1  = m_Mlm_rotated.get()[j * (j + 1) + k]; //one cell right from the central line
                float Mml2  = m_Mlm_rotated.get()[j * (j + 1) - k]; //one cell left from the central line
                m_Mlm_rotated.get()[j * (j + 1) + k] = Mml1 * cosarg - Mml2 * sinarg;
                m_Mlm_rotated.get()[j * (j + 1) - k] = Mml1 * sinarg + Mml2 * cosarg;
            }

            // rotate around x-axis
            vector<float> Mm2(2 * j + 1, 0);
            // mprime = 0:
            {
                float mm = WDTable[WDindex(j, 0, 0)] * m_Mlm_rotated.get()[j * (j + 1)];
                for (int m = 1; m <= j; ++m) {
                    float WD1 = WDTable[WDindex(j, m, 0)];//center?
                    if ((m & 1) != 0) WD1 = -WD1;
                    mm += WD1 * m_Mlm_rotated.get()[j * (j + 1) + m] * sqrt(2.0);
                }
                Mm2[j] = mm;
            }
            // mprime > 0:
            for (int mprime = 1; mprime <= j; ++mprime) {
                float mmp = WDTable[WDindex(j, mprime, 0)] * m_Mlm_rotated.get()[j * (j + 1)] * sqrt(2.0);
                float mmm = 0.0;
                // m < mprime:
                for (int m = 1; m < mprime; ++m) {
                    float WD1 = WDTable[WDindex(j, mprime, m)];
                    float WD2 = WDTable[WDindex(j, mprime, -m)];
                    if ((m & 1) != 0) WD2 = -WD2;
                    mmp += (WD1 + WD2) * m_Mlm_rotated.get()[j * (j + 1) + m] ;
                    mmm += (WD1 - WD2) * m_Mlm_rotated.get()[j * (j + 1) - m] ;
                }
                // m >= mprime:
                for (int m = mprime; m <= j; m++) {
                    float WD1 = WDTable[WDindex(j, m, mprime)];
                    float WD2 = WDTable[WDindex(j, m, -mprime)];
                    if ((mprime & 1) != 0) WD1 = -WD1;
                    if ((m & 1) != 0) WD2 = -WD2;
                    mmp += (WD1 + WD2) * m_Mlm_rotated.get()[j * (j + 1) + m];
                    mmm += (WD1 - WD2) * m_Mlm_rotated.get()[j * (j + 1) - m];
                }
                Mm2[j + mprime] = mmp;
                Mm2[j - mprime] = mmm;
            }

            // rotate around z-axis
            for (int k = 1; k <= j; ++k) {
                float arg = k * eulerAngles[0];
                float cosarg = cos(arg);
                float sinarg = sin(arg);
                float Mml1  = Mm2[j + k];
                float Mml2  = Mm2[j - k];
                Mm2[j + k] = Mml1 * cosarg - Mml2 * sinarg;
                Mm2[j - k] = Mml1 * sinarg + Mml2 * cosarg;
            }
            cout << "Mm2[i] is: " << endl;
            for (int i = 0; i < 2 * j + j; ++i) {
                
                cout << Mm2[i] << " "; 
                m_Mlm_rotated.get()[j * j + i] = Mm2[i];
            }
            cout << endl << "ends" << endl;

        }
    }




    // rotate the axis p in z-direction
    quat<float> SymmetryCollection::initMirrorZ(const vec3<float> &p) {
        float x = p.x;
        float y = p.y;
        float z = p.z + 1.0;
        float n = sqrt(x * x + y * y + z * z);
        cout << "initMirrorZ start" << endl << "x y z is: " << x << " "  << y << " " << z << endl;


        quat<float> temp(0.0f, {0.0f, 0.0f, 0.0f});
        cout << "after-assginment is: " << temp.s << " "  << temp.v.x << " " << temp.v.y << " " << temp.v.z << endl;
        if (n == 0.0f){
            return temp;
        }

        temp.v.x = x / n;
        temp.v.y = y / n;
        temp.v.z = z / n;
        return temp;

    }

    // Symmetry SymmetryCollection::findSymmetry(int type) {
    //     for (Symmetry &symmetry: SYMMETRIES) {
    //         if(symmetry.type == type) {
    //             return symmetry;
    //         }
    //     }
    //     return nullptr;
    // }


    // int SymmetryCollection::searchSymmetry(bool perpendicular) {

    //     // brute-force search for the best symmetry, initialize triangulation for search directions
    //     int highestSymmetry = TOTAL;
    //     shared_ptr<vector<vec3<float> > > vertexList;
    //     shared_ptr<vector<unordered_set<int> > > neighborList;
    //     if (perpendicular) {
    //         // using a lattice on the circle (128 points)
    //         const int CIRCLENUMBER = 128;
    //         vertexList = shared_ptr<vector<vec3<float> > >(new vector<vec3<float> >());
    //         neighborList = shared_ptr<vector<unordered_set<int> > >(new vector<unordered_set<int> >[m_vertexList->size()],
    //                                                                 default_delete<vector<unordered_set<int> >[]>());
            
    //         for (int i = 0; i < CIRCLENUMBER; ++i) {
    //             // only need to search one side of the circle
    //             float angle = PI * i / CIRCLENUMBER;
    //             vec3<float> temp(cos(angle), sin(angle), 0.0);
    //             vertexList->pushback(temp);
    //             unordered_set<int> neighbors;
    //             neighbors.insert(i + 1 < CIRCLENUMBER ? i + 1 : 0);
    //             neighbors.insert(i - 1 >= 0 ? i - 1 : CIRCLENUMBER - 1);
    //             neighborList->pushback(neighbors);
    //         }
    //     }
    //     else {
    //         // using a grid on the sphere (5 iterations)
    //         Geodesation geodesation(5);
    //         vertexList = geodesation.getVertexList();
    //         neighborList = geodesation.getNeighborList();
    //     }
        
    //     // measure the order for each vertex and each symmetry (except TOTAL and MIRROR)
    //     double[][] orderTable = new double[vertexList.size()][SYMMETRIES.length]; // ASK, one vertex can have several symmetries?
    //     for (int vi = 0; vi < vertexList.size(); vi++) {
    //         quat<float> quat;
    //         if (perpendicular) {
    //             // rotation around z-axis and then the x-axis in z-direction
    //             quat<float> roty(sqrt(0.5), 0,0f, sqrt(0.5), 0.0f);
    //             quat<float> rotz = (vertexList->at(vi)[0], 0.0f, 0.0f, vertexList->at(vi)[1]);
    //             quat = roty * (rotz * rot);
    //         }
    //         else {
    //             // rotate vertex in z-direction
    //             quat = initMirrorZ(vertexList->at(vi));
    //         }

    //         double[][] Mlm2 = sphericalHarmonics.rotate(ArrayTools.clone(Mlm), quat);
    //         for (int sym = 0; sym < SYMMETRIES.length; sym++) {// inner loop, for each vertex, measure the order for each symmetry
    //             Symmetry symmetry = SYMMETRIES[sym];
    //             if (symmetry.type == TOTAL || symmetry.type == MIRROR) continue;
    //             orderTable[vi][sym] = symmetry.measure(Mlm2);
    //         }
    //     }
        
    //     // test each symmetry separately, start with highest symmetry and then search downwards
    //     vector<bool> foundBefore(vertexList.size());
    //     for (int sym = SYMMETRIES.length - 1; sym >= 0; sym--) {
    //         Symmetry symmetry = SYMMETRIES[sym];
    //         if (symmetry.type == TOTAL || symmetry.type == MIRROR) continue;
    //         vector<bool> alreadyChecked(vertexList.size());
            
    //         // now loop recursively through the geodesation and determine ordered directions
    //         float orderFoundGlobal = -1.0;
    //         int directionFoundGlobal = -1;
    //         for (int vi = 0; vi < vertexList.size(); vi++) {
    //             // search iteratively through all neighboring directions // ASK I did not see any recursion here
    //             vector<int> searchList;
    //             searchList.pushback(vi);
                
    //             bool overlapsWithOther = false;
    //             float orderFound = -1.0;
    //             int directionFound = -1;
    //             for (int j = 0; j < searchList.size(); j++)
    //             {
    //                 int vj = searchList[j];
                    
    //                 // did we already check this axis and have we found it before?
    //                 if (alreadyChecked[vj]) continue;
    //                 alreadyChecked[vj] = true;
    //                 if (orderTable[vj][sym] < symmetry.threshold) continue;
    //                 if (foundBefore[vj]) overlapsWithOther = true;
    //                 foundBefore[vj] = true;
                    
    //                 // keep track of 'best' symmetry found so far for this axis
    //                 if (orderTable[vj][sym] > orderFound)
    //                 {
    //                     orderFound = orderTable[vj][sym];
    //                     directionFound = vj;
    //                 }
                    
    //                 // check in the neighborhood for symmetries
    //                 for (int vk: neighborList->at(vj)) 
    //                     searchList.pushback(vk);
    //             }
                
    //             // if we have a valid symmetry axis
    //             if (directionFound >= 0 && !overlapsWithOther) {
    //                 // add a marker on the BOD plot
    //                 if (symmetry.type >= 2 && !perpendicular) {
    //                     plot.addMarker(symmetry.type, vertexList->at(directionFound));
    //                 }
                    
    //                 if (orderFound > orderFoundGlobal) {
    //                     // keep track of 'best' symmetry found so far for a symmetry type
    //                     orderFoundGlobal = orderFound;
    //                     directionFoundGlobal = directionFound;
    //                 }
    //             }
    //         }

    //         // handle best overall axis for a given symmetry type
    //         if (directionFoundGlobal >= 0 && symmetry.type >= highestSymmetry) {
    //             highestSymmetry = symmetry.type;
                
    //             vec3<float> vertex = vertexList->at(directionFoundGlobal);
    //             if (perpendicular) {
    //                 // rotation around z-axis
    //                 quat<float> rotz(vertex[0], 0.0, 0.0, vertex[1]);
    //                 rot = rotz * rot; //ASK confirm
    //                 // don't have to keep searching
    //                 break;
    //             }
                
    //             rot = initMirrorZ(vertex);
    //         }
    //     }
    //     return highestSymmetry;



    // }






}; }; // end namespace freud::symmetry
