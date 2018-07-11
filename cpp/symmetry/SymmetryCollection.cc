// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <complex>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <unordered_set>
#include <time.h>

#include "SymmetryCollection.h"
#include "VectorMath.h"
#include "Geodesation.h"

using namespace std;

/*! \file SymmetricOrientation.cc
    \brief Compute the symmetric orientation.
*/

namespace freud { namespace symmetry {

    SymmetryCollection::SymmetryCollection(unsigned int maxL) :
                                           m_maxL(maxL), m_symmetric_orientation(quat<float>()) {
        m_Mlm = shared_ptr<float>(new float[(m_maxL + 1) * (m_maxL + 1)],
                                  default_delete<float[]>());

        memset((void*)m_Mlm.get(), 0, sizeof(float)*((m_maxL + 1) * (m_maxL + 1)));

        m_Mlm_rotated = shared_ptr<float>(new float[(m_maxL + 1) * (m_maxL + 1)],
                                          default_delete<float[]>());

        memset((void*)m_Mlm_rotated.get(), 0, sizeof(float)*((m_maxL + 1) * (m_maxL + 1)));

        int WDlength = (1 + maxL) * (2 + maxL) * (3 + 2 * maxL) / 6;
        WDTable.resize(WDlength);

        for (int i = 0; i < LENGTH; ++i) {

            SYMMETRIES[i].type = i - 1;
            SYMMETRIES[i].threshold = 0.75;
            if (i == 0) {
                SYMMETRIES[i].threshold = 0.5;
            } else if (i == 1) {
                SYMMETRIES[i].threshold = 15.0;
            } else if (i == 8) {
                SYMMETRIES[i].type = 8;
            } else if (i == 9) {
                SYMMETRIES[i].type = 10;
            } else if (i == 10) {
                SYMMETRIES[i].type = 12;
            }
        }

    }

    SymmetryCollection::~SymmetryCollection() {}

    void SymmetryCollection::compute(const box::Box& box,
                                     const vec3<float> *points,
                                     const freud::locality::NeighborList *nlist,
                                     unsigned int Np) {
        assert(points);
        assert(Np > 0);
        nlist->validate(Np, Np);

        // Compute the Mlm array of spherical harmonics from the bonds
        computeMlm(box, points, nlist, Np);

        // Find all approx symmetries
        searchSymmetry();

        for (auto &f: m_found_symmetries) {

            optimize(false, &f);

            searchSymmetry(&f);

            optimize(true, &f);

        }
    }


    float SymmetryCollection::measure(int type) {
        shared_ptr<float> Mlm = getMlm_rotated();

        if (Mlm.get()[0] == 0.0f) {
            return 0.0f;
        }
        float select = 0.0;
        float all = 0.0;
        int numSelect = 0;
        int numAll = 0;

        for (int l = 2; l < (m_maxL + 1); l += 2) {
           int count = -l - 1;
            for (int m = l * l; m < (l + 1) * (l + 1); ++m) {
                ++numAll;
                all += Mlm.get()[m] * Mlm.get()[m];
                ++count;
                if ((type == TOTAL) ||
                    (type == AXIAL && (m == l * (l + 1))) ||
                    (type == MIRROR && (m >= l * (l + 1))) ||
                    (type >= 2 && (count % type == 0))) {
                    ++numSelect;
                    select += Mlm.get()[m] * Mlm.get()[m];
                }
            }
        }

        if (type == TOTAL || type == AXIAL) {
            return (select / Mlm.get()[0] / (float)numSelect - 1.0f);
        }

        if (type == MIRROR || type >= 2) {
            return (select / all - (float)numSelect / (float)numAll) / (1.0f - (float)numSelect / (float)numAll);
        }
        return 0.0f;

    }

    void SymmetryCollection::rotate(const quat<float> &q) {

        vector<float> eulerAngles = toEulerAngles323(q);

        for (int l = 0; l < (m_maxL + 1) * (m_maxL + 1); ++l) {
            m_Mlm_rotated.get()[l] = m_Mlm.get()[l];
        }

        // generate Wigner-D table
        float c = cos(eulerAngles[1]);
        float s = sin(eulerAngles[1]);
        float sH = -s * 0.5;
        float cc = (1.0 + c) * 0.5;
        float ss = (1.0 - c) * 0.5;

        // initial values
        WDTable[WDindex(0, 0, 0)] = 1.0;                  // j = 0, mprime = 0, m = 0
        WDTable[WDindex(1, 0, 0)] = c;                    // j = 1, mprime = 0, m = 0
        WDTable[WDindex(1, 1, -1)] = ss;                  // j = 1, mprime = 1, m = -1
        WDTable[WDindex(1, 1, 0)] = -s * sqrt(0.5);       // j = 1, mprime = 1, m = 0
        WDTable[WDindex(1, 1, 1)] = cc;                   // j = 1, mprime = 1, m = 1

        // recursion for other values
        for (int j = 2; j <= m_maxL; ++j) {
            // mprime = 0:
            WDTable[WDindex(j, 0, 0)] = (c * WDTable[WDindex(j - 1, 0, 0)] +
                                         s * WDTable[WDindex(j - 1, 1, 0)] * sqrt((float)(j - 1) / j));
            // 0 < mprime < j:
            for (int mprime = 1; mprime < j - 1; ++mprime) {
                float Npp = sqrt((j + mprime) * (j + mprime - 1));
                float Npn = sqrt((j + mprime) * (j - mprime    ));
                float Nnn = sqrt((j - mprime) * (j - mprime - 1));

                // m = -mprime:
                WDTable[WDindex(j, mprime, -mprime)] = (ss * WDTable[WDindex(j - 1, mprime - 1, 1 - mprime)] -
                                                        s  * WDTable[WDindex(j - 1, mprime    , 1 - mprime)] * Npn / Npp +
                                                        cc * WDTable[WDindex(j - 1, mprime + 1, 1 - mprime)] * Nnn / Npp);

                // -mprime < m < mprime:
                for (int m = -mprime + 1; m < mprime; ++m) {
                    WDTable[WDindex(j, mprime, m)] = (sH * WDTable[WDindex(j - 1, mprime - 1, m)] * Npp +
                                                      c  * WDTable[WDindex(j - 1, mprime,     m)] * Npn -
                                                      sH * WDTable[WDindex(j - 1, mprime + 1, m)] * Nnn) / sqrt((j + m) * (j - m));
                }

                // m = mprime:
                WDTable[WDindex(j, mprime, mprime)] = (cc * WDTable[WDindex(j - 1, mprime - 1, mprime - 1)] +
                                                       s  * WDTable[WDindex(j - 1, mprime,     mprime - 1)] * Npn / Npp +
                                                       ss * WDTable[WDindex(j - 1, mprime + 1, mprime - 1)] * Nnn / Npp);

            }

            // mprime = j - 1:
            float Npp = sqrt(((j * 2) - 1) * ((j * 2) - 2));
            float Npn = sqrt((j * 2) - 1);

            // m = -mprime:
            WDTable[WDindex(j, j - 1, 1 - j)] = (ss * WDTable[WDindex(j - 1, j - 2, 2 - j)] -
                                                 s  * WDTable[WDindex(j - 1, j - 1, 2 - j)] * Npn / Npp);

            // -mprime < m < mprime:
            for (int m = -j + 2; m < j - 1; ++m) {
                WDTable[WDindex(j, j - 1, m)] = (sH * WDTable[WDindex(j - 1, j - 2, m)] * Npp +
                                                 c  * WDTable[WDindex(j - 1, j - 1, m)] * Npn) / sqrt((j + m) * (j - m));
            }

            // m = mprime:
            WDTable[WDindex(j, j - 1, j - 1)] = (cc * WDTable[WDindex(j - 1, j - 2, j - 2)] +
                                                 s  * WDTable[WDindex(j - 1, j - 1, j - 2)] * Npn / Npp);

            // mprime = j:
            Npp = sqrt((j * 2) * ((j * 2) - 1));

            // m = -mprime:
            WDTable[WDindex(j, j, -j)] = ss * WDTable[WDindex(j - 1, j - 1, 1 - j)];

            // -mprime < m < mprime:
            for (int m = -j + 1; m < j; ++m) {
                WDTable[WDindex(j, j, m)] = sH * WDTable[WDindex(j - 1, j - 1, m)] * Npp / sqrt((j + m) * (j - m));
            }

            // m = mprime:
            WDTable[WDindex(j, j, j)] = cc * WDTable[WDindex(j - 1, j - 1, j - 1)];

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
            float mm = WDTable[WDindex(j, 0, 0)] * m_Mlm_rotated.get()[j * (j + 1)];
            for (int m = 1; m <= j; ++m) {
                float WD1 = WDTable[WDindex(j, m, 0)];//central line
                if ((m & 1) != 0) WD1 = -WD1;
                mm += WD1 * m_Mlm_rotated.get()[j * (j + 1) + m] * sqrt(2.0);
            }
            Mm2[j] = mm;

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
                for (int m = mprime; m <= j; ++m) {
                    float WD1 = WDTable[WDindex(j, m, mprime)];
                    float WD2 = WDTable[WDindex(j, m, -mprime)];
                    if (((m - mprime) & 1) != 0) WD1 = -WD1;
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


            for (int i = 0; i < 2 * j + 1; ++i) {
                m_Mlm_rotated.get()[j * j + i] = Mm2[i];
            }

        }
    }

    void SymmetryCollection::searchSymmetry(FoundSymmetry *foundsym) {

        // brute-force search for the best symmetry, initialize triangulation for search directions
        int highestSymmetry = TOTAL;
        shared_ptr<vector<vec3<float> > > vertexList;
        shared_ptr<vector<unordered_set<int> > > neighborList;
        if (foundsym != nullptr) {
            // using a lattice on the circle (128 points)
            const int CIRCLENUMBER = 128;
            vertexList = shared_ptr<vector<vec3<float> > >(new vector<vec3<float> >());
            neighborList = shared_ptr<vector<unordered_set<int> > >(
                new vector<unordered_set<int> >[CIRCLENUMBER],
                default_delete<vector<unordered_set<int> >[]>());

            for (int i = 0; i < CIRCLENUMBER; ++i) {
                // only need to search one side of the circle
                float angle = PI * i / CIRCLENUMBER;
                vec3<float> temp(cos(angle/2), sin(angle/2), 0.0f);
                vertexList->push_back(temp);
                unordered_set<int> neighbors;
                neighbors.insert(i + 1 < CIRCLENUMBER ? i + 1 : 0);
                neighbors.insert(i - 1 >= 0 ? i - 1 : CIRCLENUMBER - 1);
                neighborList->push_back(neighbors);
            }
        } else {
            // using a grid on the sphere (5 iterations)
            Geodesation geodesation(5);
            vertexList = geodesation.getVertexList();
            neighborList = geodesation.getNeighborList();
        }

        //measure the order for each vertex and each symmetry (except TOTAL and MIRROR)
        vector<vector<float> > orderTable(vertexList->size(), vector<float>(11, 0));

        for (int vi = 0; vi < vertexList->size(); ++vi) {
            quat<float> quaternion;
            if (foundsym != nullptr) {
                // rotation around z-axis and then the x-axis in z-direction
                quat<float> roty(sqrt(0.5), {0.0f, sqrt(0.5), 0.0f});
                quat<float> rotz(vertexList->at(vi).x, {0.0f, 0.0f, vertexList->at(vi).y});
                quaternion = roty * rotz * foundsym->q;
                quaternion = normalize(quaternion);
            } else {
                // rotate vertex in z-direction
                quaternion = initMirrorZ(vertexList->at(vi));
            }

            rotate(quaternion);

            for (int sym = 0; sym < 11; ++sym) {
                // for each vertex, measure the order for each symmetry
                Symmetry symmetry = SYMMETRIES[sym];
                if (symmetry.type == TOTAL || symmetry.type == MIRROR) continue;
                orderTable[vi][sym] = measure(symmetry.type);
            }
        }

        // test each symmetry separately, start with highest symmetry and then search downwards
        vector<bool> foundBefore(vertexList->size());
        for (int sym = 11 - 1; sym >= 0; sym--) {
            Symmetry symmetry = SYMMETRIES[sym];
            if (symmetry.type == TOTAL || symmetry.type == MIRROR) continue;
            vector<bool> alreadyChecked(vertexList->size());

            // now loop recursively through the geodesation and determine ordered directions
            float orderFoundGlobal = -1.0;
            int directionFoundGlobal = -1;
            for (int vi = 0; vi < vertexList->size(); vi++) {
                // search iteratively through all neighboring directions
                vector<int> searchList;
                searchList.push_back(vi);

                bool overlapsWithOther = false;
                float orderFound = -1.0;
                int directionFound = -1;
                for (int j = 0; j < searchList.size(); ++j) {
                    int vj = searchList[j];

                    // did we already check this axis and have we found it before?
                    if (alreadyChecked[vj]) continue;
                    alreadyChecked[vj] = true;
                    if (orderTable[vj][sym] < symmetry.threshold) continue;
                    if (foundBefore[vj]) overlapsWithOther = true;
                    foundBefore[vj] = true;

                    // keep track of 'best' symmetry found so far for this axis
                    if (orderTable[vj][sym] > orderFound) {
                        orderFound = orderTable[vj][sym];
                        directionFound = vj;
                    }

                    // check in the neighborhood for symmetries
                    for (int vk: neighborList->at(vj))
                        searchList.push_back(vk);
                }

                // if we have a valid symmetry axis
                if (directionFound >= 0 && !overlapsWithOther) {

                    // add the newly found symmetry axis
                    if (symmetry.type >= 2 && foundsym == nullptr) {
                        FoundSymmetry fs;
                        fs.n = symmetry.type;
                        fs.v = vertexList->at(directionFound);
                        fs.q = initMirrorZ(vertexList->at(directionFound));
                        fs.measured_order = orderTable[directionFound][sym];
                        m_found_symmetries.push_back(fs);
                    }

                    if (orderFound > orderFoundGlobal) {
                        // keep track of 'best' symmetry found so far for a symmetry type
                        orderFoundGlobal = orderFound;
                        directionFoundGlobal = directionFound;
                    }
                }
            }

            // handle best overall axis for a given symmetry type
            if (directionFoundGlobal >= 0 && symmetry.type >= highestSymmetry) {
                highestSymmetry = symmetry.type;
                vec3<float> vertex = vertexList->at(directionFoundGlobal);
                if (foundsym != nullptr) {
                    // rotation around z-axis
                    quat<float> rotz = quat<float>(vertex.x, {0.0, 0.0, vertex.y});
                    foundsym->q = (conj(foundsym->q) * rotz * foundsym->q) * foundsym->q;
                    foundsym->v = ::rotate(foundsym->q, vec3<float>(0, 0, 1));
                    foundsym->measured_order = orderFoundGlobal;

                    // don't have to keep searching
                    break;
                }
            }
        }
    }

    void SymmetryCollection::optimize(bool perpendicular, FoundSymmetry *symm) {
        int type = symm->n;
        float step = OPTIMIZESTART;
        quat<float> quaternion = symm->q;
        quat<float> measureRotate;

        if (perpendicular) {
            measureRotate.s = sqrt(0.5);
            measureRotate.v.x = 0.0f;
            measureRotate.v.y = sqrt(0.5);
            measureRotate.v.z = 0.0f;
        }

        // measure order
        quat<float> q = measureRotate * quaternion;

        float order = symm->measured_order;

        do {
            vector<quat<float> > quats;
            if (perpendicular) {
                // generate trial rotations around the z-axis
                float c = cos(step);
                float s = sin(step);
                quats.push_back({c, {0.0f, 0.0f, +s}});
                quats.push_back({c, {0.0f, 0.0f, -s}});
            } else {
                // generate trial rotations perpendicular to the z-axis
                srand((unsigned)time(NULL));
                float angle = 2.0 * PI * (float)rand()/RAND_MAX;
                float c = cos(angle) * step;
                float s = sin(angle) * step;
                float n = sqrt(1.0 - step * step);
                quats.push_back({n, {+c, +s, 0.0f}});
                quats.push_back({n, {-c, +s, 0.0f}});
                quats.push_back({n, {+c, -s, 0.0f}});
                quats.push_back({n, {-c, -s, 0.0f}});
            }

            // search whether the test points have higher order
            bool found = false;
            for (quat<float> &testQuat: quats) {
                testQuat = testQuat * quaternion;
                // measure order
                q =  measureRotate * testQuat;
                rotate(q);
                float testOrder = measure(type);
                if (testOrder > order) {
                    quaternion = normalize(testQuat);
                    order = testOrder;
                    found = true;
                }
            }
            if (!found) step *= OPTIMIZESCALE;
        } while (step >= OPTIMIZEEND);

        symm->q = quaternion;
        symm->v = ::rotate(symm->q, vec3<float>(0, 0, 1));
        symm->measured_order = order;
    }


    string SymmetryCollection::getLaueGroup() {

        int n2 = 0;
        int n3 = 0;
        int n4 = 0;
        int n6 = 0;
        // // identify Laue group
        for (FoundSymmetry &fs: m_found_symmetries) {
            if (fs.n == 2) {
                ++n2;
            } else if(fs.n == 3) {
                ++n3;
            } else if(fs.n == 4) {
                ++n4;
            } else if(fs.n == 6) {
                ++n6;
            } else {}
        }

        if (n2 == 0 && n3 == 0 && n4 == 0 && n6 == 0) return "-1";
        if (n2 == 1 && n3 == 0 && n4 == 0 && n6 == 0) return "2/m";
        if (n2 == 3 && n3 == 0 && n4 == 0 && n6 == 0) return "mmm";
        if (n2 == 0 && n3 == 1 && n4 == 0 && n6 == 0) return "-3";
        if (n2 == 3 && n3 == 1 && n4 == 0 && n6 == 0) return "-3m";
        if (n2 == 0 && n3 == 0 && n4 == 1 && n6 == 0) return "4/m";
        if (n2 == 4 && n3 == 0 && n4 == 1 && n6 == 0) return "4/mmm";
        if (n2 == 0 && n3 == 0 && n4 == 0 && n6 == 1) return "6/m";
        if (n2 == 6 && n3 == 0 && n4 == 0 && n6 == 1) return "6/mmm";
        if (n2 == 3 && n3 == 4 && n4 == 0 && n6 == 0) return "m-3";
        if (n2 == 6 && n3 == 4 && n4 == 3 && n6 == 0) return "m-3m";
        return "Unable to identify the Laue Group";
    }

    string SymmetryCollection::getCrystalSystem() {
        if (getLaueGroup() == "-1") return "Triclinic";
        if (getLaueGroup() == "2/m") return "Monoclinic";
        if (getLaueGroup() == "mmm") return "Orthorhombic";
        if (getLaueGroup() == "-3") return "Trigonal";
        if (getLaueGroup() == "-3m") return "Trigonal";
        if (getLaueGroup() == "4/m") return "Tetragonal";
        if (getLaueGroup() == "4/mmm") return "Tetragonal";
        if (getLaueGroup() == "6/m") return "Hexagonal";
        if (getLaueGroup() == "6/mmm") return "Hexagonal";
        if (getLaueGroup() == "m-3") return "Cubic";
        if (getLaueGroup() == "m-3m") return "Cubic";
        return "Unable to identify the Crystal System";
    }


    void SymmetryCollection::computeMlm(const box::Box& box,
                                        const vec3<float> *points,
                                        const freud::locality::NeighborList *nlist,
                                        unsigned int Np) {
        // Reset Mlm to zeros
        memset((void*)m_Mlm.get(), 0, sizeof(float)*((m_maxL + 1) * (m_maxL + 1)));
        memset((void*)m_Mlm_rotated.get(), 0, sizeof(float)*((m_maxL + 1) * (m_maxL + 1)));

        m_box = box;
        m_Np = Np;
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

                if (i < j) {
                    delta[valid_bonds] = m_box.wrap(points[j] - points[i]);
                    phi[valid_bonds] = atan2(delta[valid_bonds].y, delta[valid_bonds].x);
                    theta[valid_bonds] = acos(delta[valid_bonds].z / sqrt(
                    dot(delta[valid_bonds], delta[valid_bonds])));
                    valid_bonds++;

                }
            }

        }

        //We un-normalize the spherical harmonics
        double sphNorm = sqrt(4 * PI);
        fsph::PointSPHEvaluator<float> eval(m_maxL);
        for(unsigned int i = 0; i < valid_bonds; ++i) {
            unsigned int l0_index = 0;
            unsigned int l = 0;
            unsigned int m = 0;
            double l_parity = 1;
            eval.compute(theta[i], phi[i]);
            for(typename fsph::PointSPHEvaluator<float>::iterator iter(eval.begin(false));
                iter != eval.end(); ++iter) {
                if (m > l) {
                    l++;
                    l_parity *= -1;
                    l0_index += 2 * l;
                    m = 0;
                }
                if (m == 0) {
                    m_Mlm.get()[l0_index] += l_parity * sphNorm * (*iter).real();
                } else {
                    m_Mlm.get()[l0_index + m] += sqrt(2) * l_parity * sphNorm * (*iter).real();
                    m_Mlm.get()[l0_index - m] += sqrt(2) * l_parity * sphNorm * (*iter).imag();
                }
                m++;
            }
        }

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

        float angle1 = atan2(2.0 * (q23 - q01), 2.0 * (q13 + q02)); //phi
        float angle2 = acos (q00 - q11 - q22 + q33);                //theta
        float angle3 = atan2(2.0 * (q01 + q23), 2.0 * (q02 - q13)); //psi
        if (q01 == 0.0 && q23 == 0.0 && q13 == 0.0 && q02 == 0.0) {
            // degenerate case, rotation around z-axis only
            angle1 = atan2(2.0 * (q03 - q12), q00 + q11 - q22 - q33);
            angle2 = 0.0;
            angle3 = 0.0;
        }

        if (isnan(angle1)) angle1 = 0.5 * PI;
        if (isnan(angle2)) angle2 = 0.5 * PI;
        if (isnan(angle3)) angle3 = 0.5 * PI;

        vector<float> angles{angle1, angle2, angle3};
        return angles;
    }

    // rotate the axis p in z-direction
    quat<float> SymmetryCollection::initMirrorZ(const vec3<float> &p) {
        float x = p.x;
        float y = p.y;
        float z = p.z + 1.0;
        float n = sqrt(x * x + y * y + z * z);


        quat<float> temp(0.0f, {0.0f, 0.0f, 0.0f});

        if (n == 0.0f){
            return temp;
        }

        temp.v.x = x / n;
        temp.v.y = y / n;
        temp.v.z = z / n;
        return temp;

    }

}; }; // end namespace freud::symmetry
