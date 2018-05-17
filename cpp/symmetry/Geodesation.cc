// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <complex>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <unordered_set>
#include <iostream>

#include "Geodesation.h"
#include "VectorMath.h"

using namespace std;

/*! \file Geodesation.cc
    \brief Compute the geodesation.
*/

namespace freud { namespace symmetry {


    Geodesation::Geodesation(unsigned int iteration) :
        vertexList(vector<vec3<float> >()) {

        const double TAU = 0.5 * (sqrt(5.0) + 1.0);
        const double A = 1.0 / sqrt(2.0 + TAU);
        const double B = TAU * A;

        int v0 = createVertex ( 0,  A,  B);
        int v1 = createVertex ( 0, -A,  B);
        int v2 = createVertex ( B,  0,  A);
        int v3 = createVertex (-B,  0,  A);
        int v4 = createVertex ( A,  B,  0);
        int v5 = createVertex ( A, -B,  0);
        int s0 = createSimplex(v0, v1, v2);
        int s1 = createSimplex(v0, v2, v4);
        int s2 = createSimplex(v0, v4, v5);
        int s3 = createSimplex(v0, v5, v3);
        int s4 = createSimplex(v0, v3, v1);
        int s5 = createSimplex(v1, v3, v4);
        int s6 = createSimplex(v1, v4, v5);
        int s7 = createSimplex(v1, v5, v2);
        int s8 = createSimplex(v2, v3, v4);
        int s9 = createSimplex(v2, v5, v3);

        connectSimplices(s0, s1);
        connectSimplices(s0, s4);
        connectSimplices(s0, s7);
        connectSimplices(s1, s2);
        connectSimplices(s1, s8);
        connectSimplices(s2, s3);
        connectSimplices(s2, s6);
        connectSimplices(s3, s4);
        connectSimplices(s3, s9);
        connectSimplices(s4, s5);
        connectSimplices(s5, s6);
        connectSimplices(s5, s8);
        connectSimplices(s6, s7);
        connectSimplices(s7, s9);
        connectSimplices(s8, s9);

        /* test starts*/
        // int v0 = createVertex ( 1,  2,  3);
        // int v1 = createVertex ( 6,  5,  4);
        // int v2 = createVertex ( 7,  0,  7);
        // int v3 = createVertex (-2,  -1,  1);
        // int v4 = createVertex ( 3,  8,   0);
        // int v5 = createVertex ( 6,  6,   6);
        // int s0 = createSimplex(v0, v1, v2);
        // int s1 = createSimplex(v3, v4, v5);
        /* test ends*/

        // for (int i = 0; i < iteration; ++i) {
        //     geodesate();
        // }
    }

    Geodesation::~Geodesation(){
        
    }

    void Geodesation::geodesate() {
        vector<vector<int> > oldSimplexList = simplexList;
            simplexList.clear();
            /*test starts*/
        cout << "geodesate" << endl;
        cout << "oldSimplexList.size(): " << oldSimplexList.size() << endl;   
        for (int i = 0; i < oldSimplexList.size(); ++i) {
            for (int j = 0 ; j < 6; ++j) {
                cout << "oldSimplexList[" << i <<"]["<<j<<"]: " << oldSimplexList[i][j] << " ";
            }
            cout << endl;
        }
            cout << endl;
        /*test ends*/

            for (int n = 0; n < oldSimplexList.size(); n++) {
                vector<int> simplex = oldSimplexList[n];
                
                // vertex indices
                int i0 = simplex[0];
                int i1 = simplex[1];
                int i2 = simplex[2];
                // neighbor indices
                int n0 = simplex[3];
                int n1 = simplex[4];
                int n2 = simplex[5];
                
                // int i12 = -1;
                // int i20 = -1;
                // int i01 = -1;
                // // new vertices are at edge-midpoints, either take existing ones or create new ones
                // if (n0 < n) {
                //     cout << "aaa" << endl;
                //     cout << "n0 = " << n0 << endl;
                //     cout << "oldSimplexList[n0].size() = " << oldSimplexList[n0].size() << endl;
                //     i12 = findNeighborMidVertex(oldSimplexList[n0], n);
                // } else {
                //     cout << "bbb" << endl;
                //     i12 = createMidVertex(i1, i2);
                // }

                // // if (n1 < n) {
                // //     i20 = findNeighborMidVertex(oldSimplexList[n1], n);
                // // } else {
                // //     i20 = createMidVertex(i2, i0);
                // // }


                // // if (n2 < n) {
                // //     i01 = findNeighborMidVertex(oldSimplexList[n2], n);
                // // } else {
                // //     i01 = createMidVertex(i0, i1);
                // // }




                int i12 = (n0 < n ? findNeighborMidVertex(oldSimplexList[n0], n) : createMidVertex(i1, i2));
                int i20 = (n1 < n ? findNeighborMidVertex(oldSimplexList[n1], n) : createMidVertex(i2, i0));
                int i01 = (n2 < n ? findNeighborMidVertex(oldSimplexList[n2], n) : createMidVertex(i0, i1));
                //store new midpoint vertices so we can reuse them
                oldSimplexList[n][0] = i12;
                oldSimplexList[n][1] = i20;
                oldSimplexList[n][2] = i01;

                // combine the vertices
                int s0i = createSimplex(i20, i0, i01);
                int s1i = createSimplex(i01, i1, i12);
                int s2i = createSimplex(i12, i2, i20);
                int si  = createSimplex(i01, i12, i20);

                // interior connections
                connectSimplices(s0i, si);
                connectSimplices(s1i, si);
                connectSimplices(s2i, si);
                // exterior connections
                if (n0 < n) {
                    for (int j = 0; j < 3; j++) {
                    connectSimplices(s1i, n0 * 4 + j);
                    connectSimplices(s2i, n0 * 4 + j);
                    }
                }
                if (n1 < n) {
                    for (int j = 0; j < 3; j++) {
                    connectSimplices(s0i, n1 * 4 + j);
                    connectSimplices(s2i, n1 * 4 + j);
                    }
                }
                if (n2 < n) {
                    for (int j = 0; j < 3; j++) {
                    connectSimplices(s0i, n2 * 4 + j);
                    connectSimplices(s1i, n2 * 4 + j);
                    }
                }
            }
    }

    int Geodesation::createVertex(float x, float y, float z) {
        vertexList.push_back(vec3<float>(x, y, z));

        cout << "createVertex" << endl;
        //cout << "vertexList[0].x is: " << vertexList[0].x << endl;
        cout << "vertexList.size(): " << vertexList.size() << endl;
        return vertexList.size() - 1;
    }

    int Geodesation::createSimplex(int v0, int v1, int v2) {

        // sort vertices
        if (v0 > v1) {
            int t = v0; v0 = v1; v1 = t;
        }
        if (v1 > v2) {
            int t = v1; v1 = v2; v2 = t;
        }
        if (v0 > v1) {
            int t = v0; v0 = v1; v1 = t;
        }
        vector<int> temp = {v0, v1, v2, -1, -1, -1};
        // do not specify neighbors yet
        simplexList.push_back(temp);

        /*test starts*/
        cout << "createSimplex" << endl;
        cout << "simplexList.size()-1: " << simplexList.size() - 1 << endl;   
        for (int i = 0; i < simplexList.size(); ++i) {
            for (int j = 0 ; j < 6; ++j) {
                cout << "simplexList[" << i <<"]["<<j<<"]: " << simplexList[i][j] << " ";
            }
            cout << endl;
        }
            cout << endl;
        /*test ends*/

        return simplexList.size() - 1;
    }

    void Geodesation::connectSimplices(int s0, int s1) {

        vector<int> simplex0 = simplexList[s0];
        vector<int> simplex1 = simplexList[s1];

        // find vertices that are not shared
        int i0 = 0;
        int i1 = 0;
        int n0 = 2;
        int n1 = 2;
        int equalNum = 0;
        while (i0 < 3 && i1 < 3) {
            if (simplex0[i0] < simplex1[i1]) {
                n0 = i0;
                i0++;
            } else if (simplex0[i0] > simplex1[i1]) {
                n1 = i1;
                i1++;
            } else {
                equalNum++;
                i0++;
                i1++;
            }
        }
        cout << "connectSimplices before return equalNum!=2" << endl;
        
        if (equalNum != 2) return;
        // we found neighbors, update links
        cout << "before simplex0 = {";
        for (int i = 0; i < 6; i++){
            cout << simplex0[i] << ", ";
        }
        cout << "}" << endl;
        simplexList[s0][3 + n0] = s1;
        simplexList[s1][3 + n1] = s0;
        cout << " after simplex0 = {";
        for (int i = 0; i < 6; i++){
            cout << simplex0[i] << ", ";
        }
        cout << "}" << endl;

        /*test starts*/
        cout << "connectSimplices" << endl;
        cout << "simplexList.size(): " << simplexList.size() << endl;   
        for (int i = 0; i < simplexList.size(); ++i) {
            for (int j = 0 ; j < 6; ++j) {
                cout << "simplexList[" << i <<"]["<<j<<"]: " << simplexList[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
        /*test ends*/
    }

    shared_ptr<vector<vec3<float> > > Geodesation::getVertexList() {
        
        /*test starts*/
        cout << "getVertexList" << endl;
        cout << "vertexList.size(): " << vertexList.size() << endl;   
        for (int i = 0; i < vertexList.size(); ++i) {
            cout << "vertexList[" << i << "]: "
                 << vertexList[i].x << "\t" << vertexList[i].y 
                 << "\t" << vertexList[i].z << endl;
        }
        /*test ends*/
        m_vertexList = make_shared<vector<vec3<float> > >(vertexList);
        return m_vertexList;
    }

    vector<unordered_set<int> > Geodesation::getNeighborList() {
        vector<unordered_set<int> > network;
        network.resize(vertexList.size());


        for (auto& i: simplexList) {
            
            network.at(i[0]).insert(i[1]);
            network.at(i[0]).insert(i[2]);
            network.at(i[1]).insert(i[2]);
            network.at(i[1]).insert(i[0]);
            network.at(i[2]).insert(i[0]);
            network.at(i[2]).insert(i[1]);
        }
        /*test starts*/
        cout << "getNeighborList" << endl;
        cout << "network.size(): " << network.size() << endl;   
        for (int i = 0; i < network.size(); ++i) {
            for (auto it = network[i].begin(); it != network[i].end(); ++it) {
            cout << "network[" << i << "]: " << *it << endl;
            }
        }
        
        /*test ends*/

        return network;
    }
    

    int Geodesation::findNeighborMidVertex(vector<int> points, int s) {
        // make use of the previously stored midpoints
        points.resize(6);
        cout << "findNeighborMidVertex0" << endl;
        cout << "s = " << s << endl;
        cout << "points[3] = " << points[3] << endl;
        cout << "points[4] = " << points[4] << endl;
        cout << "points[5] = " << points[5] << endl;
        if (s == points[3]) {
            cout << "points[3]" << endl;
            return points[0];
        }
        cout << "findNeighborMidVertex1" << endl;
        if (s == points[4]) return points[1];
        cout << "findNeighborMidVertex2" << endl;
        if (s == points[5]) return points[2];
        cout << "findNeighborMidVertex" << endl;
        return -1;
    }

    // vertex inbetween the vertices with indices i0 and i1
    int Geodesation::createMidVertex(int i0, int i1) {
        vec3<float> v0 = vertexList[i0];
        vec3<float> v1 = vertexList[i1];

        // consider wrapping
        int sign = (v0.x * v1.x + v0.y * v1.y + v0.z * v1.z < 0.0 ? -1 : 1);
        float x = v0.x + sign * v1.x;
        float y = v0.y + sign * v1.y;
        float z = v0.z + sign * v1.z;
        float n = sqrt(x * x + y * y + z * z);
        cout << "createMidVertex" << endl;
        return createVertex(x / n, y / n, z / n);
    }

}; }; // end namespace freud::symmetry
