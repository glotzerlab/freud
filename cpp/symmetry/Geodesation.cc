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


    Geodesation::Geodesation(unsigned int iterations) {

        m_vertexList = shared_ptr<vector<vec3<float> > >(new vector<vec3<float> >());
        m_simplexList = shared_ptr<vector<vector<int> > >(new vector<vector<int> >());

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

        //maybe move to another function and expose it to python.
        for (int i = 0; i < iterations; ++i) {
            geodesate();
        }
    }

    Geodesation::~Geodesation(){

    }

    void Geodesation::geodesate() {
        auto oldSimplexList = m_simplexList;
        m_simplexList = shared_ptr<vector<vector<int> > >(new vector<vector<int> >());

        for (int n = 0; n < oldSimplexList->size(); n++) {
            vector<int> simplex = oldSimplexList->at(n);

            // vertex indices
            int i0 = simplex[0];
            int i1 = simplex[1];
            int i2 = simplex[2];
            // neighbor indices
            int n0 = simplex[3];
            int n1 = simplex[4];
            int n2 = simplex[5];

            int i12 = (n0 < n ? findNeighborMidVertex(oldSimplexList->at(n0), n) : createMidVertex(i1, i2));
            int i20 = (n1 < n ? findNeighborMidVertex(oldSimplexList->at(n1), n) : createMidVertex(i2, i0));
            int i01 = (n2 < n ? findNeighborMidVertex(oldSimplexList->at(n2), n) : createMidVertex(i0, i1));

            //store new midpoint vertices so we can reuse them
            oldSimplexList->at(n)[0] = i12;
            oldSimplexList->at(n)[1] = i20;
            oldSimplexList->at(n)[2] = i01;

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
        m_vertexList->push_back(vec3<float>(x, y, z));
        return m_vertexList->size() - 1;
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
        m_simplexList->push_back(temp);

        return m_simplexList->size() - 1;
    }

    void Geodesation::connectSimplices(int s0, int s1) {

        vector<int> simplex0 = m_simplexList->at(s0);
        vector<int> simplex1 = m_simplexList->at(s1);

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

        if (equalNum != 2) return;
        // we found neighbors, update links
        m_simplexList->at(s0)[3 + n0] = s1;
        m_simplexList->at(s1)[3 + n1] = s0;
    }

    shared_ptr<vector<vec3<float> > > Geodesation::getVertexList() {
        return m_vertexList;
    }

    shared_ptr<vector<unordered_set<int> > > Geodesation::getNeighborList() {
        auto network = shared_ptr<vector<unordered_set<int> > >(
            new vector<unordered_set<int> >[m_vertexList->size()],
            default_delete<vector<unordered_set<int> >[]>());
        network->resize(m_vertexList->size());

        for (auto& i: *m_simplexList) {
            network->at(i[0]).insert(i[1]);
            network->at(i[0]).insert(i[2]);
            network->at(i[1]).insert(i[2]);
            network->at(i[1]).insert(i[0]);
            network->at(i[2]).insert(i[0]);
            network->at(i[2]).insert(i[1]);
        }

        return network;
    }

    unsigned int Geodesation::getNVertices() {
        return m_vertexList->size();
    }

    unsigned int Geodesation::getNSimplices() {
        return m_simplexList->size();
    }

    int Geodesation::findNeighborMidVertex(vector<int> points, int s) {
        // make use of the previously stored midpoints
        points.resize(6);
        if (s == points[3]) {
            return points[0];
        }
        if (s == points[4]) {
            return points[1];
        }
        if (s == points[5]) {
            return points[2];
        }
        return -1;
    }

    // vertex inbetween the vertices with indices i0 and i1
    int Geodesation::createMidVertex(int i0, int i1) {
        vec3<float> v0 = m_vertexList->at(i0);
        vec3<float> v1 = m_vertexList->at(i1);

        // consider wrapping
        int sign = (v0.x * v1.x + v0.y * v1.y + v0.z * v1.z < 0.0 ? -1 : 1);
        float x = v0.x + sign * v1.x;
        float y = v0.y + sign * v1.y;
        float z = v0.z + sign * v1.z;
        float n = sqrt(x * x + y * y + z * z);
        return createVertex(x / n, y / n, z / n);
    }

}; }; // end namespace freud::symmetry
