// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>
#include <tbb/tbb.h>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "Pairing2D.h"

using namespace std;
using namespace tbb;

namespace freud { namespace environment {

Pairing2D::Pairing2D(const float rmax,
                     const unsigned int k,
                     const float comp_dot_tol)
    : m_box(box::Box()), m_rmax(rmax), m_Np(0), m_No(0), m_comp_dot_tol(comp_dot_tol)
    {
    // create the unsigned int array to store whether or not a particle is paired
    m_match_array = std::shared_ptr<unsigned int>(new unsigned int[m_Np], std::default_delete<unsigned int[]>());
    for (unsigned int i = 0; i < m_Np; i++)
        {
        m_match_array.get()[i] = 0;
        }
    // create the pairing array to store particle pairs
    // m_pair_array[i] will have the pair of particle i stored at idx=i
    // if there is no pairing, it will store itself
    m_pair_array = std::shared_ptr<unsigned int>(new unsigned int[m_Np], std::default_delete<unsigned int[]>());
    for (unsigned int i = 0; i < m_Np; i++)
        {
        m_pair_array.get()[i] = i;
        }
    }

Pairing2D::~Pairing2D()
    {
    }

void Pairing2D::ComputePairing2D(const freud::locality::NeighborList *nlist,
                                 const vec3<float> *points,
                                 const float *orientations,
                                 const float *comp_orientations,
                                 const unsigned int Np,
                                 const unsigned int No)
    {
    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());
    size_t bond(0);

    // for each particle
    // there's the problem right there
    Index2D b_i = Index2D(No, Np);
    for (unsigned int i = 0; i < Np; i++)
        {
        // get the position of particle i
        const vec2<float> r_i(points[i].x, points[i].y);
        bool is_paired = false;

        if(bond < nlist->getNumBonds() && neighbor_list[2*bond] < i)
            bond = nlist->find_first_index(i);

        for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
            {
            const size_t j(neighbor_list[2*bond + 1]);
            // once a particle is paired we can stop
            if (m_match_array.get()[i] != 0)
                {
                break;
                }
            // get the position of particle j
            const vec2<float> r_j(points[j].x, points[j].y);
            // find interparticle vectors
            vec2<float> r_ij(r_j - r_i);
            vec2<float> r_ji(r_i - r_j);
            // wrap into box
            vec3<float> delta = m_box.wrap(vec3<float>(r_ij.x, r_ij.y, 0.0));
            r_ij = vec2<float>(delta.x, delta.y);
            delta = m_box.wrap(vec3<float>(r_ji.x, r_ji.y, 0.0));
            r_ji = vec2<float>(delta.x, delta.y);
            // find the squared distance for better computational efficiency
            float rsq(dot(r_ij, r_ij));

            // will skip same particle
            // shouldn't actually be needed
            if (rsq > 1e-6)
                {
                // check if the particles are paired
                // particles are paired if they are the nearest neighbors that have the complementary vector
                // pointing in the same direction as the paired centroid vectorc

                // rotate the unit interparticle vector
                rotmat2<float> my_mat = rotmat2<float>::fromAngle(-orientations[i]);
                vec2<float> u_ij(r_ij/sqrt(rsq));
                u_ij = my_mat * u_ij;
                u_ij = u_ij / sqrt(dot(u_ij, u_ij));
                my_mat = rotmat2<float>::fromAngle(-orientations[j]);
                rsq = dot(r_ji, r_ji);
                vec2<float> u_ji(r_ji/sqrt(rsq));
                u_ji = my_mat * u_ji;
                u_ji = u_ji / sqrt(dot(u_ji, u_ji));

                // for each potential complementary orientation for particle i
                for (unsigned int a=0; a<No; a++)
                    {
                    // break once pair is detected
                    if (is_paired == true)
                        {
                        break;
                        }
                    // generate vectors
                    float theta_ci = comp_orientations[b_i(a, i)];
                    vec2<float> c_i(cosf(theta_ci), sinf(theta_ci));
                    c_i = c_i / sqrt(dot(c_i, c_i));

                    // for each potential complementary orientation for particle j
                    for (unsigned int b=0; b<No; b++)
                        {
                        // break once pair is detected
                        if (is_paired == true)
                            {
                            break;
                            }
                        // generate vectors
                        float theta_cj = comp_orientations[b_i(b, j)];
                        vec2<float> c_j(cosf(theta_cj), sinf(theta_cj));
                        c_j = c_j / sqrt(dot(c_j, c_j));
                        // calculate the angle between vectors using cross and dot method
                        // only use the abs angle (direction is not necessary)
                        // using method from http://stackoverflow.com/questions/21483999/using-atan2-to-find-angle-between-two-vectors
                        float d_ij = abs(atan2(((c_i.x*u_ij.y)-(c_i.y*u_ij.x)),dot(c_i, u_ij)));
                        float d_ji = abs(atan2(((c_j.x*u_ji.y)-(c_j.y*u_ji.x)),dot(c_j, u_ji)));
                        // As the nearest neighbor list may use a larger rmax than was initialized, it has
                        // to check again
                        if ((d_ij < m_comp_dot_tol) && (d_ji < m_comp_dot_tol) && (is_paired==false) && (rsq < (m_rmax * m_rmax)))
                            {
                            m_match_array.get()[i] = 1;
                            m_pair_array.get()[i] = j;
                            is_paired = true;
                            } // done pairing particle
                        } // done checking all orientations of j
                    } // done checking all orientations of i
                } // done with not doing if the same particle (which should not happen)
            } // done looping over neighbors
        } // done looping over reference points
    }

void Pairing2D::compute(box::Box& box,
                        const freud::locality::NeighborList *nlist,
                        const vec3<float>* points,
                        const float* orientations,
                        const float* comp_orientations,
                        const unsigned int Np,
                        const unsigned int No)
    {
    m_box = box;
    // reallocate the output array if it is not the right size
    if (Np != m_Np)
        {
        m_match_array = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());
        m_pair_array = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());
        }
    // reset the arrays
    for (unsigned int i = 0; i < Np; i++)
        {
        m_match_array.get()[i] = 0;
        }
    for (unsigned int i = 0; i < Np; i++)
        {
        m_pair_array.get()[i] = i;
        }
    ComputePairing2D(nlist,
                     points,
                     orientations,
                     comp_orientations,
                     Np,
                     No);
    m_Np = Np;
    m_No = No;
    }

}; }; // end namespace freud::environment
