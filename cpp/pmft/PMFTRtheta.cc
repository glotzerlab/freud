#include "PMFTRtheta.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include <iostream>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include "VectorMath.h"
#include "saruprng.h"
#include <math.h> //For acos

using namespace std;

using namespace tbb;

/*! \internal
    \file PMFTRtheta.cc
    \brief Routines for computing 3D anisotropic potential of mean force
*/

namespace freud { namespace pmft {

PMFTRtheta::PMFTRtheta(float max_R, float max_theta, unsigned int n_bins_R, unsigned int n_bins_theta)
    : m_box(box::Box()), m_max_R(max_R), m_max_theta(max_theta),
      m_n_bins_R(n_bins_R), m_n_bins_theta(n_bins_theta), m_frame_counter(0),
      m_n_ref(0), m_n_p(0), m_n_q(0), m_reduce(true)
    {
    if (n_bins_R < 1)
        throw invalid_argument("must be at least 1 bin in R");
    if (n_bins_theta < 1)
        throw invalid_argument("must be at least 1 bin in theta");
    if (max_R < 0.0f)
        throw invalid_argument("max_R must be positive");
    if (max_theta < 0.0f)
        throw invalid_argument("max_theta must be positive");

    // calculate dR, dtheta
    m_dR = m_max_R / float(m_n_bins_R);
    m_d_theta = m_max_theta / float(m_n_bins_theta);

    if (m_dR > max_R)
        throw invalid_argument("max_R must be greater than dx");
    if (m_d_theta > max_theta)
        throw invalid_argument("max_theta must be greater than d_theta");

    //m_jacobian = m_dR * m_d_theta;

    // precompute the bin center positions for R
    m_R_array = std::shared_ptr<float>(new float[m_n_bins_R], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_R; i++)
        {
        float R = float(i) * m_dR;
        float nextR = float(i+1) * m_dR;
        m_R_array.get()[i] = ((R + nextR) / 2.0);
        }

    // precompute the bin center positions for theta
    m_theta_array = std::shared_ptr<float>(new float[m_n_bins_theta], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_theta; i++)
        {
        float theta = float(i) * m_d_theta;
        float next_theta = float(i+1) * m_d_theta;
        m_theta_array.get()[i] = ((theta + next_theta) / 2.0);
        }



    m_lc = new locality::LinkCell(m_box, m_max_R);
    // create and populate the pcf_array
    m_pcf_array = std::shared_ptr<float>(new float[m_n_bins_R*m_n_bins_theta], std::default_delete<float[]>());
    memset((void*)m_pcf_array.get(), 0, sizeof(float)*m_n_bins_R*m_n_bins_theta);
    m_bin_counts = std::shared_ptr<unsigned int>(new unsigned int[m_n_bins_R*m_n_bins_theta], std::default_delete<unsigned int[]>());
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_n_bins_R*m_n_bins_theta);

    m_r_cut = m_max_R;

    m_lc = new locality::LinkCell(m_box, m_r_cut);

/*
    for (unsigned int i = 0; i < m_n_bins_R; i++)
        {
        for (unsigned int j = 0; j < m_n_bins_theta; j++)
            {
            float r = m_R_array.get()[i];
            float rand_value = m_rand_jacobian_array.get()[b_i((int)i, (int)j)]
            m_inv_jacobian_array.get()[b_i((int)i, (int)j)] = (float)1.0 / (r * m_dR * m_d_theta);
            }
        }
        */
    }

quat <float> gen_random_quat(float r1, float r2, float r3)
{
    float u1 = r1;
    float u2 = r2*2.0*M_PI;
    float u3 = r3*2.0*M_PI;
    float c2 = sqrtf(u1);
    float c1 = sqrtf(1 - u1);

    quat<float> return_quat = quat<float>(c1*sin(u2), vec3<float>(c1*cos(u2), c2*sin(u3), c2*cos(u3))); //TKTK: need to properly define quaternion

    return return_quat;
}

// Define a function that calculates the magnitude of the shortest angle between two quaternions
float separation_angle( quat<float> q1, quat<float> q2)
{
    quat<float> Qt = conj(q1) * q2;
    float sep_angle = 2*acos(Qt.s);

    return sep_angle;
}

// Function that can find an equivalent quaternion for qn that minimizes the separation angle between it and qref.
// equivalent_orientations should be a list of quaternions that correspond to equivalent orientations of your particle
quat <float> find_min_quat( quat<float> qref, quat<float> qn, quat<float> *equivalent_orientations, unsigned int n_q)
{
    //Use the first quaternion in equivalent_orientations as a reference. This is q0
    quat <float> q0 = equivalent_orientations[0];

    float angle_array[n_q];
    float min_angle;
    quat <float> min_quat, tq, q_test; //initiate quaternions that will be changing

    min_angle = 360.0;
    min_quat = qn;
    //For each quaternion in the list, undo the q0 rotation, then apply the rotation in equivalent_orientations[i]
    //If your particle has mirror symmetry, be sure that's accounted for in the supplied equivalent_orientations
    for (unsigned int i=0; i < n_q; i++)
    {
        tq = qn * conj(q0); //Create a temporary quaternion that undoes the first rotation
        q_test = tq*equivalent_orientations[i];

        angle_array[i] = separation_angle(qref, q_test);
        if (angle_array[i] < min_angle)
        {
            min_angle = angle_array[i];
            min_quat = q_test;
        }
    }
    return min_quat;
}


PMFTRtheta::~PMFTRtheta()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        delete[] (*i);
        }
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_rand_jacobian_array.begin(); i != m_local_rand_jacobian_array.end(); ++i)
        {
        delete[] (*i);
        }
    delete m_lc;
    }

void PMFTRtheta::computeJacobian(quat<float> *equivalent_orientations, unsigned int n_q)
    {
    //Create the jacobian array by generating a random distribution of angles calculated using randomly generated quaternions
    //m_rand_jacobian_array = std::shared_ptr<unsigned int>(new unsigned int[m_n_bins_R*m_n_bins_theta], std::default_delete<unsigned int[]>());
    //m_local_bin_counts.local() = new unsigned int [m_n_bins_R*m_n_bins_theta];
    m_local_rand_jacobian_array.local() = new unsigned int [m_n_bins_R*m_n_bins_theta];
    memset((void*)m_local_rand_jacobian_array.local(), 0, sizeof(unsigned int)*m_n_bins_R*m_n_bins_theta);
    unsigned int num_rand_quats = m_n_bins_R * m_n_bins_theta * 100000;
    unsigned int m_jacobian_counter = 0;

    parallel_for(blocked_range<size_t>(0,num_rand_quats),
        [=] (const blocked_range<size_t>& r)
            {
            // precalc some values for faster computation within the loop
            float dR_inv = 1.0f / m_dR;
            float d_theta_inv = 1.0f / m_d_theta;
            Index2D b_i = Index2D(m_n_bins_R, m_n_bins_theta);

            //Define a random number generator
            Saru saru(123, 456);

            for (size_t i = r.begin(); i != r.end(); i++)
                {
                    quat<float> Qr = gen_random_quat(saru.s<float>(0,1), saru.s<float>(0,1), saru.s<float>(0,1));
                    quat<float> Q_unit = quat<float>(1.0, vec3<float>(0, 0, 0)); //TKTK: Need to do this correctly. Almost certainly wrong

                    quat<float> min_quat = find_min_quat(Qr, Q_unit, equivalent_orientations, n_q);
                    float sep_angle = separation_angle(Qr, min_quat);

                    float rand_n = saru.s<float>(0, 1);
                    cout << "Random number is " << rand_n << endl;
                    float rand_r = rand_n * m_max_R;

                    float binR = rand_r * dR_inv;
                    float bin_theta = sep_angle * d_theta_inv;

                    // fast float to int conversion with truncation
                    #ifdef __SSE2__
                    unsigned int ibinR = _mm_cvtt_ss2si(_mm_load_ss(&binR));
                    unsigned int ibin_theta = _mm_cvtt_ss2si(_mm_load_ss(&bin_theta));
                    #else
                    unsigned int ibinR = (unsigned int)(binR);
                    unsigned int ibin_theta = (unsigned int)(bin_theta);
                    #endif
                    //cout << 'Distance: ' << r << ' Bin: ' << ibinR << '    ';
                    //cout << '\nDistance: ' << ibinR;

                    // increment the bin
                    if ((ibinR < m_n_bins_R) && (ibin_theta < m_n_bins_theta))
                        {
                        ++m_local_rand_jacobian_array.local()[b_i(ibinR, ibin_theta)];
                        //m_jacobian_counter++;
                        }
                }
            });

    // calculate the jacobian array; calc'd as the inv for faster use later
    // Calculating this based on randomly drawing quaternions and see where they fall
    m_inv_jacobian_array = std::shared_ptr<float>(new float[m_n_bins_R*m_n_bins_theta], std::default_delete<float[]>());
    memset((void*)m_inv_jacobian_array.get(), 0, sizeof(unsigned int)*m_n_bins_R*m_n_bins_theta);
    parallel_for(blocked_range<size_t>(0,m_n_bins_R),
        [=] (const blocked_range<size_t>& r)
        {
        //Index2D b_i = Index2D(m_n_bins_theta, m_n_bins_R);
        Index2D b_i = Index2D(m_n_bins_R, m_n_bins_theta);
        for (size_t i = r.begin(); i != r.end(); i++)
            {
            for (size_t j = 0; j < m_n_bins_theta; j++)
                {
                for (tbb::enumerable_thread_specific<unsigned int *>::const_iterator local_bins = m_local_rand_jacobian_array.begin();
                     local_bins != m_local_rand_jacobian_array.end(); ++local_bins)
                    {
                    //TKTK: Check that this is the proper value for the inverse jacobian.
                    //m_inv_jacobian_array.get()[b_i((int)i, (int)j)] += (float)(*local_bins)[b_i((int)i, (int)j)] / (float)m_jacobian_counter;
                    m_inv_jacobian_array.get()[b_i((int)i, (int)j)] += (float)(*local_bins)[b_i((int)i, (int)j)] / (float) num_rand_quats;
                    }
                }
            }
        });
    }


//! \internal
//! helper function to reduce the thread specific arrays into the boost array
void PMFTRtheta::reducePCF()
    {
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_n_bins_R*m_n_bins_theta);
    memset((void*)m_pcf_array.get(), 0, sizeof(float)*m_n_bins_R*m_n_bins_theta);
    //parallel_for(blocked_range<size_t>(0,m_n_bins_theta),
    parallel_for(blocked_range<size_t>(0,m_n_bins_R),
        [=] (const blocked_range<size_t>& r)
        {
        //Index2D b_i = Index2D(m_n_bins_theta, m_n_bins_R);
        Index2D b_i = Index2D(m_n_bins_R, m_n_bins_theta);
        for (size_t i = r.begin(); i != r.end(); i++)
            {
            for (size_t j = 0; j < m_n_bins_theta; j++)
                {
                for (tbb::enumerable_thread_specific<unsigned int *>::const_iterator local_bins = m_local_bin_counts.begin();
                     local_bins != m_local_bin_counts.end(); ++local_bins)
                    {
                    m_bin_counts.get()[b_i((int)i, (int)j)] += (*local_bins)[b_i((int)i, (int)j)];
                    }
                }
            }
        });

    float inv_num_dens = m_box.getVolume() / (float)m_n_p;
    float inv_jacobian = (float) 1.0 / (float) m_jacobian;
    float norm_factor = (float) 1.0 / ((float) m_frame_counter * (float) m_n_ref);
    // normalize pcf_array
    parallel_for(blocked_range<size_t>(0,m_n_bins_R*m_n_bins_theta),
        [=] (const blocked_range<size_t>& r)
            {
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                //m_pcf_array.get()[i] = (float)m_bin_counts.get()[i] * norm_factor * inv_jacobian * inv_num_dens;
                m_pcf_array.get()[i] = (float)m_bin_counts.get()[i] * norm_factor * m_inv_jacobian_array.get()[i] * inv_num_dens;
                }
            });
    }

//! Get a reference to the PCF array
std::shared_ptr<unsigned int> PMFTRtheta::getBinCounts()
    {
    if (m_reduce == true)
        {
        reducePCF();
        }
    m_reduce = false;
    return m_bin_counts;
    }

//! Get a reference to the PCF array
std::shared_ptr<float> PMFTRtheta::getPCF()
    {
    if (m_reduce == true)
        {
        reducePCF();
        }
    m_reduce = false;
    return m_pcf_array;
    }

//! \internal
/*! \brief Function to reset the pcf array if needed e.g. calculating between new particle types
*/
void PMFTRtheta::resetPCF()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_n_bins_R*m_n_bins_theta);
        }
    m_frame_counter = 0;
    m_reduce = true;
    }

//! \internal
/*! \brief Helper function to direct the calculation to the correct helper class
*/
void PMFTRtheta::accumulate(box::Box& box,
                        vec3<float> *ref_points,
                        quat<float> *ref_orientations,
                        unsigned int n_ref,
                        vec3<float> *points,
                        quat<float> *orientations,
                        unsigned int n_p,
                        quat<float> *equivalent_orientations,
                        unsigned int n_q)   // quat<float> *equivalent_orientations, unsigned int n_q)
    {
    m_box = box;
    m_lc->computeCellList(m_box, points, n_p);
    parallel_for(blocked_range<size_t>(0,n_ref),
        [=] (const blocked_range<size_t>& r)
            {
            assert(ref_points);
            assert(points);
            assert(n_ref > 0);
            assert(n_p > 0);
            assert(n_q > 0);

            // precalc some values for faster computation within the loop
            float dR_inv = 1.0f / m_dR;
            float d_theta_inv = 1.0f / m_d_theta;
            float maxrsq = m_max_R * m_max_R;
            //Don't need a max theta sq

            Index2D b_i = Index2D(m_n_bins_R, m_n_bins_theta);
            //Index2D b_i = Index2D(m_n_bins_theta, m_n_bins_R);
            //Index2D q_i = Index2D(n_faces, n_p);

            bool exists;
            m_local_bin_counts.local(exists);
            if (! exists)
                {
                m_local_bin_counts.local() = new unsigned int [m_n_bins_R*m_n_bins_theta];
                memset((void*)m_local_bin_counts.local(), 0, sizeof(unsigned int)*m_n_bins_R*m_n_bins_theta);
                }

            // for each reference point
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                // get the cell the point is in
                vec3<float> ref = ref_points[i];
                // create the reference point quaternion
                quat<float> ref_q(ref_orientations[i]);
                unsigned int ref_cell = m_lc->getCell(ref);

                // loop over all neighboring cells
                const std::vector<unsigned int>& neigh_cells = m_lc->getCellNeighbors(ref_cell);
                for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
                    {
                    unsigned int neigh_cell = neigh_cells[neigh_idx];

                    // iterate over the particles in that cell
                    locality::LinkCell::iteratorcell it = m_lc->itercell(neigh_cell);
                    for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                        {
                        // make sure that the particles are wrapped into the box
                        vec3<float> delta = m_box.wrap(points[j] - ref);
                        float rsq = dot(delta, delta);

                        // check that the particle is not checking itself
                        // 1e-6 is an arbitrary value that could be set differently if needed
                        if (rsq < 1e-6)
                            {
                            continue;
                            }

                        //Need the orientation of the reference particle,
                        //the orientation of the neighbor particle, and the separation R between reference and neighbor
                        if (rsq < maxrsq)
                            {
                            quat <float> minimizing_quat = find_min_quat(ref_orientations[i], orientations[j], equivalent_orientations, n_q);
                            float sep_angle = separation_angle(ref_orientations[i], minimizing_quat);

                            float r = sqrtf(rsq);
                            float binR = r * dR_inv;
                            float bin_theta = sep_angle * d_theta_inv;
                            //float bin_theta = 0;

                            // fast float to int conversion with truncation
                            #ifdef __SSE2__
                            unsigned int ibinR = _mm_cvtt_ss2si(_mm_load_ss(&binR));
                            unsigned int ibin_theta = _mm_cvtt_ss2si(_mm_load_ss(&bin_theta));
                            #else
                            unsigned int ibinR = (unsigned int)(binR);
                            unsigned int ibin_theta = (unsigned int)(bin_theta);
                            #endif
                            //cout << 'Distance: ' << r << ' Bin: ' << ibinR << '    ';
                            //cout << '\nDistance: ' << ibinR;


                            // increment the bin
                            if ((ibinR < m_n_bins_R) && (ibin_theta < m_n_bins_theta))
                                {
                                ++m_local_bin_counts.local()[b_i(ibinR, ibin_theta)];
                                //++m_local_bin_counts.local()[b_i(ibin_theta, ibinR)];
                                }
                            }
                        }
                    }
                } // done looping over reference points
            });
    //cout << m_bin_counts;
    m_frame_counter++;
    m_n_ref = n_ref;
    m_n_p = n_p;
    m_n_q = n_q;
    //m_n_faces = n_faces;
    // flag to reduce
    m_reduce = true;
    }

}; }; // end namespace freud::pmft
