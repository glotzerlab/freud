#include "PMFTScalarInvariants.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include "VectorMath.h"

using namespace std;

using namespace tbb;

/*! \internal
    \file PMFTScalarInvariants.cc
    \brief Routines for computing 3D anisotropic potential of mean force
*/

namespace freud { namespace pmft {

PMFTScalarInvariants::PMFTScalarInvariants(float max_R, float max_S, float max_U, float max_RW, float max_RV, float max_WT,
                                        unsigned int n_bins_R, unsigned int n_bins_S, unsigned int n_bins_U,
                                        unsigned int n_bins_RW, unsigned int n_bins_RV, unsigned int n_bins_WT)
    : m_box(box::Box()), m_max_R(max_R), m_max_S(max_S), m_max_R(max_R), m_max_U(max_U), m_max_RW(max_RW), m_max_RV(max_RV), m_max_WT(max_WT),
      m_n_bins_R(n_bins_R), m_n_bins_S(n_bins_S), m_n_bins_U(n_bins_U),
      m_n_bins_RW(n_bins_RW), m_n_bins_RV(n_bins_RV), m_n_bins_WT(n_bins_WT),
      m_frame_counter(0), m_n_ref(0), m_n_p(0) m_reduce(true)
    {
    if (n_bins_R < 1)
        throw invalid_argument("must be at least 1 bin in R");
    if (n_bins_S< 1)
        throw invalid_argument("must be at least 1 bin in S");
    if (n_bins_U < 1)
        throw invalid_argument("must be at least 1 bin in U");
    if (n_bins_RW < 1)
        throw invalid_argument("must be at least 1 bin in RW");
    if (n_bins_RV< 1)
        throw invalid_argument("must be at least 1 bin in RV");
    if (n_bins_WT < 1)
        throw invalid_argument("must be at least 1 bin in WT");

    if (max_R < 0.0f)
        throw invalid_argument("max_R must be positive");
    if (max_S < 0.0f)
        throw invalid_argument("max_S must be positive");
    if (max_U < 0.0f)
        throw invalid_argument("max_U must be positive");
    if (max_RW < 0.0f)
        throw invalid_argument("max_RW must be positive");
    if (max_RV < 0.0f)
        throw invalid_argument("max_RV must be positive");
    if (max_WT < 0.0f)
        throw invalid_argument("max_WT must be positive");


    // calculate dx, dy, dz
    m_dR = 2.0 * m_max_R / float(m_n_bins_R);
    m_dS = 2.0 * m_max_S / float(m_n_bins_S);
    m_dU = 2.0 * m_max_U / float(m_n_bins_U);
    m_dRW = 2.0 * m_max_RW / float(m_n_bins_RW);
    m_dRV = 2.0 * m_max_RV / float(m_n_bins_RV);
    m_dWT = 2.0 * m_max_WT / float(m_n_bins_WT);


    if (m_dR > max_R)
        throw invalid_argument("max_R must be greater than dR");
    if (m_dS > max_S)
        throw invalid_argument("max_S must be greater than dS");
    if (m_dU > max_U)
        throw invalid_argument("max_U must be greater than dU");
    if (m_dRW > max_RW)
        throw invalid_argument("max_RW must be greater than dRW");
    if (m_dRV > max_RV)
        throw invalid_argument("max_RV must be greater than dRV");
    if (m_dWT > max_WT)
        throw invalid_argument("max_WT must be greater than dWT");




    //Note, this is an incorrect jacobian.
    m_jacobian = m_dR * m_dS * m_dU * m_dRW * m_dRV * m_dWT;

    // precompute the bin center positions for R
    m_R_array = std::shared_ptr<float>(new float[m_n_bins_R], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_R; i++)
        {
        float R = float(i) * m_dR;
        float nextR = float(i+1) * m_dR;
        m_R_array.get()[i] = ((R + nextR) / 2.0);
        }

    // precompute the bin center positions for S
    m_S_array = std::shared_ptr<float>(new float[m_n_bins_S], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_S; i++)
        {
        float S = float(i) * m_dS;
        float nextS = float(i+1) * m_dS;
        m_S_array.get()[i] = ((S + nextS) / 2.0);
        }
    // precompute the bin center positions for U
    m_U_array = std::shared_ptr<float>(new float[m_n_bins_U], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_U; i++)
        {
        float U = float(i) * m_dU;
        float nextU = float(i+1) * m_dU;
        m_U_array.get()[i] = ((U + nextU) / 2.0);
        }

    // precompute the bin center positions for RW
    m_RW_array = std::shared_ptr<float>(new float[m_n_bins_RW], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_RW; i++)
        {
        float RW = float(i) * m_dRW;
        float nextRW = float(i+1) * m_dRW;
        m_RW_array.get()[i] = ((RW + nextRW) / 2.0);
        }

    // precompute the bin center positions for RV
    m_RV_array = std::shared_ptr<float>(new float[m_n_bins_RV], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_RV; i++)
        {
        float RV = float(i) * m_dRV;
        float nextRV = float(i+1) * m_dRV;
        m_RV_array.get()[i] = ((RV + nextRV) / 2.0);
        }

    // precompute the bin center positions for WT
    m_WT_array = std::shared_ptr<float>(new float[m_n_bins_WT], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_WT; i++)
        {
        float WT = float(i) * m_dWT;
        float nextWT = float(i+1) * m_dWT;
        m_WT_array.get()[i] = ((WT + nextWT) / 2.0);
        }

    m_r_cut = m_max_R;

    m_lc = new locality::LinkCell(m_box, m_r_cut);
    }

PMFTScalarInvariants::~PMFTScalarInvariants()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        delete[] (*i);
        }
    delete m_lc;
    }

//! \internal
//! helper function to reduce the thread specific arrays into the boost array
void PMFTScalarInvariants::reducePCF()
    {
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_n_bins_R*m_n_bins_S*m_n_bins_U*m_n_bins_RW*m_n_bins_RV*m_n_bins_WT);
    memset((void*)m_pcf_array.get(), 0, sizeof(float)*m_n_bins_R*m_n_bins_S*m_n_bins_U*m_n_bins_RW*m_n_bins_RV*m_n_bins_WT);
    parallel_for(blocked_range<size_t>(0,m_n_bins_R),
        [=] (const blocked_range<size_t>& r)
            {
            Index6D b_i = Index6D(m_n_bins_R, m_n_bins_S, m_n_bins_U, m_n_bins_RW, m_n_bins_RV, m_n_bins_WT);
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                for (size_t j = 0; j < m_n_bins_S; j++)
                    {
                    for (size_t k = 0; k < m_n_bins_U; k++)
                        {
                        for (size_t l = 0; l < m_n_bins_RW; l++)
                            {
                            for (size_t m = 0; m < m_n_bins_RV; m++)
                                {
                                for (size_t n = 0; n < m_n_bins_WT; n++)
                                    {
                                    for (tbb::enumerable_thread_specific<unsigned int *>::const_iterator local_bins = m_local_bin_counts.begin();
                                         local_bins != m_local_bin_counts.end(); ++local_bins)
                                        {
                                        m_bin_counts.get()[b_i((int)i, (int)j, (int)k, (int)l, (int)m, (int)n)] += (*local_bins)[b_i((int)i, (int)j, (int)k, (int)l, (int)m, (int)n)];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
    float inv_num_dens = m_box.getVolume() / (float)m_n_p;
    float inv_jacobian = (float) 1.0 / (float) m_jacobian;
    float norm_factor = (float) 1.0 / ((float) m_frame_counter * (float) m_n_ref);// * (float) m_n_faces);
    // normalize pcf_array
    parallel_for(blocked_range<size_t>(0,m_n_bins_R*m_n_bins_S*m_n_bins_U*m_n_bins_RW*m_n_bins_RV*m_n_bins_WT),
        [=] (const blocked_range<size_t>& r)
            {
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                m_pcf_array.get()[i] = (float)m_bin_counts.get()[i] * norm_factor * inv_jacobian * inv_num_dens;
                }
            });
    }

//! Get a reference to the PCF array
std::shared_ptr<unsigned int> PMFTScalarInvariants::getBinCounts()
    {
    if (m_reduce == true)
        {
        reducePCF();
        }
    m_reduce = false;
    return m_bin_counts;
    }

//! Get a reference to the PCF array
std::shared_ptr<float> PMFTXYZ::getPCF()
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
void PMFTScalarInvariants::resetPCF()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_n_bins_R*m_n_bins_S*m_n_bins_U*m_n_bins_RW*m_n_bins_RV*m_n_bins_WT);
        }
    m_frame_counter = 0;
    m_reduce = true;
    }

//! \internal
/*! \brief Helper function to direct the calculation to the correct helper class
*/
void PMFTScalarInvariants::accumulate(box::Box& box,
                        vec3<float> *ref_points,
                        quat<float> *ref_orientations,
                        unsigned int n_ref,
                        vec3<float> *points,
                        quat<float> *orientations,
                        unsigned int n_p);
                        //quat<float> *face_orientations,
                        //unsigned int n_faces)
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
            assert(n_faces > 0);

            // precalc some values for faster computation within the loop
            float dR_inv = 1.0f / m_dR;
            float maxRsq = m_max_R * m_max_R;
            float dS_inv = 1.0f / m_dS;
            float maxSsq = m_max_S * m_max_S;
            float dU_inv = 1.0f / m_dU;
            float maxUsq = m_max_U * m_max_U;
            float dRW_inv = 1.0f / m_dRW;
            float maxRWsq = m_max_RW * m_max_RW;
            float dRV_inv = 1.0f / m_dRV;
            float maxRVsq = m_max_RV * m_max_RV;
            float dWT_inv = 1.0f / m_dWT;
            float maxWTsq = m_max_WT * m_max_WT;


            Index6D b_i = Index6D(m_n_bins_R, m_n_bins_S, m_n_bins_U, m_n_bins_RW, m_n_bins_RV, m_n_bins_WT);
            //Index1D q_i = Index1D(n_p);

            bool exists;
            m_local_bin_counts.local(exists);
            if (! exists)
                {
                m_local_bin_counts.local() = new unsigned int [m_n_bins_R*m_n_bins_S*m_n_bins_U*m_n_bins_RW*m_n_bins_RV*m_n_bins_WT];
                memset((void*)m_local_bin_counts.local(), 0, sizeof(unsigned int)*m_n_bins_R*m_n_bins_S*m_n_bins_U*m_n_bins_RW*m_n_bins_RV*m_n_bins_WT);
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
                        for (unsigned int k=0; k<n_faces; k++)
                            {
                            // create tmp vector
                            vec3<float> my_vector(delta);
                            // rotate vector
                            // create the extra quaternion
                            quat<float> qe(face_orientations[q_i(k, i)]);
                            // create point vector
                            vec3<float> v(delta);
                            // rotate the vector
                            v = rotate(conj(ref_q), v);
                            v = rotate(qe, v);

                            float x = v.x + m_max_x;
                            float y = v.y + m_max_y;
                            float z = v.z + m_max_z;

                            // bin that point
                            float binx = floorf(x * dx_inv);
                            float biny = floorf(y * dy_inv);
                            float binz = floorf(z * dz_inv);
                            // fast float to int conversion with truncation
                            #ifdef __SSE2__
                            unsigned int ibinx = _mm_cvtt_ss2si(_mm_load_ss(&binx));
                            unsigned int ibiny = _mm_cvtt_ss2si(_mm_load_ss(&biny));
                            unsigned int ibinz = _mm_cvtt_ss2si(_mm_load_ss(&binz));
                            #else
                            unsigned int ibinx = (unsigned int)(binx);
                            unsigned int ibiny = (unsigned int)(biny);
                            unsigned int ibinz = (unsigned int)(binz);
                            #endif

                            // increment the bin
                            if ((ibinx < m_n_bins_x) && (ibiny < m_n_bins_y) && (ibinz < m_n_bins_z))
                                {
                                ++m_local_bin_counts.local()[b_i(ibinx, ibiny, ibinz)];
                                }
                            }
                        }
                    }
                } // done looping over reference points
            });
    m_frame_counter++;
    m_n_ref = n_ref;
    m_n_p = n_p;
    m_n_faces = n_faces;
    // flag to reduce
    m_reduce = true;
    }

}; }; // end namespace freud::pmft
