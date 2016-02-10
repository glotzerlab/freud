#include "PMFTR12.h"
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
    \file PMFTR12.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace pmft {

PMFTR12::PMFTR12(float max_r, unsigned int nbins_r, unsigned int nbins_T1, unsigned int nbins_T2)
    : m_box(trajectory::Box()), m_max_r(max_r), m_max_T1(2.0*M_PI), m_max_T2(2.0*M_PI),
      m_nbins_r(nbins_r), m_nbins_T1(nbins_T1), m_nbins_T2(nbins_T2)
    {
    if (nbins_r < 1)
        throw invalid_argument("must be at least 1 bin in r");
    if (nbins_T1 < 1)
        throw invalid_argument("must be at least 1 bin in T1");
    if (nbins_T2 < 1)
        throw invalid_argument("must be at least 1 bin in T2");
    if (max_r < 0.0f)
        throw invalid_argument("max_r must be positive");
    // calculate dr, dT1, dT2
    m_dr = m_max_r / float(m_nbins_r);
    m_dT1 = m_max_T1 / float(m_nbins_T1);
    m_dT2 = m_max_T2 / float(m_nbins_T2);

    if (m_dr > max_r)
        throw invalid_argument("max_r must be greater than dr");
    if (m_dT1 > m_max_T1)
        throw invalid_argument("max_T1 must be greater than dT1");
    if (m_dT2 > m_max_T2)
        throw invalid_argument("max_T2 must be greater than dT2");

    // precompute the bin center positions for r
    m_r_array = boost::shared_array<float>(new float[m_nbins_r]);
    for (unsigned int i = 0; i < m_nbins_r; i++)
        {
        float r = float(i) * m_dr;
        float nextr = float(i+1) * m_dr;
        m_r_array[i] = 2.0f / 3.0f * (nextr*nextr*nextr - r*r*r) / (nextr*nextr - r*r);
        }

    // precompute the bin center positions for T1
    m_T1_array = boost::shared_array<float>(new float[m_nbins_T1]);
    for (unsigned int i = 0; i < m_nbins_T1; i++)
        {
        float T1 = float(i) * m_dT1;
        float nextT1 = float(i+1) * m_dT1;
        m_T1_array[i] = ((T1 + nextT1) / 2.0);
        }

    // precompute the bin center positions for T2
    m_T2_array = boost::shared_array<float>(new float[m_nbins_T2]);
    for (unsigned int i = 0; i < m_nbins_T2; i++)
        {
        float T2 = float(i) * m_dT2;
        float nextT2 = float(i+1) * m_dT2;
        m_T2_array[i] = ((T2 + nextT2) / 2.0);
        }

    // create and populate the pcf_array
    m_pcf_array = boost::shared_array<unsigned int>(new unsigned int[m_nbins_r*m_nbins_T1*m_nbins_T2]);
    memset((void*)m_pcf_array.get(), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_T1*m_nbins_T2);

    m_lc = new locality::LinkCell(m_box, m_max_r);
    }

PMFTR12::~PMFTR12()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_pcf_array.begin(); i != m_local_pcf_array.end(); ++i)
        {
        delete[] (*i);
        }
    delete m_lc;
    }

class CombinePCFR12
    {
    private:
        unsigned int m_nbins_r;
        unsigned int m_nbins_T1;
        unsigned int m_nbins_T2;
        unsigned int *m_pcf_array;
        tbb::enumerable_thread_specific<unsigned int *>& m_local_pcf_array;
    public:
        CombinePCFR12(unsigned int nbins_r,
                      unsigned int nbins_T1,
                      unsigned int nbins_T2,
                      unsigned int *pcf_array,
                      tbb::enumerable_thread_specific<unsigned int *>& local_pcf_array)
            : m_nbins_r(nbins_r), m_nbins_T1(nbins_T1), m_nbins_T2(nbins_T2), m_pcf_array(pcf_array),
              m_local_pcf_array(local_pcf_array)
        {
        }
        void operator()( const blocked_range<size_t> &myBin ) const
            {
            Index3D b_i = Index3D(m_nbins_T1, m_nbins_T2, m_nbins_r);
            // T1
            for (size_t i = myBin.begin(); i != myBin.end(); i++)
                {
                // T2
                for (size_t j = 0; j < m_nbins_T2; j++)
                    {
                    for (size_t k = 0; k < m_nbins_r; k++)
                        {
                        for (tbb::enumerable_thread_specific<unsigned int *>::const_iterator local_bins = m_local_pcf_array.begin();
                             local_bins != m_local_pcf_array.end(); ++local_bins)
                            {
                            m_pcf_array[b_i((int)i, (int)j, (int)k)] += (*local_bins)[b_i((int)i, (int)j, (int)k)];
                            }
                        }
                    }
                }
            }
    };

class ComputePMFTR12
    {
    private:
        tbb::enumerable_thread_specific<unsigned int *>& m_pcf_array;
        unsigned int m_nbins_r;
        unsigned int m_nbins_T1;
        unsigned int m_nbins_T2;
        const trajectory::Box m_box;
        const float m_max_r;
        const float m_max_T1;
        const float m_max_T2;
        const float m_dr;
        const float m_dT1;
        const float m_dT2;
        const locality::LinkCell *m_lc;
        vec3<float> *m_ref_points;
        float *m_ref_orientations;
        const unsigned int m_Nref;
        vec3<float> *m_points;
        float *m_orientations;
        const unsigned int m_Np;
    public:
        ComputePMFTR12(tbb::enumerable_thread_specific<unsigned int *>& pcf_array,
                       unsigned int nbins_r,
                       unsigned int nbins_T1,
                       unsigned int nbins_T2,
                       const trajectory::Box &box,
                       const float max_r,
                       const float max_T1,
                       const float max_T2,
                       const float dr,
                       const float dT1,
                       const float dT2,
                       const locality::LinkCell *lc,
                       vec3<float> *ref_points,
                       float *ref_orientations,
                       unsigned int Nref,
                       vec3<float> *points,
                       float *orientations,
                       unsigned int Np)
            : m_pcf_array(pcf_array), m_nbins_r(nbins_r), m_nbins_T1(nbins_T1), m_nbins_T2(nbins_T2), m_box(box),
              m_max_r(max_r), m_max_T1(max_T1), m_max_T2(max_T2), m_dr(dr), m_dT1(dT1), m_dT2(dT2), m_lc(lc),
              m_ref_points(ref_points), m_ref_orientations(ref_orientations), m_Nref(Nref), m_points(points),
              m_orientations(orientations), m_Np(Np)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            assert(m_ref_points);
            assert(m_points);
            assert(m_Nref > 0);
            assert(m_Np > 0);

            float dr_inv = 1.0f / m_dr;
            float maxrsq = m_max_r * m_max_r;
            float dT1_inv = 1.0f / m_dT1;
            float dT2_inv = 1.0f / m_dT2;

            Index3D b_i = Index3D(m_nbins_T1, m_nbins_T2, m_nbins_r);

            bool exists;
            m_pcf_array.local(exists);
            if (! exists)
                {
                m_pcf_array.local() = new unsigned int [m_nbins_r*m_nbins_T1*m_nbins_T2];
                memset((void*)m_pcf_array.local(), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_T1*m_nbins_T2);
                }

            // for each reference point
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                // get the cell the point is in
                vec3<float> ref = m_ref_points[i];
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
                        vec3<float> delta = m_box.wrap(m_points[j] - ref);

                        float rsq = dot(delta, delta);
                        if (rsq < 1e-6)
                            {
                            continue;
                            }
                        if (rsq < maxrsq)
                            {
                            float r = sqrtf(rsq);
                            // calculate angles
                            float dTheta1 = atan2(delta.y, delta.x);
                            float dTheta2 = atan2(-delta.y, -delta.x);
                            float T1 = m_ref_orientations[i] - dTheta1;
                            float T2 = m_orientations[j] - dTheta2;
                            // make sure that T1, T2 are bounded between 0 and 2PI
                            T1 = (T1 < 0) ? T1+2*M_PI : T1;
                            T1 = (T1 > 2*M_PI) ? T1-2*M_PI : T1;
                            T2 = (T2 < 0) ? T2+2*M_PI : T2;
                            T2 = (T2 > 2*M_PI) ? T2-2*M_PI : T2;
                            // bin that point
                            float binr = r * dr_inv;
                            float binT1 = floorf(T1 * dT1_inv);
                            float binT2 = floorf(T2 * dT2_inv);
                            // fast float to int conversion with truncation
                            #ifdef __SSE2__
                            unsigned int ibinr = _mm_cvtt_ss2si(_mm_load_ss(&binr));
                            unsigned int ibinT1 = _mm_cvtt_ss2si(_mm_load_ss(&binT1));
                            unsigned int ibinT2 = _mm_cvtt_ss2si(_mm_load_ss(&binT2));
                            #else
                            unsigned int ibinr = (unsigned int)(binr);
                            unsigned int ibinT1 = (unsigned int)(binT1);
                            unsigned int ibinT2 = (unsigned int)(binT2);
                            #endif

                            if ((ibinr < m_nbins_r) && (ibinT1 < m_nbins_T1) && (ibinT2 < m_nbins_T2))
                                {
                                ++m_pcf_array.local()[b_i(ibinT1, ibinT2, ibinr)];
                                }
                            }
                        }
                    }
                } // done looping over reference points
            }
    };

//! \internal
//! helper function to reduce the thread specific arrays into the boost array
void PMFTR12::reducePCF()
    {
    memset((void*)m_pcf_array.get(), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_T1*m_nbins_T2);
    parallel_for(blocked_range<size_t>(0,m_nbins_T1),
                 CombinePCFR12(m_nbins_r,
                               m_nbins_T1,
                               m_nbins_T2,
                               m_pcf_array.get(),
                               m_local_pcf_array));
    }

//! Get a reference to the PCF array
boost::shared_array<unsigned int> PMFTR12::getPCF()
    {
    reducePCF();
    return m_pcf_array;
    }

void PMFTR12::resetPCF()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_pcf_array.begin(); i != m_local_pcf_array.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_T1*m_nbins_T2);
        }
    }

void PMFTR12::accumulate(trajectory::Box& box,
                         vec3<float> *ref_points,
                         float *ref_orientations,
                         unsigned int Nref,
                         vec3<float> *points,
                         float *orientations,
                         unsigned int Np)
    {
    m_box = box;
    m_lc->computeCellList(m_box, points, Np);
    parallel_for(blocked_range<size_t>(0,Nref),
                 ComputePMFTR12(m_local_pcf_array,
                                m_nbins_r,
                                m_nbins_T1,
                                m_nbins_T2,
                                m_box,
                                m_max_r,
                                m_max_T1,
                                m_max_T2,
                                m_dr,
                                m_dT1,
                                m_dT2,
                                m_lc,
                                ref_points,
                                ref_orientations,
                                Nref,
                                points,
                                orientations,
                                Np));
    }

}; }; // end namespace freud::pmft
