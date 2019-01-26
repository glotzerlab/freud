// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "RDF.h"

using namespace std;
using namespace tbb;

/*! \file RDF.cc
    \brief Routines for computing radial density functions.
*/

namespace freud { namespace density {

RDF::RDF(float rmax, float dr, float rmin)
    : m_box(box::Box()), m_rmax(rmax), m_rmin(rmin), m_dr(dr), m_frame_counter(0), m_reduce(true)
    {
    if (dr <= 0.0f)
        throw invalid_argument("RDF requires dr to be positive.");
    if (rmax <= 0.0f)
        throw invalid_argument("RDF requires rmax to be positive.");
    if (dr > rmax)
        throw invalid_argument("RDF requires dr must be less than or equal to rmax.");
    if (rmax <= rmin)
        throw invalid_argument("RDF requires that rmax must be greater than rmin.");
    if (rmax-rmin < dr)
        throw invalid_argument("RDF requires that the range (rmax-rmin) must be greater than dr.");

    m_nbins = int(floorf((m_rmax-m_rmin) / m_dr));
    assert(m_nbins > 0);
    m_rdf_array = std::shared_ptr<float>(new float[m_nbins], std::default_delete<float[]>());
    memset((void*)m_rdf_array.get(), 0, sizeof(float)*m_nbins);
    m_bin_counts = std::shared_ptr<unsigned int>(new unsigned int[m_nbins], std::default_delete<unsigned int[]>());
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    m_avg_counts = std::shared_ptr<float>(new float[m_nbins], std::default_delete<float[]>());
    memset((void*)m_avg_counts.get(), 0, sizeof(float)*m_nbins);
    m_N_r_array = std::shared_ptr<float>(new float[m_nbins], std::default_delete<float[]>());
    memset((void*)m_N_r_array.get(), 0, sizeof(unsigned int)*m_nbins);

    // precompute the bin center positions and cell volumes
    m_r_array = std::shared_ptr<float>(new float[m_nbins], std::default_delete<float[]>());
    m_vol_array = std::shared_ptr<float>(new float[m_nbins], std::default_delete<float[]>());
    memset((void*)m_vol_array.get(), 0, sizeof(float)*m_nbins);
    m_vol_array2D = std::shared_ptr<float>(new float[m_nbins], std::default_delete<float[]>());
    memset((void*)m_vol_array2D.get(), 0, sizeof(float)*m_nbins);
    m_vol_array3D = std::shared_ptr<float>(new float[m_nbins], std::default_delete<float[]>());
    memset((void*)m_vol_array3D.get(), 0, sizeof(float)*m_nbins);
    for (unsigned int i = 0; i < m_nbins; i++)
        {
        float r = float(i) * m_dr + m_rmin;
        float nextr = float(i+1) * m_dr + m_rmin;
        m_r_array.get()[i] = 2.0f / 3.0f * (nextr*nextr*nextr - r*r*r) / (nextr*nextr - r*r);
        m_vol_array2D.get()[i] = M_PI * (nextr*nextr - r*r);
        m_vol_array3D.get()[i] = 4.0f / 3.0f * M_PI * (nextr*nextr*nextr - r*r*r);
        }

    }  // end RDF::RDF

RDF::~RDF()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        delete[] (*i);
        }
    }

//! \internal
//! CumulativeCount class to perform a parallel reduce to get the cumulative count for each histogram bin
class CumulativeCount
    {
    private:
        float m_sum;
        float* m_N_r_array;
        float* m_avg_counts;
    public:
        CumulativeCount( float *N_r_array,
              float *avg_counts )
            : m_sum(0), m_N_r_array(N_r_array), m_avg_counts(avg_counts)
        {
        }
        float get_sum() const
            {
            return m_sum;
            }
        template<typename Tag>
        void operator()( const blocked_range<size_t>& r, Tag )
            {
            float temp = m_sum;
            for( size_t i=r.begin(); i<r.end(); i++ )
                {
                temp = temp + m_avg_counts[i];
                if( Tag::is_final_scan() )
                    m_N_r_array[i] = temp;
                }
            m_sum = temp;
            }
        CumulativeCount( CumulativeCount& b, split )
            : m_sum(0), m_N_r_array(b.m_N_r_array), m_avg_counts(b.m_avg_counts)
        {
        }
        void reverse_join( CumulativeCount& a )
            {
            m_sum = a.m_sum + m_sum;
            }
        void assign( CumulativeCount& b )
            {
            m_sum = b.m_sum;
            }
    };



//! \internal
//! helper function to reduce the thread specific arrays into one array
void RDF::reduceRDF()
    {
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    memset((void*)m_avg_counts.get(), 0, sizeof(float)*m_nbins);
    // now compute the rdf
    float ndens = float(m_Np) / m_box.getVolume();
    m_rdf_array.get()[0] = 0.0f;
    m_N_r_array.get()[0] = 0.0f;
    m_N_r_array.get()[1] = 0.0f;
    if (m_box.is2D())
        m_vol_array = m_vol_array2D;
    else
        m_vol_array = m_vol_array3D;
    // now compute the rdf
    parallel_for(blocked_range<size_t>(1,m_nbins),
            [=] (const blocked_range<size_t>& r)
        {
        for (size_t i = r.begin(); i != r.end(); i++)
            {
            for (tbb::enumerable_thread_specific<unsigned int *>::const_iterator \
                 local_bins = m_local_bin_counts.begin();
                 local_bins != m_local_bin_counts.end(); ++local_bins)
                {
                m_bin_counts.get()[i] += (*local_bins)[i];
                }
            m_avg_counts.get()[i] = (float)m_bin_counts.get()[i] / m_n_ref;
            m_rdf_array.get()[i] = m_avg_counts.get()[i] / m_vol_array.get()[i] / ndens;
            }
        });

    CumulativeCount myN_r(m_N_r_array.get(), m_avg_counts.get());
    parallel_scan( blocked_range<size_t>(0, m_nbins), myN_r);
    for (unsigned int i=0; i<m_nbins; i++)
        {
        m_rdf_array.get()[i] /= m_frame_counter;
        m_N_r_array.get()[i] /= m_frame_counter;
        }
    }

//! get a reference to the histogram bin centers array
std::shared_ptr<float> RDF::getR()
    {
    return m_r_array;
    }

//! Get a reference to the RDF histogram array
std::shared_ptr<float> RDF::getRDF()
    {
    if (m_reduce == true)
        {
        reduceRDF();
        }
    m_reduce = false;
    return m_rdf_array;
    }

//! Get a reference to the cumulative RDF histogram array
std::shared_ptr<float> RDF::getNr()
    {
    if (m_reduce == true)
        {
        reduceRDF();
        }
    m_reduce = false;
    return m_N_r_array;
    }

//! Get number of bins
unsigned int RDF::getNBins()
    {
    return m_nbins;
    }

//! \internal
/*! \brief Function to reset the rdf array if needed e.g. calculating between new particle types
*/
void RDF::reset()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_nbins);
        }
    // reset the frame counter
    m_frame_counter = 0;
    m_reduce = true;
    }

//! \internal
/*! \brief Function to accumulate the given points to the histogram in memory
*/
void RDF::accumulate(box::Box& box,
                     const locality::NeighborList *nlist,
                     const vec3<float> *ref_points,
                     unsigned int Nref,
                     const vec3<float> *points,
                     unsigned int Np)
    {
    m_box = box;
    m_Np = Np;
    m_n_ref = Nref;
    nlist->validate(Nref, Np);
    const size_t *neighbor_list(nlist->getNeighbors());
    parallel_for(blocked_range<size_t>(0, nlist->getNumBonds()),
            [=] (const blocked_range<size_t>& r)
        {
        assert(ref_points);
        assert(points);
        assert(m_n_ref > 0);
        assert(Np > 0);

        float dr_inv = 1.0f / m_dr;
        float rminsq = m_rmin * m_rmin;
        float rmaxsq = m_rmax * m_rmax;

        bool exists;
        m_local_bin_counts.local(exists);
        if (!exists)
            {
            m_local_bin_counts.local() = new unsigned int [m_nbins];
            memset((void*) m_local_bin_counts.local(), 0,
                   sizeof(unsigned int)*m_nbins);
            }

        size_t last_i(-1);
        vec3<float> ref;

        // for each bond
        for (size_t bond = r.begin(); bond != r.end(); bond++)
            {
            size_t i(neighbor_list[2*bond]);

            if (i != last_i)
                {
                ref = ref_points[i];
                }
            last_i = i;

            const unsigned int j(neighbor_list[2*bond + 1]);
            // compute r between the two particles
            vec3<float> delta = m_box.wrap(points[j] - ref);

            float rsq = dot(delta, delta);

            if (rsq < rmaxsq && rsq > rminsq)
                {
                float r = sqrtf(rsq);

                // bin that r
                float binr = (r - m_rmin) * dr_inv;
                // fast float to int conversion with truncation
#ifdef __SSE2__
                unsigned int bin = _mm_cvtt_ss2si(_mm_load_ss(&binr));
#else
                unsigned int bin = (unsigned int)(binr);
#endif

                // There may be a case where rsq < rmaxsq but
                // (r - m_rmin) * dr_inv rounds up to m_nbins.
                // This additional check prevents a seg fault.
                if (bin < m_nbins)
                    {
                    ++m_local_bin_counts.local()[bin];
                    }
                }
            } // done looping over bonds
        });
    m_frame_counter += 1;
    m_reduce = true;
    }
}; }; // end namespace freud::density
