#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <tbb/tbb.h>

using namespace std;
using namespace freud;

using namespace tbb;

/*! \file CorrelationFunction.cc
    \brief Generic pairwise correlation functions
*/

template<typename T>
CorrelationFunction<T>::CorrelationFunction(float rmax, float dr)
    : m_box(box::Box()), m_rmax(rmax), m_dr(dr), m_frame_counter(0)
    {
    if (dr < 0.0f)
        throw invalid_argument("dr must be positive");
    if (rmax < 0.0f)
        throw invalid_argument("rmax must be positive");
    if (dr > rmax)
        throw invalid_argument("rmax must be greater than dr");

    m_nbins = int(floorf(m_rmax / m_dr));
    assert(m_nbins > 0);
    m_rdf_array = boost::shared_array<T>(new T[m_nbins]);
    // Less efficient: initialize each bin sequentially using default ctor
    for(size_t i(0); i < m_nbins; ++i)
        m_rdf_array[i] = T();
    m_bin_counts = boost::shared_array<unsigned int>(new unsigned int[m_nbins]);
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);

    // precompute the bin center positions
    m_r_array = boost::shared_array<float>(new float[m_nbins]);
    for (unsigned int i = 0; i < m_nbins; i++)
        {
        float r = float(i) * m_dr;
        float nextr = float(i+1) * m_dr;
        m_r_array[i] = 2.0f / 3.0f * (nextr*nextr*nextr - r*r*r) / (nextr*nextr - r*r);
        }
    m_lc = new locality::LinkCell(m_box, m_rmax);
    }

template<typename T>
CorrelationFunction<T>::~CorrelationFunction()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        delete[] (*i);
        }
    for (typename tbb::enumerable_thread_specific<T *>::iterator i = m_local_rdf_array.begin(); i != m_local_rdf_array.end(); ++i)
        {
        delete[] (*i);
        }
    delete m_lc;
    }

template<typename T>
class CombineOCF
    {
    private:
        unsigned int m_nbins;
        unsigned int *m_bin_counts;
        tbb::enumerable_thread_specific<unsigned int *>& m_local_bin_counts;
        T *m_rdf_array;
        tbb::enumerable_thread_specific<T *>& m_local_rdf_array;
        float m_Nref;
    public:
        CombineOCF(unsigned int nbins,
                   unsigned int *bin_counts,
                   tbb::enumerable_thread_specific<unsigned int *>& local_bin_counts,
                   T *rdf_array,
                   tbb::enumerable_thread_specific<T *>& local_rdf_array,
                   float Nref)
            : m_nbins(nbins), m_bin_counts(bin_counts), m_local_bin_counts(local_bin_counts), m_rdf_array(rdf_array),
              m_local_rdf_array(local_rdf_array), m_Nref(Nref)
        {
        }
        void operator()( const tbb::blocked_range<size_t> &myBin ) const
            {
            for (size_t i = myBin.begin(); i != myBin.end(); i++)
                {
                for (tbb::enumerable_thread_specific<unsigned int *>::const_iterator local_bins = m_local_bin_counts.begin();
                     local_bins != m_local_bin_counts.end(); ++local_bins)
                    {
                    m_bin_counts[i] += (*local_bins)[i];
                    }
                for (typename tbb::enumerable_thread_specific<T *>::const_iterator local_rdf = m_local_rdf_array.begin();
                     local_rdf != m_local_rdf_array.end(); ++local_rdf)
                    {
                    m_rdf_array[i] += (*local_rdf)[i];
                    }
                if (m_bin_counts[i])
                    {
                    m_rdf_array[i] /= m_bin_counts[i];
                    }
                }
            }
    };

template<typename T>
class ComputeOCF
    {
    private:
        const unsigned int m_nbins;
        tbb::enumerable_thread_specific<unsigned int *>& m_bin_counts;
        tbb::enumerable_thread_specific<T *>& m_rdf_array;
        const box::Box m_box;
        const float m_rmax;
        const float m_dr;
        const locality::LinkCell *m_lc;
        const vec3<float> *m_ref_points;
        const T *m_ref_values;
        const unsigned int m_Nref;
        const vec3<float> *m_points;
        const T *m_point_values;
        unsigned int m_Np;
    public:
        ComputeOCF(const unsigned int nbins,
                   tbb::enumerable_thread_specific<unsigned int *>& bin_counts,
                   tbb::enumerable_thread_specific<T *>& rdf_array,
                   const box::Box &box,
                   const float rmax,
                   const float dr,
                   const locality::LinkCell *lc,
                   const vec3<float> *ref_points,
                   const T *ref_values,
                   unsigned int Nref,
                   const vec3<float> *points,
                   const T *point_values,
                   unsigned int Np)
            : m_nbins(nbins), m_bin_counts(bin_counts), m_rdf_array(rdf_array), m_box(box), m_rmax(rmax), m_dr(dr),
              m_lc(lc), m_ref_points(ref_points), m_ref_values(ref_values), m_Nref(Nref), m_points(points),
              m_point_values(point_values), m_Np(Np)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            assert(m_ref_points);
            assert(m_ref_values);
            assert(m_points);
            assert(m_point_values);
            assert(m_Nref > 0);
            assert(m_Np > 0);

            float dr_inv = 1.0f / m_dr;
            float rmaxsq = m_rmax * m_rmax;

            bool bin_exists;
            m_bin_counts.local(bin_exists);
            if (! bin_exists)
                {
                m_bin_counts.local() = new unsigned int [m_nbins];
                memset((void*)m_bin_counts.local(), 0, sizeof(unsigned int)*m_nbins);
                }

            bool rdf_exists;
            m_rdf_array.local(rdf_exists);
            if (! rdf_exists)
                {
                m_rdf_array.local() = new T [m_nbins];
                memset((void*)m_rdf_array.local(), 0, sizeof(T)*m_nbins);
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
                        // compute r between the two particles
                        vec3<float> delta = m_box.wrap(m_points[j] - ref);

                        float rsq = dot(delta, delta);

                        // check that the particle is not checking itself, if it is the same list
                        if ((i != j || m_points != m_ref_points) && rsq < rmaxsq)
                            {
                            float r = sqrtf(rsq);

                            // bin that r
                            float binr = r * dr_inv;
                            // fast float to int conversion with truncation
                            #ifdef __SSE2__
                            unsigned int bin = _mm_cvtt_ss2si(_mm_load_ss(&binr));
                            #else
                            unsigned int bin = (unsigned int)(binr);
                            #endif

                            if (bin < m_nbins)
                                {
                                ++m_bin_counts.local()[bin];
                                m_rdf_array.local()[bin] += m_ref_values[i]*m_point_values[j];
                                }
                            }
                        }
                    }
                } // done looping over reference points
            }
    };

//! \internal
//! helper function to reduce the thread specific arrays into the boost array
template<typename T>
void CorrelationFunction<T>::reduceCorrelationFunction()
    {
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    for(size_t i(0); i < m_nbins; ++i)
        m_rdf_array[i] = T();
    // now compute the rdf
    parallel_for(tbb::blocked_range<size_t>(0,m_nbins), CombineOCF<T>(m_nbins,
                                                              m_bin_counts.get(),
                                                              m_local_bin_counts,
                                                              m_rdf_array.get(),
                                                              m_local_rdf_array,
                                                              (float)m_Nref));
    }

//! Get a reference to the RDF array
template<typename T>
boost::shared_array<T> CorrelationFunction<T>::getRDF()
    {
    reduceCorrelationFunction();
    return m_rdf_array;
    }

//! \internal
/*! \brief Function to reset the pcf array if needed e.g. calculating between new particle types
*/
template<typename T>
void CorrelationFunction<T>::resetCorrelationFunction()
    {
    // zero the bin counts for totaling
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_nbins);
        }
    for (typename tbb::enumerable_thread_specific<T *>::iterator i = m_local_rdf_array.begin(); i != m_local_rdf_array.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(T)*m_nbins);
        }
    // reset the frame counter
    m_frame_counter = 0;
    }

template<typename T>
void CorrelationFunction<T>::accumulate(const box::Box &box,
                             const vec3<float> *ref_points,
                             const T *ref_values,
                             unsigned int Nref,
                             const vec3<float> *points,
                             const T *point_values,
                             unsigned int Np)
    {
    m_box = box;
    m_lc->computeCellList(m_box, points, Np);
    parallel_for(tbb::blocked_range<size_t>(0, Nref), ComputeOCF<T>(m_nbins,
                                                                    m_local_bin_counts,
                                                                    m_local_rdf_array,
                                                                    m_box,
                                                                    m_rmax,
                                                                    m_dr,
                                                                    m_lc,
                                                                    ref_points,
                                                                    ref_values,
                                                                    Nref,
                                                                    points,
                                                                    point_values,
                                                                    Np));
    m_frame_counter += 1;
    }

// // Default implementation: assume we're dealing with floats
// template<typename T>
// void checkCFType(boost::python::numeric::array values)
//     {
//     num_util::check_type(values, NPY_FLOAT);
//     }

// template<>
// void checkCFType<std::complex<float> >(boost::python::numeric::array values)
//     {
//     num_util::check_type(values, NPY_COMPLEX64);
//     }
