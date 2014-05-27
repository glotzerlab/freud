#include "RDF.h"
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
using namespace boost::python;

using namespace tbb;

/*! \file RDF.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace density {

RDF::RDF(const trajectory::Box& box, float rmax, float dr)
    : m_box(box), m_rmax(rmax), m_dr(dr)
    {
    if (dr < 0.0f)
        throw invalid_argument("dr must be positive");
    if (rmax < 0.0f)
        throw invalid_argument("rmax must be positive");
    if (dr > rmax)
        throw invalid_argument("rmax must be greater than dr");
    if (rmax > box.getLx()/2 || rmax > box.getLy()/2)
        throw invalid_argument("rmax must be smaller than half the smallest box size");
    if (rmax > box.getLz()/2 && !box.is2D())
        throw invalid_argument("rmax must be smaller than half the smallest box size");

    m_nbins = int(floorf(m_rmax / m_dr));
    assert(m_nbins > 0);
    m_rdf_array = boost::shared_array<float>(new float[m_nbins]);
    memset((void*)m_rdf_array.get(), 0, sizeof(float)*m_nbins);
    m_bin_counts = boost::shared_array<unsigned int>(new unsigned int[m_nbins]);
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    m_N_r_array = boost::shared_array<float>(new float[m_nbins]);
    memset((void*)m_N_r_array.get(), 0, sizeof(unsigned int)*m_nbins);

    // precompute the bin center positions
    m_r_array = boost::shared_array<float>(new float[m_nbins]);
    for (unsigned int i = 0; i < m_nbins; i++)
        {
        float r = float(i) * m_dr;
        float nextr = float(i+1) * m_dr;
        m_r_array[i] = 2.0f / 3.0f * (nextr*nextr*nextr - r*r*r) / (nextr*nextr - r*r);
        }

    // precompute cell volumes
    m_vol_array = boost::shared_array<float>(new float[m_nbins]);
    for (unsigned int i = 0; i < m_nbins; i++)
        {
        float r = float(i) * m_dr;
        float nextr = float(i+1) * m_dr;
        if (m_box.is2D())
            m_vol_array[i] = M_PI * (nextr*nextr - r*r);
        else
            m_vol_array[i] = 4.0f / 3.0f * M_PI * (nextr*nextr*nextr - r*r*r);
        }

    if (useCells())
        {
        m_lc = new locality::LinkCell(box, rmax);
        }
    }

RDF::~RDF()
    {
    if(useCells())
    delete m_lc;
    }

class ComputeRDFWithoutCellList
    {
    private:
        unsigned int m_nbins;
        atomic<float> *m_rdf_array;
        atomic<unsigned int> *m_bin_counts;
        atomic<float> *m_N_r_array;
        atomic<float> *m_vol_array;
        const trajectory::Box m_box;
        const float m_rmax;
        const float m_dr;
        const float3 *m_ref_points;
        const unsigned int m_Nref;
        const float3 *m_points;
        const unsigned int m_Np;
    public:
        ComputeRDFWithoutCellList(unsigned int nbins,
                                  atomic<float> *rdf_array,
                                  atomic<unsigned int> *bin_counts,
                                  atomic<float> *N_r_array,
                                  atomic<float> *vol_array,
                                  const trajectory::Box &box,
                                  const float rmax,
                                  const float dr,
                                  const float3 *ref_points,
                                  unsigned int Nref,
                                  const float3 *points,
                                  unsigned int Np)
            : m_nbins(nbins), m_rdf_array(rdf_array), m_bin_counts(bin_counts), m_N_r_array(N_r_array),
              m_vol_array(vol_array), m_box(box), m_rmax(rmax), m_dr(dr), m_ref_points(ref_points), m_Nref(Nref),
              m_points(points), m_Np(Np)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            float dr_inv = 1.0f / m_dr;
            float rmaxsq = m_rmax * m_rmax;

            // for each reference point
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                float3 ref = m_ref_points[i];
                for (unsigned int j = 0; j < m_Np; j++)
                    {
                    float3 point = m_points[j];
                    // compute r between the two particles
                    float dx = float(ref.x - point.x);
                    float dy = float(ref.y - point.y);
                    float dz = float(ref.z - point.z);

                    float3 delta = m_box.wrap(make_float3(dx, dy, dz));

                    float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    if (rsq < rmaxsq)
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
                            m_bin_counts[bin]++;
                            }
                        }
                    }
                } // done looping over reference points
            }
    };

class ComputeRDFWithCellList
    {
    private:
        unsigned int m_nbins;
        atomic<float> *m_rdf_array;
        atomic<unsigned int> *m_bin_counts;
        atomic<float> *m_N_r_array;
        atomic<float> *m_vol_array;
        const trajectory::Box m_box;
        const float m_rmax;
        const float m_dr;
        const locality::LinkCell *m_lc;
        const float3 *m_ref_points;
        const unsigned int m_Nref;
        const float3 *m_points;
        const unsigned int m_Np;
    public:
        ComputeRDFWithCellList(unsigned int nbins,
                               atomic<float> *rdf_array,
                               atomic<unsigned int> *bin_counts,
                               atomic<float> *N_r_array,
                               atomic<float> *vol_array,
                               const trajectory::Box &box,
                               const float rmax,
                               const float dr,
                               const locality::LinkCell *lc,
                               const float3 *ref_points,
                               unsigned int Nref,
                               const float3 *points,
                               unsigned int Np)
            : m_nbins(nbins), m_rdf_array(rdf_array), m_bin_counts(bin_counts), m_N_r_array(N_r_array),
              m_vol_array(vol_array), m_box(box), m_rmax(rmax), m_dr(dr), m_lc(lc), m_ref_points(ref_points),
              m_Nref(Nref), m_points(points), m_Np(Np)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            assert(m_ref_points);
            assert(m_points);
            assert(m_Nref > 0);
            assert(m_Np > 0);

            float dr_inv = 1.0f / m_dr;
            float rmaxsq = m_rmax * m_rmax;

            // for each reference point
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                // get the cell the point is in
                float3 ref = m_ref_points[i];
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
                        float3 point = m_points[j];
                        float dx = float(ref.x - point.x);
                        float dy = float(ref.y - point.y);
                        float dz = float(ref.z - point.z);
                        float3 delta = m_box.wrap(make_float3(dx, dy, dz));

                        float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;

                        if (rsq < rmaxsq)
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
                                m_bin_counts[bin]++;
                                }
                            }
                        }
                    }
                } // done looping over reference points
            }
    };

bool RDF::useCells()
    {
    float l_min = fmin(m_box.getLx(), m_box.getLy());

    if (!m_box.is2D())
        l_min = fmin(l_min, m_box.getLz());

    if (m_rmax < l_min/3.0f)
        return true;

    return false;
    }

void RDF::compute(const float3 *ref_points,
                  unsigned int Nref,
                  const float3 *points,
                  unsigned int Np)
    {
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    if (useCells())
        {
        m_lc->computeCellList(points, Np);
        parallel_for(blocked_range<size_t>(0,Nref), ComputeRDFWithCellList(m_nbins,
                                                                           (atomic<float>*)m_rdf_array.get(),
                                                                           (atomic<unsigned int>*)m_bin_counts.get(),
                                                                           (atomic<float>*)m_N_r_array.get(),
                                                                           (atomic<float>*)m_vol_array.get(),
                                                                           m_box,
                                                                           m_rmax,
                                                                           m_dr,
                                                                           m_lc,
                                                                           ref_points,
                                                                           Nref,
                                                                           points,
                                                                           Np));
        }
    else
        {
        parallel_for(blocked_range<size_t>(0,Nref), ComputeRDFWithoutCellList(m_nbins,
                                                                              (atomic<float>*)m_rdf_array.get(),
                                                                              (atomic<unsigned int>*)m_bin_counts.get(),
                                                                              (atomic<float>*)m_N_r_array.get(),
                                                                              (atomic<float>*)m_vol_array.get(),
                                                                              m_box,
                                                                              m_rmax,
                                                                              m_dr,
                                                                              ref_points,
                                                                              Nref,
                                                                              points,
                                                                              Np));
        }

    // now compute the rdf
    float ndens = float(Np) / m_box.getVolume();
    m_rdf_array[0] = 0.0f;
    m_N_r_array[0] = 0.0f;
    m_N_r_array[1] = 0.0f;

    for (unsigned int bin = 1; bin < m_nbins; bin++)
        {
        float avg_counts = m_bin_counts[bin] / float(Nref);
        m_rdf_array[bin] = avg_counts / m_vol_array[bin] / ndens;

        if (bin+1 < m_nbins)
            m_N_r_array[bin+1] = m_N_r_array[bin] + avg_counts;
        }
    }

void RDF::computePy(boost::python::numeric::array ref_points,
                    boost::python::numeric::array points)
    {
    // validate input type and rank
    num_util::check_type(ref_points, PyArray_FLOAT);
    num_util::check_rank(ref_points, 2);
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    num_util::check_dim(ref_points, 1, 3);
    unsigned int Nref = num_util::shape(ref_points)[0];

    // get the raw data pointers and compute the cell list
    float3* ref_points_raw = (float3*) num_util::data(ref_points);
    float3* points_raw = (float3*) num_util::data(points);

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(ref_points_raw, Nref, points_raw, Np);
        }
    }

void export_RDF()
    {
    class_<RDF>("RDF", init<trajectory::Box&, float, float>())
        .def("getBox", &RDF::getBox, return_internal_reference<>())
        .def("compute", &RDF::computePy)
        .def("getRDF", &RDF::getRDFPy)
        .def("getR", &RDF::getRPy)
        .def("getNr", &RDF::getNrPy)
        ;
    }

}; }; // end namespace freud::density
