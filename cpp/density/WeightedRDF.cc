#include <complex>

#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace boost::python;
using namespace freud;

/*! \file WeightedRDF.cc
    \brief Weighted radial density functions
*/

template<typename T>
WeightedRDF<T>::WeightedRDF(const trajectory::Box& box, float rmax, float dr)
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
    m_rdf_array = boost::shared_array<T>(new T[m_nbins]);
    // memset((void*)m_rdf_array.get(), 0, sizeof(T)*m_nbins);
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

template<typename T>
WeightedRDF<T>::~WeightedRDF()
    {
    if(useCells())
    delete m_lc;
    }

template<typename T>
bool WeightedRDF<T>::useCells()
    {
    float l_min = fmin(m_box.getLx(), m_box.getLy());

    if (!m_box.is2D())
        l_min = fmin(l_min, m_box.getLz());

    if (m_rmax < l_min/3.0f)
        return true;

    return false;
    }

template<typename T>
// void WeightedRDF<T>::compute(const float3 *ref_points,
//                              const T *ref_values,
//                              unsigned int Nref,
//                              const float3 *points,
//                              const T *point_values,
//                              unsigned int Np)
void WeightedRDF<T>::compute(const vec3<float> *ref_points,
                             const T *ref_values,
                             unsigned int Nref,
                             const vec3<float> *points,
                             const T *point_values,
                             unsigned int Np)
    {
    // zero the bin counts for totaling
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    for(size_t i(0); i < m_nbins; ++i)
        m_rdf_array[i] = T();
    if (useCells())
        computeWithCellList(ref_points, ref_values, Nref, points, point_values, Np);
    else
        computeWithoutCellList(ref_points, ref_values, Nref, points, point_values, Np);
    }

template<typename T>
// void WeightedRDF<T>::computeWithoutCellList(const float3 *ref_points,
//                  const T *ref_values,
//                  unsigned int Nref,
//                  const float3 *points,
//                  const T *point_values,
//                  unsigned int Np)
void WeightedRDF<T>::computeWithoutCellList(const vec3<float> *ref_points,
                 const T *ref_values,
                 unsigned int Nref,
                 const vec3<float> *points,
                 const T *point_values,
                 unsigned int Np)
    {
    float dr_inv = 1.0f / m_dr;
    float rmaxsq = m_rmax * m_rmax;

    // for each reference point
    for (unsigned int i = 0; i < Nref; i++)
        {

        for (unsigned int j = 0; j < Np; j++)
            {
            // compute r between the two particles
            vec3<float> delta = ref_points[i] - points[j];
            // float dx = float(ref_points[i].x - points[j].x);
            // float dy = float(ref_points[i].y - points[j].y);
            // float dz = float(ref_points[i].z - points[j].z);

            delta = m_box.wrap(delta);

            // float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
            float rsq = dot(delta, delta);
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

                m_bin_counts[bin]++;
                m_rdf_array[bin] += ref_values[i]*point_values[j];
                }
            }
        } // done looping over reference points

    // now compute the rdf
    float ndens = float(Np) / m_box.getVolume();
    m_rdf_array[0] = T();

    for (unsigned int bin = 1; bin < m_nbins; bin++)
        {
        if (m_bin_counts[bin])
            {
            m_rdf_array[bin] /= m_bin_counts[bin];
            }
        }
    }

template<typename T>
// void WeightedRDF<T>::computeWithCellList(const float3 *ref_points,
//                   const T *ref_values,
//                   unsigned int Nref,
//                   const float3 *points,
//                   const T *point_values,
//                   unsigned int Np)
void WeightedRDF<T>::computeWithCellList(const vec3<float> *ref_points,
                  const T *ref_values,
                  unsigned int Nref,
                  const vec3<float> *points,
                  const T *point_values,
                  unsigned int Np)
    {
    assert(ref_points);
    assert(ref_values);
    assert(points);
    assert(point_values);
    assert(Nref > 0);
    assert(Np > 0);

    // bin the x,y,z particles
    m_lc->computeCellList(points, Np);

    float dr_inv = 1.0f / m_dr;
    float rmaxsq = m_rmax * m_rmax;

    // for each reference point
    for (unsigned int i = 0; i < Nref; i++)
        {
        // get the cell the point is in
        // float3 ref = ref_points[i];
        vec3<float> ref = ref_points[i];
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
                vec3<float> delta = ref - points[j];
                // float dx = float(ref.x - points[j].x);
                // float dy = float(ref.y - points[j].y);
                // float dz = float(ref.z - points[j].z);
                // float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                delta = m_box.wrap(delta);

                // float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                float rsq = dot(delta);

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

                    m_bin_counts[bin]++;
                    m_rdf_array[bin] += ref_values[i]*point_values[j];
                    }
                }
            }
        } // done looping over reference points

    // now compute the rdf
    float ndens = float(Np) / m_box.getVolume();
    m_rdf_array[0] = T();

    for (unsigned int bin = 1; bin < m_nbins; bin++)
        {
        if (m_bin_counts[bin])
            {
            m_rdf_array[bin] /= m_bin_counts[bin];
            }
        }
    }

template<typename T>
void WeightedRDF<T>::computePy(boost::python::numeric::array ref_points,
                            boost::python::numeric::array ref_values,
                            boost::python::numeric::array points,
                            boost::python::numeric::array point_values)
    {
    // validate input type and rank
    num_util::check_type(ref_points, PyArray_FLOAT);
    num_util::check_rank(ref_points, 2);
    num_util::check_rank(ref_values, 1);
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    num_util::check_rank(point_values, 1);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];
    assert(Np == num_util::shape(point_values)[0]);

    num_util::check_dim(ref_points, 1, 3);
    unsigned int Nref = num_util::shape(ref_points)[0];
    assert(Nref == num_util::shape(ref_values)[0]);

    // get the raw data pointers and compute the cell list
    // float3* ref_points_raw = (float3*) num_util::data(ref_points);
    vec3<float>* ref_points_raw = (vec3<float>*) num_util::data(ref_points);
    T* ref_values_raw = (T*) num_util::data(ref_values);
    // float3* points_raw = (float3*) num_util::data(points);
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
    T* point_values_raw = (T*) num_util::data(point_values);

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(ref_points_raw, ref_values_raw, Nref, points_raw, point_values_raw, Np);
        }
    }

void export_WeightedRDF()
    {
    typedef WeightedRDF<std::complex<float> > ComplexWRDF;
    class_<ComplexWRDF>("ComplexWRDF", init<trajectory::Box&, float, float>())
        .def("getBox", &ComplexWRDF::getBox, return_internal_reference<>())
        .def("compute", &ComplexWRDF::computePy)
        .def("getRDF", &ComplexWRDF::getRDFPy)
        .def("getCounts", &ComplexWRDF::getCountsPy)
        .def("getR", &ComplexWRDF::getRPy)
        ;
    typedef WeightedRDF<float> FloatWRDF;
    class_<FloatWRDF>("FloatWRDF", init<trajectory::Box&, float, float>())
        .def("getBox", &FloatWRDF::getBox, return_internal_reference<>())
        .def("compute", &FloatWRDF::computePy)
        .def("getRDF", &FloatWRDF::getRDFPy)
        .def("getCounts", &FloatWRDF::getCountsPy)
        .def("getR", &FloatWRDF::getRPy)
        ;
    }
