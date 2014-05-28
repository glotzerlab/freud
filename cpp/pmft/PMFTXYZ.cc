#include "PMFTXYZ.h"
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

/*! \file PMFTXYZ.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace pmft {

PMFTXYZ::PMFTXYZ(const trajectory::Box& box, float max_x, float max_y, float max_z, float dx, float dy, float dz)
    : m_box(box), m_max_x(max_x), m_max_y(max_y), m_max_z(max_z), m_dx(dx), m_dy(dy), m_dz(dz)
    {
    if (dx < 0.0f)
        throw invalid_argument("dx must be positive");
    if (dy < 0.0f)
        throw invalid_argument("dy must be positive");
    if (dz < 0.0f)
        throw invalid_argument("dz must be positive");
    if (max_x < 0.0f)
        throw invalid_argument("max_x must be positive");
    if (max_y < 0.0f)
        throw invalid_argument("max_y must be positive");
    if (max_z < 0.0f)
        throw invalid_argument("max_z must be positive");
    if (dx > max_x)
        throw invalid_argument("max_x must be greater than dx");
    if (dy > max_y)
        throw invalid_argument("max_y must be greater than dy");
    if (dz > max_y)
        throw invalid_argument("max_y must be greater than dz");
    if (max_x > box.getLx()/2 || max_y > box.getLy()/2)
        throw invalid_argument("max_x, max_y must be smaller than half the smallest box size");
    if (max_z > box.getLz()/2 && !box.is2D())
        throw invalid_argument("max_z must be smaller than half the smallest box size");

    m_nbins_x = int(floorf(2 * m_max_x / m_dx));
    assert(m_nbins_x > 0);
    m_nbins_y = int(floorf(2 * m_max_y / m_dy));
    assert(m_nbins_y > 0);
    m_nbins_z = int(floorf(2 * m_max_z / m_dz));
    assert(m_nbins_z > 0);
    // should this be a 1D array?
    m_pcf_array = boost::shared_array<unsigned int>(new unsigned int[m_nbins_x*m_nbins_y*m_nbins_z]);
    memset((void*)m_pcf_array.get(), 0, sizeof(unsigned int)*m_nbins_x*m_nbins_y*m_nbins_z);

    // precompute the bin center positions for x
    // what should this calc be?
    m_x_array = boost::shared_array<float>(new float[m_nbins_x]);
    for (unsigned int i = 0; i < m_nbins_x; i++)
        {
        float x = -m_max_x + float(i) * m_dx;
        m_x_array[i] = x;
        }

    // precompute the bin center positions for y
    // what should this calc be?
    m_y_array = boost::shared_array<float>(new float[m_nbins_y]);
    for (unsigned int i = 0; i < m_nbins_y; i++)
        {
        float y = -m_max_y + float(i) * m_dy;
        m_y_array[i] = y;
        }

    // precompute the bin center positions for x
    // what should this calc be?
    m_z_array = boost::shared_array<float>(new float[m_nbins_z]);
    for (unsigned int i = 0; i < m_nbins_z; i++)
        {
        float z = -m_max_z + float(i) * m_dz;
        m_z_array[i] = z;
        }

    if (useCells())
        {
        m_lc = new locality::LinkCell(box, sqrtf(max_x*max_x + max_y*max_y + max_z*max_z));
        }
    }

PMFTXYZ::~PMFTXYZ()
    {
    if(useCells())
    delete m_lc;
    }

class ComputePMFTWithoutCellList
    {
    private:
        unsigned int m_nbins_x;
        unsigned int m_nbins_y;
        unsigned int m_nbins_z;
        atomic<unsigned int> *m_pcf_array;
        const trajectory::Box m_box;
        const float m_max_x;
        const float m_max_y;
        const float m_max_z;
        const float m_dx;
        const float m_dy;
        const float m_dz;
        const float3 *m_ref_points;
        const unsigned int m_Nref;
        const float3 *m_points;
        const unsigned int m_Np;
    public:
        ComputePMFTWithoutCellList(unsigned int nbins_x,
                                  unsigned int nbins_y,
                                  unsigned int nbins_z,
                                  atomic<unsigned int> *pcf_array,
                                  const trajectory::Box &box,
                                  const float max_x,
                                  const float max_y,
                                  const float max_z,
                                  const float dx,
                                  const float dy,
                                  const float dz,
                                  const float3 *ref_points,
                                  unsigned int Nref,
                                  const float3 *points,
                                  unsigned int Np)
            : m_nbins_x(nbins_x), m_nbins_y(nbins_y), m_nbins_z(nbins_z), m_pcf_array(pcf_array), m_box(box),
              m_max_x(max_x), m_max_y(max_y), m_max_z(max_z), m_dx(dx), m_dy(dy), m_dz(dz), m_ref_points(ref_points),
              m_Nref(Nref), m_points(points), m_Np(Np)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            float dx_inv = 1.0f / m_dx;
            float maxxsq = m_max_x * m_max_x;
            float dy_inv = 1.0f / m_dy;
            float maxysq = m_max_y * m_max_y;
            float dz_inv = 1.0f / m_dz;
            float maxzsq = m_max_z * m_max_z;

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

                    float xsq = delta.x*delta.x;
                    float ysq = delta.y*delta.y;
                    float zsq = delta.z*delta.z;
                    if ((xsq < maxxsq) && (ysq < maxysq) && (zsq < maxzsq))
                        {
                        float x = sqrtf(xsq);
                        float y = sqrtf(ysq);
                        float z = sqrtf(zsq);

                        // bin that point
                        float binx = x * dx_inv;
                        float biny = y * dy_inv;
                        float binz = z * dz_inv;
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

                        if ((ibinx < m_nbins_x) && (ibiny < m_nbins_y) && (ibinz < m_nbins_z))
                            {
                            m_pcf_array[ibinx*m_nbins_y*m_nbins_z + ibiny*m_nbins_z + ibinz]++;
                            }
                        }
                    }
                } // done looping over reference points
            }
    };

// class ComputeRDFWithCellList
//     {
//     private:
//         unsigned int m_nbins;
//         atomic<float> *m_rdf_array;
//         atomic<unsigned int> *m_bin_counts;
//         atomic<float> *m_N_r_array;
//         atomic<float> *m_vol_array;
//         const trajectory::Box m_box;
//         const float m_rmax;
//         const float m_dr;
//         const locality::LinkCell *m_lc;
//         const float3 *m_ref_points;
//         const unsigned int m_Nref;
//         const float3 *m_points;
//         const unsigned int m_Np;
//     public:
//         ComputeRDFWithCellList(unsigned int nbins,
//                                atomic<float> *rdf_array,
//                                atomic<unsigned int> *bin_counts,
//                                atomic<float> *N_r_array,
//                                atomic<float> *vol_array,
//                                const trajectory::Box &box,
//                                const float rmax,
//                                const float dr,
//                                const locality::LinkCell *lc,
//                                const float3 *ref_points,
//                                unsigned int Nref,
//                                const float3 *points,
//                                unsigned int Np)
//             : m_nbins(nbins), m_rdf_array(rdf_array), m_bin_counts(bin_counts), m_N_r_array(N_r_array),
//               m_vol_array(vol_array), m_box(box), m_rmax(rmax), m_dr(dr), m_lc(lc), m_ref_points(ref_points),
//               m_Nref(Nref), m_points(points), m_Np(Np)
//         {
//         }
//         void operator()( const blocked_range<size_t> &myR ) const
//             {
//             assert(m_ref_points);
//             assert(m_points);
//             assert(m_Nref > 0);
//             assert(m_Np > 0);

//             float dr_inv = 1.0f / m_dr;
//             float rmaxsq = m_rmax * m_rmax;

//             // for each reference point
//             for (size_t i = myR.begin(); i != myR.end(); i++)
//                 {
//                 // get the cell the point is in
//                 float3 ref = m_ref_points[i];
//                 unsigned int ref_cell = m_lc->getCell(ref);

//                 // loop over all neighboring cells
//                 const std::vector<unsigned int>& neigh_cells = m_lc->getCellNeighbors(ref_cell);
//                 for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
//                     {
//                     unsigned int neigh_cell = neigh_cells[neigh_idx];

//                     // iterate over the particles in that cell
//                     locality::LinkCell::iteratorcell it = m_lc->itercell(neigh_cell);
//                     for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
//                         {
//                         // compute r between the two particles
//                         float3 point = m_points[j];
//                         float dx = float(ref.x - point.x);
//                         float dy = float(ref.y - point.y);
//                         float dz = float(ref.z - point.z);
//                         float3 delta = m_box.wrap(make_float3(dx, dy, dz));

//                         float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;

//                         if (rsq < rmaxsq)
//                             {
//                             float r = sqrtf(rsq);

//                             // bin that r
//                             float binr = r * dr_inv;
//                             // fast float to int conversion with truncation
//                             #ifdef __SSE2__
//                             unsigned int bin = _mm_cvtt_ss2si(_mm_load_ss(&binr));
//                             #else
//                             unsigned int bin = (unsigned int)(binr);
//                             #endif

//                             if (bin < m_nbins)
//                                 {
//                                 m_bin_counts[bin]++;
//                                 }
//                             }
//                         }
//                     }
//                 } // done looping over reference points
//             }
//     };

bool PMFTXYZ::useCells()
    {
    float l_min = fmin(m_box.getLx(), m_box.getLy());

    if (!m_box.is2D())
        l_min = fmin(l_min, m_box.getLz());

    float rmax = sqrtf(m_max_x*m_max_x + m_max_y*m_max_y + m_max_z*m_max_z);

    if (rmax < l_min/3.0f)
        return true;

    return false;
    }

void PMFTXYZ::compute(const float3 *ref_points,
                  unsigned int Nref,
                  const float3 *points,
                  unsigned int Np)
    {
    // memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
    memset((void*)m_pcf_array.get(), 0, sizeof(unsigned int)*m_nbins_x*m_nbins_y*m_nbins_z);
    // if (useCells())
    //     {
    //     m_lc->computeCellList(points, Np);
    //     parallel_for(blocked_range<size_t>(0,Nref), ComputeRDFWithCellList(m_nbins,
    //                                                                        (atomic<float>*)m_rdf_array.get(),
    //                                                                        (atomic<unsigned int>*)m_bin_counts.get(),
    //                                                                        (atomic<float>*)m_N_r_array.get(),
    //                                                                        (atomic<float>*)m_vol_array.get(),
    //                                                                        m_box,
    //                                                                        m_rmax,
    //                                                                        m_dr,
    //                                                                        m_lc,
    //                                                                        ref_points,
    //                                                                        Nref,
    //                                                                        points,
    //                                                                        Np));
    //     }
    // else
    //     {
    //     parallel_for(blocked_range<size_t>(0,Nref), ComputeRDFWithoutCellList(m_nbins,
    //                                                                           (atomic<float>*)m_rdf_array.get(),
    //                                                                           (atomic<unsigned int>*)m_bin_counts.get(),
    //                                                                           (atomic<float>*)m_N_r_array.get(),
    //                                                                           (atomic<float>*)m_vol_array.get(),
    //                                                                           m_box,
    //                                                                           m_rmax,
    //                                                                           m_dr,
    //                                                                           ref_points,
    //                                                                           Nref,
    //                                                                           points,
    //                                                                           Np));
    //     }

    parallel_for(blocked_range<size_t>(0,Nref), ComputePMFTWithoutCellList(m_nbins_x,
                                                                          m_nbins_y,
                                                                          m_nbins_z,
                                                                          (atomic<unsigned int>*)m_pcf_array.get(),
                                                                          m_box,
                                                                          m_max_x,
                                                                          m_max_y,
                                                                          m_max_z,
                                                                          m_dx,
                                                                          m_dy,
                                                                          m_dz,
                                                                          ref_points,
                                                                          Nref,
                                                                          points,
                                                                          Np));

    }

void PMFTXYZ::computePy(boost::python::numeric::array ref_points,
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

void export_PMFTXYZ()
    {
    class_<PMFTXYZ>("PMFTXYZ", init<trajectory::Box&, float, float, float, float, float, float>())
        .def("getBox", &PMFTXYZ::getBox, return_internal_reference<>())
        .def("compute", &PMFTXYZ::computePy)
        .def("getPCF", &PMFTXYZ::getPCFPy)
        .def("getX", &PMFTXYZ::getXPy)
        .def("getY", &PMFTXYZ::getYPy)
        .def("getZ", &PMFTXYZ::getZPy)
        ;
    }

}; }; // end namespace freud::density
