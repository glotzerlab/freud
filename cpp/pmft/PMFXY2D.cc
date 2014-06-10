#include "PMFXY2D.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <tbb/tbb.h>

#include "VectorMath.h"

using namespace std;
using namespace boost::python;

using namespace tbb;

/*! \file PMFXY2D.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace pmft {

PMFXY2D::PMFXY2D(const trajectory::Box& box, float max_x, float max_y, float dx, float dy)
    : m_box(box), m_max_x(max_x), m_max_y(max_y), m_dx(dx), m_dy(dy)
    {
    if (dx < 0.0f)
        throw invalid_argument("dx must be positive");
    if (dy < 0.0f)
        throw invalid_argument("dy must be positive");
    if (max_x < 0.0f)
        throw invalid_argument("max_x must be positive");
    if (max_y < 0.0f)
        throw invalid_argument("max_y must be positive");
    if (dx > max_x)
        throw invalid_argument("max_x must be greater than dx");
    if (dy > max_y)
        throw invalid_argument("max_y must be greater than dy");
    if (max_x > box.getLx()/2 || max_y > box.getLy()/2)
        throw invalid_argument("max_x, max_y must be smaller than half the smallest box size");
    if (!box.is2D())
        throw invalid_argument("box must be 2D");

    m_nbins_x = int(2 * floorf(m_max_x / m_dx));
    assert(m_nbins_x > 0);
    m_nbins_y = int(2 * floorf(m_max_y / m_dy));
    assert(m_nbins_y > 0);

    // precompute the bin center positions for x
    m_x_array = boost::shared_array<float>(new float[m_nbins_x]);
    for (unsigned int i = 0; i < m_nbins_x; i++)
        {
        float x = float(i) * m_dx;
        float nextx = float(i+1) * m_dx;
        m_x_array[i] = -m_max_x + ((x + nextx) / 2.0);
        }

    // precompute the bin center positions for y
    m_y_array = boost::shared_array<float>(new float[m_nbins_y]);
    for (unsigned int i = 0; i < m_nbins_y; i++)
        {
        float y = float(i) * m_dy;
        float nexty = float(i+1) * m_dy;
        m_y_array[i] = -m_max_y + ((y + nexty) / 2.0);
        }

    if (useCells())
        {
        // float max_val = fmax(max_x, max_y);
        float max_val = sqrtf(max_x*max_x + max_y*max_y);
        m_lc = new locality::LinkCell(box, max_val);
        }
    }

PMFXY2D::~PMFXY2D()
    {
    if(useCells())
    delete m_lc;
    }

class ComputePMFXY2DWithoutCellList
    {
    private:
        atomic<unsigned int> *m_pcf_array;
        unsigned int m_nbins_x;
        unsigned int m_nbins_y;
        const trajectory::Box m_box;
        const float m_max_x;
        const float m_max_y;
        const float m_dx;
        const float m_dy;
        const float3 *m_ref_points;
        float *m_ref_orientations;
        const unsigned int m_Nref;
        const float3 *m_points;
        float *m_orientations;
        const unsigned int m_Np;
    public:
        ComputePMFXY2DWithoutCellList(atomic<unsigned int> *pcf_array,
                                   unsigned int nbins_x,
                                   unsigned int nbins_y,
                                   const trajectory::Box &box,
                                   const float max_x,
                                   const float max_y,
                                   const float dx,
                                   const float dy,
                                   const float3 *ref_points,
                                   float *ref_orientations,
                                   unsigned int Nref,
                                   const float3 *points,
                                   float *orientations,
                                   unsigned int Np)
            : m_pcf_array(pcf_array), m_nbins_x(nbins_x), m_nbins_y(nbins_y), m_box(box),
              m_max_x(max_x), m_max_y(max_y), m_dx(dx), m_dy(dy), m_ref_points(ref_points), m_ref_orientations(orientations),
              m_Nref(Nref), m_points(points), m_orientations(orientations), m_Np(Np)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            float dx_inv = 1.0f / m_dx;
            float maxxsq = m_max_x * m_max_x;
            float dy_inv = 1.0f / m_dy;
            float maxysq = m_max_y * m_max_y;

            // for each reference point
            for (size_t i = myR.begin(); i != myR.end(); i++)
                {
                float3 ref = m_ref_points[i];
                for (unsigned int j = 0; j < m_Np; j++)
                    {
                    if (i == j)
                        {
                        continue;
                        }
                    float3 point = m_points[j];
                    // compute r between the two particles
                    float dx = float(point.x - ref.x);
                    float dy = float(point.y - ref.y);

                    float3 delta = m_box.wrap(make_float3(dx, dy, (float)0));

                    float xsq = delta.x*delta.x;
                    float ysq = delta.y*delta.y;
                    if ((xsq < maxxsq) && (ysq < maxysq))
                        {
                        // rotate interparticle vector
                        vec2<Scalar> myVec(delta.x, delta.y);
                        rotmat2<Scalar> myMat = rotmat2<Scalar>::fromAngle(-m_ref_orientations[i]);
                        vec2<Scalar> rotVec = myMat * myVec;
                        float x = rotVec.x + m_max_x;
                        float y = rotVec.y + m_max_y;

                        float binx = floorf(x * dx_inv);
                        float biny = floorf(y * dy_inv);
                        // fast float to int conversion with truncation
                        #ifdef __SSE2__
                        unsigned int ibinx = _mm_cvtt_ss2si(_mm_load_ss(&binx));
                        unsigned int ibiny = _mm_cvtt_ss2si(_mm_load_ss(&biny));
                        #else
                        unsigned int ibinx = (unsigned int)(binx);
                        unsigned int ibiny = (unsigned int)(biny);
                        #endif

                        if ((ibinx < m_nbins_x) && (ibiny < m_nbins_y))
                            {
                            m_pcf_array[ibiny*m_nbins_x + ibinx]++;
                            }
                        }
                    }
                } // done looping over reference points
            }
    };

class ComputePMFXY2DWithCellList
    {
    private:
        atomic<unsigned int> *m_pcf_array;
        unsigned int m_nbins_x;
        unsigned int m_nbins_y;
        const trajectory::Box m_box;
        const float m_max_x;
        const float m_max_y;
        const float m_dx;
        const float m_dy;
        const locality::LinkCell *m_lc;
        float3 *m_ref_points;
        float *m_ref_orientations;
        const unsigned int m_Nref;
        float3 *m_points;
        float *m_orientations;
        const unsigned int m_Np;
    public:
        ComputePMFXY2DWithCellList(atomic<unsigned int> *pcf_array,
                                unsigned int nbins_x,
                                unsigned int nbins_y,
                                const trajectory::Box &box,
                                const float max_x,
                                const float max_y,
                                const float dx,
                                const float dy,
                                const locality::LinkCell *lc,
                                float3 *ref_points,
                                float *ref_orientations,
                                unsigned int Nref,
                                float3 *points,
                                float *orientations,
                                unsigned int Np)
            : m_pcf_array(pcf_array), m_nbins_x(nbins_x), m_nbins_y(nbins_y), m_box(box),
              m_max_x(max_x), m_max_y(max_y), m_dx(dx), m_dy(dy), m_lc(lc),
              m_ref_points(ref_points), m_ref_orientations(orientations), m_Nref(Nref), m_points(points), m_orientations(orientations), m_Np(Np)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            assert(m_ref_points);
            assert(m_points);
            assert(m_Nref > 0);
            assert(m_Np > 0);

            float dx_inv = 1.0f / m_dx;
            float maxxsq = m_max_x * m_max_x;
            float dy_inv = 1.0f / m_dy;
            float maxysq = m_max_y * m_max_y;

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
                        // will skip same particle
                        if (i == j)
                            {
                            continue;
                            }
                        float3 point = m_points[j];
                        float dx = float(point.x - ref.x);
                        float dy = float(point.y - ref.y);
                        float3 delta = m_box.wrap(make_float3(dx, dy, (float)0));

                        float xsq = delta.x*delta.x;
                        float ysq = delta.y*delta.y;
                        // if ((xsq < maxxsq) && (ysq < maxysq))
                            {
                            // rotate interparticle vector
                            vec2<Scalar> myVec(delta.x, delta.y);
                            rotmat2<Scalar> myMat = rotmat2<Scalar>::fromAngle(-m_ref_orientations[i]);
                            vec2<Scalar> rotVec = myMat * myVec;
                            float x = rotVec.x + m_max_x;
                            float y = rotVec.y + m_max_y;

                            float binx = floorf(x * dx_inv);
                            float biny = floorf(y * dy_inv);
                            // fast float to int conversion with truncation
                            #ifdef __SSE2__
                            unsigned int ibinx = _mm_cvtt_ss2si(_mm_load_ss(&binx));
                            unsigned int ibiny = _mm_cvtt_ss2si(_mm_load_ss(&biny));
                            #else
                            unsigned int ibinx = (unsigned int)(binx);
                            unsigned int ibiny = (unsigned int)(biny);
                            #endif

                            if ((ibinx < m_nbins_x) && (ibiny < m_nbins_y))
                                {
                                m_pcf_array[ibiny*m_nbins_x + ibinx]++;
                                }
                            }
                        }
                    }
                } // done looping over reference points
            }
    };

bool PMFXY2D::useCells()
    {
    float l_min = fmin(m_box.getLx(), m_box.getLy());

    if (!m_box.is2D())
        l_min = fmin(l_min, m_box.getLz());

    float rmax = sqrtf(m_max_x*m_max_x + m_max_y*m_max_y);

    if (rmax < l_min/3.0f)
        return true;

    return false;
    }

void PMFXY2D::compute(unsigned int *pcf_array,
                        float3 *ref_points,
                        float *ref_orientations,
                        unsigned int Nref,
                        float3 *points,
                        float *orientations,
                        unsigned int Np)
    {
    if (useCells())
        {
        m_lc->computeCellList(points, Np);
        parallel_for(blocked_range<size_t>(0,Nref), ComputePMFXY2DWithCellList((atomic<unsigned int>*)pcf_array,
                                                                            m_nbins_x,
                                                                            m_nbins_y,
                                                                            m_box,
                                                                            m_max_x,
                                                                            m_max_y,
                                                                            m_dx,
                                                                            m_dy,
                                                                            m_lc,
                                                                            ref_points,
                                                                            ref_orientations,
                                                                            Nref,
                                                                            points,
                                                                            orientations,
                                                                            Np));
        }
    else
        {
        parallel_for(blocked_range<size_t>(0,Nref), ComputePMFXY2DWithoutCellList((atomic<unsigned int>*)pcf_array,
                                                                               m_nbins_x,
                                                                               m_nbins_y,
                                                                               m_box,
                                                                               m_max_x,
                                                                               m_max_y,
                                                                               m_dx,
                                                                               m_dy,
                                                                               ref_points,
                                                                               ref_orientations,
                                                                               Nref,
                                                                               points,
                                                                               orientations,
                                                                               Np));
        }
    }

void PMFXY2D::computePy(boost::python::numeric::array pcf_array,
                          boost::python::numeric::array ref_points,
                          boost::python::numeric::array ref_orientations,
                          boost::python::numeric::array points,
                          boost::python::numeric::array orientations)
    {
    // validate input type and rank
    num_util::check_type(pcf_array, PyArray_INT);
    num_util::check_rank(pcf_array, 2);
    num_util::check_type(ref_points, PyArray_FLOAT);
    num_util::check_rank(ref_points, 2);
    num_util::check_type(ref_orientations, PyArray_FLOAT);
    num_util::check_rank(ref_orientations, 1);
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    num_util::check_type(orientations, PyArray_FLOAT);
    num_util::check_rank(orientations, 1);

    // validate array dims
    num_util::check_dim(pcf_array, 0, m_nbins_y);
    num_util::check_dim(pcf_array, 1, m_nbins_x);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    num_util::check_dim(ref_points, 1, 3);
    unsigned int Nref = num_util::shape(ref_points)[0];

    // check the size of angles to be correct
    num_util::check_dim(ref_orientations, 0, Nref);
    // num_util::check_dim(ref_orientations, 0, 1);
    num_util::check_dim(orientations, 0, Np);

    // get the raw data pointers and compute the cell list
    unsigned int* pcf_array_raw = (unsigned int*) num_util::data(pcf_array);
    float3* ref_points_raw = (float3*) num_util::data(ref_points);
    float* ref_orientations_raw = (float*) num_util::data(ref_orientations);
    float3* points_raw = (float3*) num_util::data(points);
    float* orientations_raw = (float*) num_util::data(orientations);

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(pcf_array_raw,
                ref_points_raw,
                ref_orientations_raw,
                Nref,
                points_raw,
                orientations_raw,
                Np);
        }
    }

void export_PMFXY2D()
    {
    class_<PMFXY2D>("PMFXY2D", init<trajectory::Box&, float, float, float, float>())
        .def("getBox", &PMFXY2D::getBox, return_internal_reference<>())
        .def("compute", &PMFXY2D::computePy)
        .def("getX", &PMFXY2D::getXPy)
        .def("getY", &PMFXY2D::getYPy)
        ;
    }

}; }; // end namespace freud::pmft
