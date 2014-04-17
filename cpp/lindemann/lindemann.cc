#include "lindemann.h"
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

/*! \file lind.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace lindemann {

Lind::Lind(const trajectory::Box& box, float rmax, float dr)
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
    }

class ComputeLindex
    {
    private:
        float *m_lindex_array;
        const trajectory::Box m_box;
        const float m_rmax;
        const float m_dr;
        const float3 *m_points;
        const unsigned int m_Np;
        const unsigned int m_Nf;
    public:
        ComputeLindex(float *lindex_array,
                        const trajectory::Box& box,
                        const float rmax,
                        const float dr,
                        const float3 *points,
                        const unsigned int Np,
                        const unsigned int Nf)
            : m_box(box), m_rmax(rmax), m_dr(dr), m_lindex_array(lindex_array), m_points(points), m_Np(Np), m_Nf(Nf)
            {
            }
        void operator()( const blocked_range<size_t>& r ) const
            {
            // zero the bin counts for totaling
            // memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins);
            float dr_inv = 1.0f / m_dr;
            float rmaxsq = m_rmax * m_rmax;

            // for each reference point
            float lindex;
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                lindex = 0;
                m_lindex_array[i] = 0;
                for (unsigned int j = 0; j < m_Np; j++)
                    {
                    // avoid calling on same particle
                    if (i == j)
                        {
                        continue;
                        }
                    float3 r_ij;
                    double rsq_ij = 0;
                    for (unsigned int k = 0; k < m_Nf; k++)
                        {
                        // compute r between the two particles
                        float dx = float(m_points[k * m_Np + i].x - m_points[k * m_Np + j].x);
                        float dy = float(m_points[k * m_Np + i].y - m_points[k * m_Np + j].y);
                        float dz = float(m_points[k * m_Np + i].z - m_points[k * m_Np + j].z);

                        float3 delta = m_box.wrap(make_float3(dx, dy, dz));

                        float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                        float r = sqrtf(rsq);
                        r_ij.x += delta.x;
                        r_ij.y += delta.y;
                        r_ij.z += delta.z;
                        rsq_ij += rsq;
                        } // done looping over frames
                    r_ij.x /= (double) m_Nf;
                    r_ij.y /= (double) m_Nf;
                    r_ij.z /= (double) m_Nf;
                    double avg_r_ij = sqrt(r_ij.x*r_ij.x  + r_ij.y*r_ij.y + r_ij.z*r_ij.z);
                    if (avg_r_ij < 0.0)
                        {
                        printf("prepare to die mortal scum; avg_r_ij = %f", avg_r_ij);
                        }
                    double avg_rsq_ij = rsq_ij / ((float) m_Nf);
                    double tmp_lindex = (sqrtf(abs(avg_rsq_ij - (avg_r_ij * avg_r_ij))) / avg_r_ij);
                    // printf("r_ij = %f\n", avg_r_ij);
                    // printf("diff = %f \n", diff);
                    // printf("rsq_ij = %f\n", rsq_ij);
                    // printf("avg_r_ij = %f\n", avg_r_ij);
                    // printf("avg_rsq_ij = %f\n", avg_rsq_ij);
                    // printf("inner sqrtf = %f\n", sqrtf(avg_rsq_ij - (avg_r_ij * avg_r_ij)) / avg_r_ij);
                    // printf("inner sqrtf = %f\n", tmp_lindex);
                    // if (abs(diff) > 0.00001)
                        // {
                        // printf("diff = %f \n", diff);
                        // }
                    // printf("tmp_lindex = %f \n", tmp_lindex);
                    if (tmp_lindex < 0.0)
                    {
                        printf("prepare to die mortal scum; tmp_lindex = %f", tmp_lindex);
                    }
                    lindex += tmp_lindex;
                    }
                lindex = (1.0 / (((float) m_Np) - 1.0)) * lindex;
                m_lindex_array[i] += lindex;
                } // done looping over reference points
            }
    };

void Lind::compute(const float3 *points,
                    unsigned int Np,
                    unsigned int Nf)
    {
    parallel_for(blocked_range<size_t>(0,Np), ComputeLindex(m_lindex_array.get(),
                                                            m_box,
                                                            m_rmax,
                                                            m_dr,
                                                            points,
                                                            Np,
                                                            Nf));
    }

void Lind::computePy(boost::python::numeric::array points)
    {
    // validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 3);

    // validate that the 3rd dimension is only 3
    num_util::check_dim(points, 2, 3);
    // Nf = number of frames
    // Np = number of particles
    unsigned int Nf = num_util::shape(points)[0];
    unsigned int Np = num_util::shape(points)[1];
    m_Np = Np;

    // validate that the length of the lindex array is the number of particles x 1
    m_lindex_array = boost::shared_array<float>(new float[Np]);

    // get the raw data pointers and compute the cell list
    float3* points_raw = (float3*) num_util::data(points);

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(points_raw, Np, Nf);
        }
    }

void export_lindemann()
    {
    class_<Lind>("Lind", init<trajectory::Box&, float, float>())
        .def("getBox", &Lind::getBox, return_internal_reference<>())
        .def("compute", &Lind::computePy)
        .def("getLindexArray", &Lind::getLindexArrayPy)
        ;
    }

}; }; // end namespace freud::lindemann
