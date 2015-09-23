#include "GaussianDensity.h"
#include "ScopedGILRelease.h"

#include <stdexcept>

using namespace std;

using namespace tbb;

/*! \file GaussianDensity.cc
    \brief Routines for computing Gaussian smeared densities from points
*/

namespace freud { namespace density {


GaussianDensity::GaussianDensity(unsigned int width, float r_cut, float sigma)
    : m_box(trajectory::Box()), m_width_x(width), m_width_y(width), m_width_z(width), m_rcut(r_cut), m_sigma(sigma),
      m_frame_counter(0)
    {
    if (width <= 0)
            throw invalid_argument("width must be a positive integer");
    if (r_cut <= 0.0f)
            throw invalid_argument("r_cut must be positive");
    }

GaussianDensity::GaussianDensity(unsigned int width_x, unsigned int width_y,
                                 unsigned int width_z, float r_cut, float sigma)
    : m_box(trajectory::Box()), m_width_x(width_x), m_width_y(width_y), m_width_z(width_z), m_rcut(r_cut), m_sigma(sigma)
    {
    if (width_x <= 0 || width_y <=0 || width_z <=0)
            throw invalid_argument("width must be a positive integer");
    if (r_cut <= 0.0f)
            throw invalid_argument("r_cut must be positive");
    }

GaussianDensity::~GaussianDensity()
    {
    for (tbb::enumerable_thread_specific<float *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        delete[] (*i);
        }
    }

class CombineGaussianArrays
    {
    private:
        float *m_bin_counts;
        tbb::enumerable_thread_specific<float *>& m_local_bin_counts;
    public:
        CombineGaussianArrays(float *bin_counts,
                              tbb::enumerable_thread_specific<float *>& local_bin_counts)
            : m_bin_counts(bin_counts), m_local_bin_counts(local_bin_counts)
        {
        }
        void operator()( const blocked_range<size_t> &myBin ) const
            {
            for (size_t i = myBin.begin(); i != myBin.end(); i++)
                {
                for (tbb::enumerable_thread_specific<float *>::const_iterator local_bins = m_local_bin_counts.begin();
                     local_bins != m_local_bin_counts.end(); ++local_bins)
                    {
                    m_bin_counts[i] += (*local_bins)[i];
                    }
                }
            }
    };

class ComputeGaussianDensity
    {
    private:
        Index3D m_bi;
        tbb::enumerable_thread_specific<float *>& m_bin_counts;
        const trajectory::Box m_box;
        const float m_rcut;
        const float m_sigma;
        const unsigned int m_width_x;
        const unsigned int m_width_y;
        const unsigned int m_width_z;
        const vec3<float> *m_points;
        const unsigned int m_Np;
    public:
        ComputeGaussianDensity(Index3D bi,
                               tbb::enumerable_thread_specific<float *>& bin_counts,
                               const trajectory::Box &box,
                               const float rcut,
                               const float sigma,
                               const unsigned int width_x,
                               const unsigned int width_y,
                               const unsigned int width_z,
                               const vec3<float> *points,
                               const unsigned int Np)
            : m_bi(bi), m_bin_counts(bin_counts), m_box(box), m_rcut(rcut), m_sigma(sigma),
              m_width_x(width_x), m_width_y(width_y), m_width_z(width_z), m_points(points), m_Np(Np)
        {
        }
        void operator()( const blocked_range<size_t> &myR ) const
            {
            assert(m_points);
            assert(m_Np > 0);

            bool exists;
            m_bin_counts.local(exists);
            if (! exists)
                {
                m_bin_counts.local() = new float [m_bi.getNumElements()];
                memset((void*)m_bin_counts.local(), 0, sizeof(float)*m_bi.getNumElements());
                }

            // set up some constants first
            float lx = m_box.getLx();
            float ly = m_box.getLy();
            float lz = m_box.getLz();

            float grid_size_x = lx/m_width_x;
            float grid_size_y = ly/m_width_y;
            float grid_size_z = lz/m_width_z;

            float sigmasq = m_sigma*m_sigma;
            float A = sqrt(1.0f/(2.0f*M_PI*sigmasq));

            // for each reference point
            for (size_t idx = myR.begin(); idx != myR.end(); idx++)
                {
                // find the distance of that particle to bins
                // will use this information to evaluate the Gaussian
                // Find the which bin the particle is in
                int bin_x = int((m_points[idx].x+lx/2.0f)/grid_size_x);
                int bin_y = int((m_points[idx].y+ly/2.0f)/grid_size_y);
                int bin_z = int((m_points[idx].z+lz/2.0f)/grid_size_z);

                // Find the number of bins within r_cut
                int bin_cut_x = int(m_rcut/grid_size_x);
                int bin_cut_y = int(m_rcut/grid_size_y);
                int bin_cut_z = int(m_rcut/grid_size_z);

                // in 2D, only loop over the 0 z plane
                if (m_box.is2D())
                    {
                    bin_z = 0;
                    bin_cut_z = 0;
                    grid_size_z = 0;
                    }
                // Only evaluate over bins that are within the cut off to reduce the number of computations
                for (int k = bin_z - bin_cut_z; k <= bin_z + bin_cut_z; k++)
                    {
                    float dz = float((grid_size_z*k + grid_size_z/2.0f) - m_points[idx].z - lz/2.0f);

                    for (int j = bin_y - bin_cut_y; j <= bin_y + bin_cut_y; j++)
                        {
                        float dy = float((grid_size_y*j + grid_size_y/2.0f) - m_points[idx].y - ly/2.0f);

                        for (int i = bin_x - bin_cut_x; i<= bin_x + bin_cut_x; i++)
                            {
                            // calculate the distance from the grid cell to particular particle
                            float dx = float((grid_size_x*i + grid_size_x/2.0f) - m_points[idx].x - lx/2.0f);
                            vec3<float> delta = m_box.wrap(vec3<float>(dx, dy, dz));

                            float rsq = dot(delta, delta);
                            float rsqrt = sqrtf(rsq);

                            // check to see if this distance is within the specified r_cut
                            if (rsqrt < m_rcut)
                                {
                                // evaluate the gaussian ...

                                float x_gaussian = A*exp((-1.0f)*(delta.x*delta.x)/(2.0f*sigmasq));
                                float y_gaussian = A*exp((-1.0f)*(delta.y*delta.y)/(2.0f*sigmasq));
                                float z_gaussian = A*exp((-1.0f)*(delta.z*delta.z)/(2.0f*sigmasq));

                                // Assure that out of range indices are corrected for storage in the array
                                // i.e. bin -1 is actually bin 29 for nbins = 30
                                unsigned int ni = (i + m_width_x) % m_width_x;
                                unsigned int nj = (j + m_width_y) % m_width_y;
                                unsigned int nk = (k + m_width_z) % m_width_z;

                                // store the product of these values in an array - n[i, j, k] = gx*gy*gz
                                m_bin_counts.local()[m_bi(ni, nj, nk)] += x_gaussian*y_gaussian*z_gaussian;
                                }
                            }
                        }
                    }
                }
            }
    };

void GaussianDensity::reduceDensity()
    {
    memset((void*)m_Density_array.get(), 0, sizeof(float)*m_bi.getNumElements());
    // combine arrays
    parallel_for(blocked_range<size_t>(0,m_bi.getNumElements()), CombineGaussianArrays(m_Density_array.get(),
                                                                                       m_local_bin_counts));
    }

//!Get a reference to the last computed Density
boost::shared_array<float> GaussianDensity::getDensity()
    {
    reduceDensity();
    return m_Density_array;
    }

// //!Python wrapper for getDensity() (returns a copy)
// boost::python::numeric::array GaussianDensity::getDensityPy()
//     {
//     reduceDensity();
//     float *arr = m_Density_array.get();
//     std::vector<intp> dims;
//     if (!m_box.is2D())
//         dims.push_back(m_width_z);
//     dims.push_back(m_width_y);
//     dims.push_back(m_width_x);

//     return num_util::makeNum(arr, dims);
//     }

//! \internal
/*! \brief Function to reset the density array if needed e.g. calculating between new particle types
*/
void GaussianDensity::resetDensity()
    {
    for (tbb::enumerable_thread_specific<float *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(float)*m_bi.getNumElements());
        }
    // reset the frame counter
    m_frame_counter = 0;
    }

void GaussianDensity::accumulate(const trajectory::Box &box, const vec3<float> *points, unsigned int Np)
    {
    m_box = box;
    if (m_box.is2D())
        {
        m_bi = Index3D(m_width_x, m_width_y, 1);
        }
    else
        {
        m_bi = Index3D(m_width_x, m_width_y, m_width_z);
        }
    // this does not agree with rest of freud
    m_Density_array = boost::shared_array<float>(new float[m_bi.getNumElements()]);

    parallel_for(blocked_range<size_t>(0,Np), ComputeGaussianDensity(m_bi,
                                                                     m_local_bin_counts,
                                                                     m_box,
                                                                     m_rcut,
                                                                     m_sigma,
                                                                     m_width_x,
                                                                     m_width_y,
                                                                     m_width_z,
                                                                     points,
                                                                     Np));
    }

// void GaussianDensity::accumulatePy(trajectory::Box& box,
//                                    boost::python::numeric::array points)
//     {
//     // validate input type and rank
//     m_box = box;
//     if (m_box.is2D())
//         {
//         m_bi = Index3D(m_width_x, m_width_y, 1);
//         }
//     else
//         {
//         m_bi = Index3D(m_width_x, m_width_y, m_width_z);
//         }
//     // this does not agree with rest of freud
//     m_Density_array = boost::shared_array<float>(new float[m_bi.getNumElements()]);
//     num_util::check_type(points, NPY_FLOAT);
//     num_util::check_rank(points, 2);

//     // validate that the 2nd dimension is only 3
//     num_util::check_dim(points, 1, 3);
//     unsigned int Np = num_util::shape(points)[0];

//     // get the raw data pointers
//     vec3<float>* points_raw = (vec3<float>*) num_util::data(points);

//         // compute with the GIL released
//         {
//         util::ScopedGILRelease gil;
//         accumulate(points_raw, Np);
//         }
//     }

// void GaussianDensity::computePy(trajectory::Box& box,
//                                 boost::python::numeric::array points)
//     {
//     resetDensity();
//     accumulatePy(box, points);
//     }

// void export_GaussianDensity()
//     {
//     class_<GaussianDensity>("GaussianDensity", init<unsigned int, float, float>())
//             .def(init<unsigned int, unsigned int, unsigned int, float, float>())
//             .def("getBox", &GaussianDensity::getBox, return_internal_reference<>())
//             .def("accumulate", &GaussianDensity::accumulatePy)
//             .def("compute", &GaussianDensity::computePy)
//             .def("getGaussianDensity", &GaussianDensity::getDensityPy)
//             .def("resetDensity", &GaussianDensity::resetDensityPy)
//             ;
//     }

}; }; // end namespace freud::density
