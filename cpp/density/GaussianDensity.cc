#include "GaussianDensity.h"
#include "ScopedGILRelease.h"

#include <stdexcept>

using namespace std;
using namespace boost::python;

/*! \file GaussianDensity.cc
    \brief Routines for computing Gaussian smeared densities from points
*/

namespace freud { namespace density {


GaussianDensity::GaussianDensity(const trajectory::Box& box, unsigned int width, float r_cut, float sigma)
    : m_box(box), m_width_x(width), m_width_y(width), m_width_z(width), m_r_cut(r_cut), m_sigma(sigma)
    {
    if (width <= 0)
            throw invalid_argument("width must be a positive integer");
    if (r_cut <= 0.0f)
            throw invalid_argument("r_cut must be positive");

    // index proper
    if (m_box.is2D())
        m_bi = Index3D(m_width_x, m_width_y, 1);
    else
        m_bi = Index3D(m_width_x, m_width_y, m_width_z);

    m_Density_array = boost::shared_array<float>(new float[m_bi.getNumElements()]);
    memset((void*)m_Density_array.get(), 0, sizeof(float)*m_bi.getNumElements());
    }

GaussianDensity::GaussianDensity(const trajectory::Box& box, unsigned int width_x, unsigned int width_y,
                                 unsigned int width_z, float r_cut, float sigma)
    : m_box(box), m_width_x(width_x), m_width_y(width_y), m_width_z(width_z), m_r_cut(r_cut), m_sigma(sigma)
    {
    if (width_x <= 0 || width_y <=0 || width_z <=0)
            throw invalid_argument("width must be a positive integer");
    if (r_cut <= 0.0f)
            throw invalid_argument("r_cut must be positive");

    // index proper
    if (m_box.is2D())
        m_bi = Index3D(m_width_x, m_width_y, 1);
    else
        m_bi = Index3D(m_width_x, m_width_y, m_width_z);

    m_Density_array = boost::shared_array<float>(new float[m_bi.getNumElements()]);
    memset((void*)m_Density_array.get(), 0, sizeof(float)*m_bi.getNumElements());
    }

// void GaussianDensity::compute(const float3 *points, unsigned int Np)
void GaussianDensity::compute(const vec3<float> *points, unsigned int Np)
    {
    assert(points);
    assert(Np > 0);

    // reset the memory so multiple Densities can be computed in 1 script
    memset((void*)m_Density_array.get(), 0, sizeof(float)*m_bi.getNumElements());

    float lx = m_box.getLx();
    float ly = m_box.getLy();
    float lz = m_box.getLz();

    float grid_size_x = lx/m_width_x;
    float grid_size_y = ly/m_width_y;
    float grid_size_z = lz/m_width_z;

    // for each particle
    for (unsigned int particle = 0; particle < Np; particle++)
        {
        // find the distance of that particle to bins
        // will use this information to evaluate the Gaussian
        // Find the which bin the particle is in
        int bin_x = int((points[particle].x+lx/2.0f)/grid_size_x);
        int bin_y = int((points[particle].y+ly/2.0f)/grid_size_y);
        int bin_z = int((points[particle].z+lz/2.0f)/grid_size_z);

        // Find the number of bins within r_cut
        int bin_cut_x = int(m_r_cut/grid_size_x);
        int bin_cut_y = int(m_r_cut/grid_size_y);
        int bin_cut_z = int(m_r_cut/grid_size_z);

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
            float dz = float((grid_size_z*k + grid_size_z/2.0f) - points[particle].z - lz/2.0f);

            for (int j = bin_y - bin_cut_y; j <= bin_y + bin_cut_y; j++)
                {
                float dy = float((grid_size_y*j + grid_size_y/2.0f) - points[particle].y - ly/2.0f);

                for (int i = bin_x - bin_cut_x; i<= bin_x + bin_cut_x; i++)
                    {
                    // calculate the distance from the grid cell to particular particle
                    float dx = float((grid_size_x*i + grid_size_x/2.0f) - points[particle].x - lx/2.0f);
                    // float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                    vec3<float> delta = m_box.wrap(vec3<float>(dx, dy, dz));

                    // float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    float rsq = dot(delta, delta);
                    float rsqrt = sqrtf(rsq);

                    // check to see if this distance is within the specified r_cut
                    if (rsqrt < m_r_cut)
                        {
                        // evaluate the gaussian ...
                        // set up some constants first
                        float sigmasq = m_sigma*m_sigma;
                        float A = sqrt(1.0f/(2.0f*M_PI*sigmasq));

                        float x_gaussian = A*exp((-1.0f)*(delta.x*delta.x)/(2.0f*sigmasq));
                        float y_gaussian = A*exp((-1.0f)*(delta.y*delta.y)/(2.0f*sigmasq));
                        float z_gaussian = A*exp((-1.0f)*(delta.z*delta.z)/(2.0f*sigmasq));

                        // Assure that out of range indices are corrected for storage in the array
                        // i.e. bin -1 is actually bin 29 for nbins = 30
                        unsigned int ni = (i + m_width_x) % m_width_x;
                        unsigned int nj = (j + m_width_y) % m_width_y;
                        unsigned int nk = (k + m_width_z) % m_width_z;

                        // store the product of these values in an array - n[i, j, k] = gx*gy*gz
                        m_Density_array[m_bi(ni, nj, nk)] += x_gaussian*y_gaussian*z_gaussian;
                        }
                    }
                }
            }
        }
    }

void GaussianDensity::computePy(boost::python::numeric::array points)
    {
    // validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers
    // float3* points_raw = (float3*) num_util::data(points);
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(points_raw, Np);
        }
    }

void export_GaussianDensity()
    {
    class_<GaussianDensity>("GaussianDensity", init<trajectory::Box&, unsigned int, float, float>())
            .def(init<trajectory::Box&, unsigned int, unsigned int, unsigned int, float, float>()) 
            .def("getBox", &GaussianDensity::getBox, return_internal_reference<>())
            .def("compute", &GaussianDensity::computePy)
            .def("getGaussianDensity", &GaussianDensity::getDensityPy)
            ;
    }

}; }; // end namespace freud::density
