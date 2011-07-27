#include "GaussianDensity.h"

#include <stdexcept>
#include <emmintrin.h>

using namespace std;
using namespace boost::python;

namespace freud { namespace density {

GaussianDensity::GaussianDensity(const trajectory::Box& box, unsigned int nbins, float r_cut, float sigma)
		: m_box(box), m_nbins(nbins), m_r_cut(r_cut), m_sigma(sigma)
		{
		if (nbins < 0)
				throw invalid_argument("grid_size must be a positive integer");
		if (r_cut < 0.0f)
				throw invalid_argument("r_cut must be positive");
		
		unsigned int binscube = m_nbins*m_nbins*m_nbins;

		assert(m_nbins > 0);
		m_Density_array = boost::shared_array<float>(new float[binscube]);
		memset((void*)m_Density_array.get(), 0, sizeof(float)*binscube);
		}

void GaussianDensity::compute(const float3 *points,
															unsigned int Np)
		{
		assert(points);
		assert(Np > 0);

		// reset the memory so multiple Densities can be computed in 1 script
		unsigned int binscube = m_nbins*m_nbins*m_nbins;
		memset((void*)m_Density_array.get(), 0, sizeof(float)*binscube);

		float lx = m_box.getLx();
		float ly = m_box.getLy();
		float lz = m_box.getLz();

		float grid_size_x = lx/m_nbins;
		float grid_size_y = ly/m_nbins;
		float grid_size_z = lz/m_nbins;

		// for each particle
		for (unsigned int particle = 0; particle < Np; particle++)
				{
				// find the distance of that particle to bins
				// will use this information to evaluate the Gaussian
				// Find the which bin the particle is in
				int bin_x = int(points[particle].x/grid_size_x);
				int bin_y = int(points[particle].y/grid_size_y);
				int bin_z = int(points[particle].z/grid_size_z);

				// Find the number of bins within r_cut
				int bin_cut_x = int(m_r_cut/grid_size_x);
				int bin_cut_y = int(m_r_cut/grid_size_y);
				int bin_cut_z = int(m_r_cut/grid_size_z);

				// Only evaluate over bins that are within the cut off to reduce the number of computations
				for (int i = bin_x - bin_cut_x; i<= bin_x + bin_cut_x; i++)
						{
						for (int j = bin_y - bin_cut_y; j <= bin_y + bin_cut_y; j++)
								{
								for (int k = bin_z - bin_cut_z; k <= bin_z + bin_cut_z; k++)
										{
										// calculate the distance from the grid cell to particular particle
										float dx = float(((grid_size_x)*i + (grid_size_x)/2.0f) - points[particle].x);
										float dy = float(((grid_size_y)*j + (grid_size_y)/2.0f) - points[particle].y);
										float dz = float(((grid_size_z)*k + (grid_size_z)/2.0f) - points[particle].z);
										float3 delta = m_box.wrap(make_float3(dx, dy, dz));

										float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
										float rsqrt = sqrtf(rsq);

										// check to see if this distance is within the specified r_cut
										if (rsqrt < m_r_cut)
												{
												  // evaluate the gaussian ...
													// set up some constants first
													float sigmasq = m_sigma*m_sigma;
													float A = (1.0f/(2.0f*M_PI*m_sigma));

													float x_gaussian = A*exp((-1.0f)*(delta.x*delta.x)/(2.0f*sigmasq));
													float y_gaussian = A*exp((-1.0f)*(delta.y*delta.y)/(2.0f*sigmasq));
													float z_gaussian = A*exp((-1.0f)*(delta.z*delta.z)/(2.0f*sigmasq));
												
													// Assure that out of range indices are corrected for storage in the array
													// i.e. bin -1 is actually bin 29 for nbins = 30
													unsigned int ni = i % m_nbins;
													unsigned int nj = j % m_nbins;
													unsigned int nk = k % m_nbins;

													// store the product of these values in an array - n[i, j, k] = gx*gy*gz
													m_Density_array[ni*m_nbins*m_nbins + nj*m_nbins + nk] += x_gaussian*y_gaussian*z_gaussian;
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

		// validate that the 2nd dimension is only 3 - ??? Do I need this
		num_util::check_dim(points, 1, 3);
		unsigned int Np = num_util::shape(points)[0];

		// get the raw data pointers
		float3* points_raw = (float3*) num_util::data(points);

		compute(points_raw, Np);
		}

void export_GaussianDensity()
		{
		class_<GaussianDensity>("GaussianDensity", init<trajectory::Box&, unsigned int, float, float>())
				.def("getBox", &GaussianDensity::getBox, return_internal_reference<>())
				.def("compute", &GaussianDensity::computePy)
				.def("getGaussianDensity", &GaussianDensity::getDensityPy)
				;
		}

}; }; // end namespace freud::density
