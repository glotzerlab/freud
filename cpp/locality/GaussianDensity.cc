#include "GaussianDensity.h"

#include <stdexcept>
#include <emmintrin.h>

using namespace std;
using namespace boost::python;

GaussianDensity::GaussianDensity(const Box& box, unsigned int nbins, float r_cut, float sigma)
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
				for (unsigned int i = 0; i < m_nbins; i++)
						{
						for (unsigned int j = 0; j < m_nbins; j++)
								{
								for (unsigned int k = 0; k < m_nbins; k++)
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
												
													// store the product of these values in an array - n[i, j, k] = gx*gy*gz
													m_Density_array[i*m_nbins*m_nbins + j*m_nbins + k] += x_gaussian*y_gaussian*z_gaussian;
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
		class_<GaussianDensity>("GaussianDensity", init<Box&, unsigned int, float, float>())
				.def("getBox", &GaussianDensity::getBox, return_internal_reference<>())
				.def("compute", &GaussianDensity::computePy)
				.def("getGaussianDensity", &GaussianDensity::getDensityPy)
				;
		}

