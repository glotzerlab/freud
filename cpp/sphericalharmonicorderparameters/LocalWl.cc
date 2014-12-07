#include "LocalWl.h"
#include "wigner3j.h"
#include <stdexcept>
#include <complex>
#include <algorithm>
//#include <boost/math/special_functions.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

using namespace std;
using namespace boost::python;

/*! \file LocalWl.cc
    \brief Compute a Wl per particle.  Returns NaN if no neighbors.
*/

namespace freud { namespace sphericalharmonicorderparameters {

LocalWl::LocalWl(const trajectory::Box& box, float rmax, unsigned int l)
    :m_box(box), m_rmax(rmax), m_lc(box, rmax), m_l(l)
    {
    if (m_rmax < 0.0f)
        throw invalid_argument("rmax must be positive!");
    if (m_l < 2)
        throw invalid_argument("l must be two or greater (and even)!");
    if (m_l%2 == 1)
        {
        fprintf(stderr,"Current value of m_l is %d\n",m_l);
        throw invalid_argument("This method requires even values of l!");
        }
    m_normalizeWl = false;
    }

void LocalWl::Ylm(const double theta, const double phi, std::vector<std::complex<double> > &Y)
    {
    if(Y.size() != 2*m_l+1)
        Y.resize(2*m_l+1);

    for(int m = -m_l; m <=0; m++)
        //Doc for boost spherical harmonic
        //http://www.boost.org/doc/libs/1_53_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/sph_harm.html
        // theta = colatitude = 0..Pi
        // phi = azimuthal (longitudinal) 0..2pi).
        Y[m+m_l]= boost::math::spherical_harmonic(m_l, m, theta, phi);

    for(unsigned int i = 1; i <= m_l; i++)
        Y[i+m_l] = Y[-i+m_l];
    }

// void LocalWl::compute(const float3 *points, unsigned int Np)
void LocalWl::compute(const vec3<float> *points, unsigned int Np)
    {
    //Get wigner3j coefficients from wigner3j.cc
    int m_wignersize[10]={19,61,127,217,331,469,631,817,1027,1261};
    std::vector<double> m_wigner3jvalues (m_wignersize[m_l/2-1]);
    m_wigner3jvalues = getWigner3j(m_l);

    //Set local data size
    m_Np = Np;

    //Initialize cell list
    m_lc.computeCellList(m_box,points,m_Np);

    double rmaxsq = m_rmax * m_rmax;

    //newmanrs:  For efficiency, if Np != m_Np, we could not reallocate these! Maybe.
    // for safety and debugging laziness, reallocate each time
    m_Qlmi = boost::shared_array<complex<double> >(new complex<double> [(2*m_l+1)*m_Np]);
    m_Qli = boost::shared_array<double>(new double[m_Np]);
    m_Wli = boost::shared_array<complex<double> >(new complex<double>[m_Np]);
    m_Qlm = boost::shared_array<complex<double> >(new complex<double>[2*m_l+1]);
    memset((void*)m_Qlmi.get(), 0, sizeof(complex<double>)*(2*m_l+1)*m_Np);
    memset((void*)m_Wli.get(), 0, sizeof(complex<double>)*m_Np);
    memset((void*)m_Qlm.get(), 0, sizeof(complex<double>)*(2*m_l+1));
    memset((void*)m_Qli.get(), 0, sizeof(double)*m_Np);

    for (unsigned int i = 0; i<m_Np; i++)
        {
        //get cell point is in
        // float3 ref = points[i];
        vec3<float> ref = points[i];
        unsigned int ref_cell = m_lc.getCell(ref);
        unsigned int neighborcount=0;

        //loop over neighboring cells
        const std::vector<unsigned int>& neigh_cells = m_lc.getCellNeighbors(ref_cell);
        for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
            {
            unsigned int neigh_cell = neigh_cells[neigh_idx];

            //iterate over particles in neighboring cells
            locality::LinkCell::iteratorcell it = m_lc.itercell(neigh_cell);
            for (unsigned int j = it.next(); !it.atEnd(); j = it.next())
                {
                if (i == j)
                {
                    continue;
                }
                // rij = rj - ri, from i pointing to j.
                // float dx = float(points[j].x - ref.x);
                // float dy = float(points[j].y - ref.y);
                // float dz = float(points[j].z - ref.z);

                // float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                vec3<float> delta = m_box.wrap(points[j] - ref);
                // float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                float rsq = dot(delta, delta);

                if (rsq < rmaxsq)
                    {
                    double phi = atan2(delta.y,delta.x);      //0..2Pi
                    double theta = acos(delta.z / sqrt(rsq)); //0..Pi

                    std::vector<std::complex<double> > Y;
                    LocalWl::Ylm(theta, phi,Y);  //Fill up Ylm vector
                    for(unsigned int k = 0; k < (2*m_l+1); ++k)
                        {
                        // change to Index later
                        m_Qlmi[(2*m_l+1)*i+k]+=Y[k];
                        }
                    neighborcount++;
                    }
                }
            } //End loop going over neighbor cells (and thus all neighboring particles);
            //Normalize!
            for(unsigned int k = 0; k < (2*m_l+1); ++k)
                {
                m_Qlmi[(2*m_l+1)*i+k]/= neighborcount;
				m_Qli[i]+=abs( m_Qlmi[(2*m_l+1)*i+k]*conj(m_Qlmi[(2*m_l+1)*i+k]) );
                m_Qlm[k]+= m_Qlmi[(2*m_l+1)*i+k];
                } //Ends loop over particles i for Qlmi calcs
	    m_Qli[i]=sqrt(m_Qli[i]);//*sqrt(m_Qli[i])*sqrt(m_Qli[i]);//Normalize factor for Wli

        //Wli calculation
	    unsigned int counter = 0;
	    for(unsigned int u1 = 0; u1 < (2*m_l+1); ++u1)
	    	{
	    	for(unsigned int u2 = max( 0,int(m_l)-int(u1)); u2 < (min(3*m_l+1-u1,2*m_l+1)); ++u2)
	    		{
	    		unsigned int u3 = 3*m_l-u1-u2;
	    		m_Wli[i] += m_wigner3jvalues[counter]*m_Qlmi[(2*m_l+1)*i+u1]*m_Qlmi[(2*m_l+1)*i+u2]*m_Qlmi[(2*m_l+1)*i+u3];
                counter+=1;
	    		}
	    	}//Ends loop for Wli calcs
        if(m_normalizeWl)
            {
	        m_Wli[i]/=(m_Qli[i]*m_Qli[i]*m_Qli[i]);//Normalize
            }
	    m_counter = counter;
        }
    }

// void LocalWl::computeAve(const float3 *points, unsigned int Np)
void LocalWl::computeAve(const vec3<float> *points, unsigned int Np)
    {

    //Get wigner3j coefficients from wigner3j.cc
    int m_wignersize[10]={19,61,127,217,331,469,631,817,1027,1261};
    std::vector<double> m_wigner3jvalues (m_wignersize[m_l/2-1]);
    m_wigner3jvalues = getWigner3j(m_l);

    //Set local data size
    m_Np = Np;

    //Initialize cell list
    m_lc.computeCellList(m_box,points,m_Np);

    double rmaxsq = m_rmax * m_rmax;
    double normalizationfactor = 4*M_PI/(2*m_l+1);


    //newmanrs:  For efficiency, if Np != m_Np, we could not reallocate these! Maybe.
    // for safety and debugging laziness, reallocate each time
    m_AveQlmi = boost::shared_array<complex<double> >(new complex<double> [(2*m_l+1)*m_Np]);
    m_AveWli = boost::shared_array<complex<double> >(new complex<double> [m_Np]);
    memset((void*)m_AveQlmi.get(), 0, sizeof(complex<double>)*(2*m_l+1)*m_Np);
    memset((void*)m_AveWli.get(), 0, sizeof(double)*m_Np);

    for (unsigned int i = 0; i<m_Np; i++)
        {
        //get cell point is in
        // float3 ref = points[i];
        vec3<float> ref = points[i];
        unsigned int ref_cell = m_lc.getCell(ref);
        unsigned int neighborcount=1;

        //loop over neighboring cells
        const std::vector<unsigned int>& neigh_cells = m_lc.getCellNeighbors(ref_cell);
        for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
            {
            //get cell points of 1st neighbor
            unsigned int neigh_cell = neigh_cells[neigh_idx];

            //iterate over particles in neighboring cells
            locality::LinkCell::iteratorcell shell1 = m_lc.itercell(neigh_cell);
            for (unsigned int n1 = shell1.next(); !shell1.atEnd(); n1 = shell1.next())
                {
                // float3 ref1 = points[n1];
                vec3<float> ref1 = points[n1];
                unsigned int ref1_cell = m_lc.getCell(ref1);
                if (n1 == i)
                    {
                        continue;
                    }
                // rij = rj - ri, from i pointing to j.
                // float dx = float(points[n1].x - ref.x);
                // float dy = float(points[n1].y - ref.y);
                // float dz = float(points[n1].z - ref.z);

                // float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                vec3<float> delta = m_box.wrap(points[n1] - ref);
                // float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                float rsq = dot(delta, delta);
                if (rsq < rmaxsq)
                    {

                    //loop over 2nd neighboring cells
                    const std::vector<unsigned int>& neigh1_cells = m_lc.getCellNeighbors(ref1_cell);
                    for (unsigned int neigh1_idx = 0; neigh1_idx < neigh1_cells.size(); neigh1_idx++)
                        {
                        //get cell points of 2nd neighbor
                        unsigned int neigh1_cell = neigh1_cells[neigh1_idx];

                        //iterate over particles in neighboring cells
                        locality::LinkCell::iteratorcell it = m_lc.itercell(neigh1_cell);
                        for (unsigned int j = it.next(); !it.atEnd(); j = it.next())
                            {
                            if (n1 == j)
                                {
                                    continue;
                                }
                            // rij = rj - ri, from i pointing to j.
                            // float dx1 = float(points[j].x - ref1.x);
                            // float dy1 = float(points[j].y - ref1.y);
                            // float dz1 = float(points[j].z - ref1.z);

                            // float3 delta1 = m_box.wrap(make_float3(dx1, dy1, dz1));
                            vec3<float> delta1 = m_box.wrap(points[j] - ref1);
                            // float rsq1 = delta1.x*delta1.x + delta1.y*delta1.y + delta1.z*delta1.z;
                            float rsq1 = dot(delta1, delta1);

                            if (rsq1 < rmaxsq)
                                {
                                for(unsigned int k = 0; k < (2*m_l+1); ++k)
                                    {
                                    m_AveQlmi[(2*m_l+1)*i+k] += m_Qlmi[(2*m_l+1)*j+k];
                                    }
                                neighborcount++;
                                }
                            }
                        }
                    }
                }
            }
         //Normalize!
        for (unsigned int k = 0; k < (2*m_l+1); ++k)
            {
                m_AveQlmi[(2*m_l+1)*i+k] += m_Qlmi[(2*m_l+1)*i+k];
                m_AveQlmi[(2*m_l+1)*i+k]/= neighborcount;
            }
        //Ave Wli calculation
	    unsigned int counter = 0;
	    for(unsigned int u1 = 0; u1 < (2*m_l+1); ++u1)
	    	{
	    	for(unsigned int u2 = max( 0,int(m_l)-int(u1)); u2 < (min(3*m_l+1-u1,2*m_l+1)); ++u2)
	    		{
	    		unsigned int u3 = 3*m_l-u1-u2;
	    		m_AveWli[i]+= m_wigner3jvalues[counter]*m_AveQlmi[(2*m_l+1)*i+u1]*m_AveQlmi[(2*m_l+1)*i+u2]*m_AveQlmi[(2*m_l+1)*i+u3];
                counter+=1;
	    		}
	    	}//Ends loop for Norm Wli calcs
	    m_counter = counter;

        } //Ends loop over particles i for Qlmi calcs
    }

// void LocalWl::computeNorm(const float3 *points, unsigned int Np)
void LocalWl::computeNorm(const vec3<float> *points, unsigned int Np)
    {

    //Get wigner3j coefficients from wigner3j.cc
    int m_wignersize[10]={19,61,127,217,331,469,631,817,1027,1261};
    std::vector<double> m_wigner3jvalues (m_wignersize[m_l/2-1]);
    m_wigner3jvalues = getWigner3j(m_l);

    //Set local data size
    m_Np = Np;

    m_WliNorm = boost::shared_array<complex<double> >(new complex<double>[m_Np]);
    memset((void*)m_WliNorm.get(), 0, sizeof(complex<double>)*m_Np);

    //Average Q_lm over all particles, which was calculated in compute
    for(unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_Qlm[k]/= m_Np;
        }

    for(unsigned int i = 0; i < m_Np; ++i)
        {
        //Norm Wli calculation
	    unsigned int counter = 0;
	    for(unsigned int u1 = 0; u1 < (2*m_l+1); ++u1)
	    	{
	    	for(unsigned int u2 = max( 0,int(m_l)-int(u1)); u2 < (min(3*m_l+1-u1,2*m_l+1)); ++u2)
	    		{
	    		unsigned int u3 = 3*m_l-u1-u2;
	    		m_WliNorm[i]+= m_wigner3jvalues[counter]*m_Qlm[u1]*m_Qlm[u2]*m_Qlm[u3];
                counter+=1;
	    		}
	    	}//Ends loop for Norm Wli calcs
	    m_counter = counter;
        }
    }

//python wrapper for compute
void LocalWl::computePy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, NPY_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute the cell list
    // float3* points_raw = (float3*) num_util::data(points);
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
    compute(points_raw, Np);
    }

void LocalWl::computeNormPy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, NPY_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute the cell list
    // float3* points_raw = (float3*) num_util::data(points);
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
    compute(points_raw, Np);
    computeNorm(points_raw, Np);
    }

void LocalWl::computeAvePy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, NPY_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute the cell list
    // float3* points_raw = (float3*) num_util::data(points);
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
    compute(points_raw, Np);
    computeAve(points_raw, Np);
    }


/*! get wigner3j coefficients from python wrapper
 old version of getting wigner3j from python wrapper
void LocalWl::setWigner3jPy(boost::python::numeric::array wigner3jvalues)
	{
	//validate input type and rank
    num_util::check_type(wigner3jvalues, NPY_DOUBLE);
    num_util::check_rank(wigner3jvalues, 1);

    // get dimension
    unsigned int num_wigner3jcoefs = num_util::shape(wigner3jvalues)[0];
    m_wigner3jvalues = boost::shared_array<double>(new double[num_wigner3jcoefs]);

    // get the raw data pointers and compute the cell list
    double* wig3j = (double*) num_util::data(wigner3jvalues);
    for(unsigned int i = 0; i < num_wigner3jcoefs; i++)
    	{
    	m_wigner3jvalues[i] = wig3j[i];
    	}
    }
 */

void export_LocalWl()
    {
    class_<LocalWl>("LocalWl", init<trajectory::Box&, float, unsigned int>())
        .def("getBox", &LocalWl::getBox, return_internal_reference<>())
        .def("compute", &LocalWl::computePy)
        .def("computeNorm", &LocalWl::computeNormPy)
        .def("computeAve", &LocalWl::computeAvePy)
        .def("getWl", &LocalWl::getWlPy)
        .def("getWlNorm", &LocalWl::getWlNormPy)
        .def("getAveWl", &LocalWl::getAveWlPy)
        .def("getQl", &LocalWl::getQlPy)
        .def("setBox",&LocalWl::setBox)
        //.def("setWigner3j", &LocalWl::setWigner3jPy)
        .def("enableNormalization", &LocalWl::enableNormalization)
        .def("disableNormalization", &LocalWl::disableNormalization)
        ;
    }

}; }; // end namespace freud::localwl


