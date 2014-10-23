#include "LocalQlNear.h"

#include <stdexcept>
#include <complex>
#include <boost/math/special_functions/spherical_harmonic.hpp>

using namespace std;
using namespace boost::python;

/*! \file LocalQl.cc
    \brief Compute a Ql per particle using N nearest neighbors instead of r_cut
*/

namespace freud { namespace sphericalharmonicorderparameters {

LocalQlNear::LocalQlNear(const trajectory::Box& box, float rmax, unsigned int l, unsigned int kn = 12)
    :m_box(box), m_rmax(rmax), m_nn(box, rmax, kn), m_l(l), m_k(kn)
    {
    if (m_rmax < 0.0f)
        throw invalid_argument("rmax must be positive!");
    if (kn < 0)
        throw invalid_argument("# of nearest neighbors must be positive!");
    if (m_l < 2)
        throw invalid_argument("l must be two or greater (and even)!");
    if (m_l%2 == 1)
        {
        fprintf(stderr,"Current value of m_l is %d\n",m_l);
        throw invalid_argument("This method requires even values of l!");
        }
    }

void LocalQlNear::Ylm(const double theta, const double phi, std::vector<std::complex<double> > &Y)
    {
    if(Y.size() != 2*m_l+1)
        Y.resize(2*m_l+1);

    for(int m = -m_l; m <=0; m++)
        //Doc for boost spherical harmonic
        //http://www.boost.org/doc/libs/1_53_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/sph_harm.html
        // theta = colatitude = 0..Pi
        // Phi = azimuthal (longitudinal) 0..2pi).
        Y[m+m_l]= boost::math::spherical_harmonic(m_l, m, theta, phi);

    for(unsigned int i = 1; i <= m_l; i++)
        Y[i+m_l] = Y[-i+m_l];
    }

// void LocalQl::compute(const float3 *points, unsigned int Np)
void LocalQlNear::compute(const vec3<float> *points, unsigned int Np)
    {

    //Set local data size
    m_Np = Np;

    //Initialize cell list
    m_nn.compute(points,Np);
    //m_nn.computeCellList(points,m_Np);

    double rmaxsq = m_rmax * m_rmax;
    double normalizationfactor = 4*M_PI/(2*m_l+1);


    //newmanrs:  For efficiency, if Np != m_Np, we could not reallocate these! Maybe.
    // for safety and debugging laziness, reallocate each time
    m_Qlmi = boost::shared_array<complex<double> >(new complex<double> [(2*m_l+1)*m_Np]);
    m_Qli = boost::shared_array<double>(new double[m_Np]);
    m_Qlm = boost::shared_array<complex<double> >(new complex<double>[2*m_l+1]);
    memset((void*)m_Qlmi.get(), 0, sizeof(complex<double>)*(2*m_l+1)*m_Np);
    memset((void*)m_Qli.get(), 0, sizeof(double)*m_Np);
    memset((void*)m_Qlm.get(), 0, sizeof(complex<double>)*(2*m_l+1));

    for (unsigned int i = 0; i<m_Np; i++)
        {
        //get cell point is in
        // float3 ref = points[i];
        vec3<float> ref = points[i];
        boost::shared_array<unsigned int> neighbors = m_nn.getNeighbors(i);
        //unsigned int ref_cell = m_nn.getCell(ref);
        //unsigned int neighborcount=0;

        //loop over neighboring cells
        //const std::vector<unsigned int>& neigh_cells = m_nn.getCellNeighbors(ref_cell);
        for (unsigned int neigh_idx = 0; neigh_idx < m_k; neigh_idx++)
            {
            unsigned int j = neighbors[neigh_idx];

            //compute r between the two particles.
            vec3<float> delta = m_box.wrap(points[j] - ref);
            float rsq = dot(delta, delta);
            
            if (rsq > 1e-6)
                {
                double phi = atan2(delta.y,delta.x);      //0..2Pi
                double theta = acos(delta.z / sqrt(rsq)); //0..Pi

                std::vector<std::complex<double> > Y;
                LocalQlNear::Ylm(theta, phi,Y);  //Fill up Ylm vector
                for(unsigned int k = 0; k < (2*m_l+1); ++k)
                    {
                    m_Qlmi[(2*m_l+1)*i+k]+=Y[k];
                    }
                }
            } //End loop going over neighbor cells (and thus all neighboring particles);
            //Normalize!
        for(unsigned int k = 0; k < (2*m_l+1); ++k)
            {
            m_Qlmi[(2*m_l+1)*i+k]/= m_k;
            m_Qli[i]+= abs( m_Qlmi[(2*m_l+1)*i+k]*conj(m_Qlmi[(2*m_l+1)*i+k]) ); //Square by multiplying self w/ complex conj, then take real comp
            m_Qlm[k]+= m_Qlmi[(2*m_l+1)*i+k];
            }
        m_Qli[i]*=normalizationfactor;
        m_Qli[i]=sqrt(m_Qli[i]);
        } //Ends loop over particles i for Qlmi calcs
    }

// void LocalQl::computeAve(const float3 *points, unsigned int Np)
void LocalQlNear::computeAve(const vec3<float> *points, unsigned int Np)
    {
    //Set local data size
    m_Np = Np;

    // compute the cell list
    m_nn.compute(points,Np);

    double rmaxsq = m_rmax * m_rmax;
    double normalizationfactor = 4*M_PI/(2*m_l+1);


    //newmanrs:  For efficiency, if Np != m_Np, we could not reallocate these! Maybe.
    // for safety and debugging laziness, reallocate each time
    m_AveQlmi = boost::shared_array<complex<double> >(new complex<double> [(2*m_l+1)*m_Np]);
    m_AveQli = boost::shared_array<double>(new double[m_Np]);
    m_AveQlm = boost::shared_array<complex<double> >(new complex<double> [(2*m_l+1)*m_Np]);
    memset((void*)m_AveQlmi.get(), 0, sizeof(complex<double>)*(2*m_l+1)*m_Np);
    memset((void*)m_AveQli.get(), 0, sizeof(double)*m_Np);
    memset((void*)m_AveQlm.get(), 0, sizeof(double)*m_Np);

    for (unsigned int i = 0; i<m_Np; i++)
        {
        //get cell point is in
        // float3 ref = points[i];
        vec3<float> ref = points[i];
        unsigned int neighborcount = 1;
        boost::shared_array<unsigned int> neighbors = m_nn.getNeighbors(i);
        //unsigned int ref_cell = m_nn.getCell(ref);
        //unsigned int neighborcount=1;

        //loop over neighboring cells
        //const std::vector<unsigned int>& neigh_cells = m_nn.getCellNeighbors(ref_cell);
        //loop over neighbors
        for (unsigned int neigh_idx = 0; neigh_idx < m_k; neigh_idx++)
            {
            //get cell points of 1st neighbor
            unsigned int j = neighbors[neigh_idx];

            if (j == i)
                {
                continue;
                }

            vec3<float> ref1 = points[j];
            vec3<float> delta = m_box.wrap(points[j] - ref);
            
            float rsq = dot(delta, delta);
            if (rsq > 1e-6)
                {
                boost::shared_array<unsigned int> neighbors_2 = m_nn.getNeighbors(j);

                for (unsigned int neigh1_idx = 0; neigh1_idx < m_k; neigh1_idx++)
                    {
                    //get cell points of 2nd neighbor
                    unsigned int n1 = neighbors_2[neigh1_idx];
                    if (n1 == j)
                        {
                        continue;
                        }

                    vec3<float> delta1 = m_box.wrap(points[n1] - ref1);
                    float rsq1 = dot(delta1, delta1);
    
                    if (rsq1 > 1e-6)
                        {
                        for(unsigned int k = 0; k < (2*m_l+1); ++k)
                            {
                            //adding all the Qlm of the neighbors
                            // change to Index?
                            m_AveQlmi[(2*m_l+1)*i+k] += m_Qlmi[(2*m_l+1)*n1+k];
                            }
                        neighborcount++;
                        }
                    }
                } //End loop going over neighbor cells (and thus all neighboring particles);
            }
        //Normalize!
        for (unsigned int k = 0; k < (2*m_l+1); ++k)
            {
                //adding the Qlm of the particle i itself
                m_AveQlmi[(2*m_l+1)*i+k] += m_Qlmi[(2*m_l+1)*i+k];
                m_AveQlmi[(2*m_l+1)*i+k]/= neighborcount;
                m_AveQlm[k]+= m_AveQlmi[(2*m_l+1)*i+k];
                m_AveQli[i]+= abs( m_AveQlmi[(2*m_l+1)*i+k]*conj(m_AveQlmi[(2*m_l+1)*i+k]) ); //Square by multiplying self w/ complex conj, then take real comp
            }
        m_AveQli[i]*=normalizationfactor;
        m_AveQli[i]=sqrt(m_AveQli[i]);
        } //Ends loop over particles i for Qlmi calcs
    }

// void LocalQl::computeNorm(const float3 *points, unsigned int Np)
void LocalQlNear::computeNorm(const vec3<float> *points, unsigned int Np)
    {

    //Set local data size
    m_Np = Np;
    double normalizationfactor = 4*M_PI/(2*m_l+1);

    m_QliNorm = boost::shared_array<double>(new double[m_Np]);
    memset((void*)m_QliNorm.get(), 0, sizeof(double)*m_Np);

    //Average Q_lm over all particles, which was calculated in compute
    for(unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_Qlm[k]/= m_Np;
        } 

    for(unsigned int i = 0; i < m_Np; ++i)
        {
        for(unsigned int k = 0; k < (2*m_l+1); ++k)
             {
            m_QliNorm[i]+= abs( m_Qlm[k]*conj(m_Qlm[k]) ); //Square by multiplying self w/ complex conj, then take real comp
            }
            m_QliNorm[i]*=normalizationfactor;
            m_QliNorm[i]=sqrt(m_QliNorm[i]);
        } 
    }

// void LocalQl::computeAveNorm(const float3 *points, unsigned int Np)
void LocalQlNear::computeAveNorm(const vec3<float> *points, unsigned int Np)
    { 

    //Set local data size
    m_Np = Np;
    double normalizationfactor = 4*M_PI/(2*m_l+1);

    m_QliAveNorm = boost::shared_array<double>(new double[m_Np]);
    memset((void*)m_QliAveNorm.get(), 0, sizeof(double)*m_Np);

    //Average Q_lm over all particles, which was calculated in compute
    for(unsigned int k = 0; k < (2*m_l+1); ++k)
        {
        m_AveQlm[k]/= m_Np;
        } 

    for(unsigned int i = 0; i < m_Np; ++i)
        { 
        for(unsigned int k = 0; k < (2*m_l+1); ++k)
            {
            m_QliAveNorm[i]+= abs( m_AveQlm[k]*conj(m_AveQlm[k]) ); //Square by multiplying self w/ complex conj, then take real comp
             }
            m_QliAveNorm[i]*=normalizationfactor;
            m_QliAveNorm[i]=sqrt(m_QliAveNorm[i]);
        }
    }


void LocalQlNear::computePy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute the cell list
    // float3* points_raw = (float3*) num_util::data(points);
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
    compute(points_raw, Np);
    }

void LocalQlNear::computeAvePy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
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

void LocalQlNear::computeNormPy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
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

void LocalQlNear::computeAveNormPy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute the cell list
    // float3* points_raw = (float3*) num_util::data(points);
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
    compute(points_raw, Np);
    computeAve(points_raw, Np);
    computeAveNorm(points_raw, Np);
    }

void export_LocalQlNear()
    {
    class_<LocalQlNear>("LocalQlNear", init<trajectory::Box&, float, unsigned int, optional<unsigned int> >())
        .def("getBox", &LocalQlNear::getBox, return_internal_reference<>())
        .def("setBox", &LocalQlNear::setBox)
        .def("compute", &LocalQlNear::computePy)
        .def("computeAve", &LocalQlNear::computeAvePy)
        .def("computeNorm", &LocalQlNear::computeNormPy)
        .def("computeAveNorm", &LocalQlNear::computeAveNormPy)
        .def("getQl", &LocalQlNear::getQlPy)
        .def("getAveQl", &LocalQlNear::getAveQlPy)
        .def("getQlNorm", &LocalQlNear::getQlNormPy)
        .def("getQlAveNorm", &LocalQlNear::getQlAveNormPy)
        ;
    }

}; }; // end namespace freud::localqinear
