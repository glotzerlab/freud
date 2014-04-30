#include "LocalQl.h"

#include <stdexcept>
#include <complex>
//#include <boost/math/special_functions.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

using namespace std;
using namespace boost::python;

/*! \file LocalQl.cc
    \brief Compute a Ql per particle
*/

namespace freud { namespace sphericalharmonicorderparameters {

LocalQl::LocalQl(const trajectory::Box& box, float rmax, unsigned int l)
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
    }

void LocalQl::Ylm(const double theta, const double phi, std::vector<std::complex<double> > &Y)
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

void LocalQl::compute(const float3 *points, unsigned int Np)
    {

    //Set local data size
    m_Np = Np;

    //Initialize cell list
    m_lc.computeCellList(points,m_Np);

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
        float3 ref = points[i];
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
                float dx = float(points[j].x - ref.x);
                float dy = float(points[j].y - ref.y);
                float dz = float(points[j].z - ref.z);

                float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;

                if (rsq < rmaxsq)
                    {
                    double phi = atan2(delta.y,delta.x);      //0..2Pi
                    double theta = acos(delta.z / sqrt(rsq)); //0..Pi

                    std::vector<std::complex<double> > Y;
                    LocalQl::Ylm(theta, phi,Y);  //Fill up Ylm vector
                    for(unsigned int k = 0; k < (2*m_l+1); ++k)
                        {
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
                m_Qli[i]+= abs( m_Qlmi[(2*m_l+1)*i+k]*conj(m_Qlmi[(2*m_l+1)*i+k]) ); //Square by multiplying self w/ complex conj, then take real comp
                m_Qlm[k]+= m_Qlmi[(2*m_l+1)*i+k];
                }
        m_Qli[i]*=normalizationfactor;
        m_Qli[i]=sqrt(m_Qli[i]);
        } //Ends loop over particles i for Qlmi calcs
    }

void LocalQl::computeAve(const float3 *points, unsigned int Np)
    {
    //Set local data size
    m_Np = Np;

    //Initialize cell list
    m_lc.computeCellList(points,m_Np);

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
        float3 ref = points[i];
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
                float3 ref1 = points[n1];
                unsigned int ref1_cell = m_lc.getCell(ref1);
                if (n1 == i) 
                    {
                        continue;
                    }
                // rij = rj - ri, from i pointing to j.
                float dx = float(points[n1].x - ref.x);
                float dy = float(points[n1].y - ref.y);
                float dz = float(points[n1].z - ref.z);

                float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;

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
                            float dx1 = float(points[j].x - ref1.x);
                            float dy1 = float(points[j].y - ref1.y);
                            float dz1 = float(points[j].z - ref1.z);

                            float3 delta1 = m_box.wrap(make_float3(dx1, dy1, dz1));
                            float rsq1 = delta1.x*delta1.x + delta1.y*delta1.y + delta1.z*delta1.z;

                            if (rsq1 < rmaxsq)
                                {
                                for(unsigned int k = 0; k < (2*m_l+1); ++k)
                                    {
                                    //adding all the Qlm of the neighbors
                                    m_AveQlmi[(2*m_l+1)*i+k] += m_Qlmi[(2*m_l+1)*j+k];
                                    }
                                neighborcount++;
                                }
                            }
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
    
void LocalQl::computeNorm(const float3 *points, unsigned int Np)
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

void LocalQl::computeAveNorm(const float3 *points, unsigned int Np)
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


void LocalQl::computePy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute the cell list
    float3* points_raw = (float3*) num_util::data(points);
    compute(points_raw, Np);
    }

void LocalQl::computeAvePy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute the cell list
    float3* points_raw = (float3*) num_util::data(points);
    compute(points_raw, Np);
    computeAve(points_raw, Np);
    }

void LocalQl::computeNormPy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute the cell list
    float3* points_raw = (float3*) num_util::data(points);
    compute(points_raw, Np);
    computeNorm(points_raw, Np);
    }

void LocalQl::computeAveNormPy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute the cell list
    float3* points_raw = (float3*) num_util::data(points);
    compute(points_raw, Np);
    computeAve(points_raw, Np);
    computeAveNorm(points_raw, Np);
    }

void export_LocalQl()
    {
    class_<LocalQl>("LocalQl", init<trajectory::Box&, float, unsigned int>())
        .def("getBox", &LocalQl::getBox, return_internal_reference<>())
        .def("setBox", &LocalQl::setBox)
        .def("compute", &LocalQl::computePy)
        .def("computeAve", &LocalQl::computeAvePy)
        .def("computeNorm", &LocalQl::computeNormPy)
        .def("computeAveNorm", &LocalQl::computeAveNormPy)
        .def("getQl", &LocalQl::getQlPy)
        .def("getAveQl", &LocalQl::getAveQlPy)
        .def("getQlNorm", &LocalQl::getQlNormPy)
        .def("getQlAveNorm", &LocalQl::getQlAveNormPy)
        ;
    }

}; }; // end namespace freud::localqi


