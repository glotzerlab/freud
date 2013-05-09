#include "LocalQi.h"

#include <stdexcept>
#include <complex>
#include <boost/math/special_functions.hpp>
//#include <boost/math/special_functions/spheric_harmonic.hpp>

using namespace std;
using namespace boost::python;

namespace freud { namespace localqi {
    
LocalQi::LocalQi(const trajectory::Box& box, float rmax, unsigned int l)
    :m_box(box), m_rmax(rmax), m_lc(box, rmax), m_l(l)
    {
    if (m_rmax < 0.0f)
        throw invalid_argument("rmax must be positive");
    if (m_l%2 == 1)
        fprintf(stderr,"Current value of m_l is %d\n",m_l);     
        throw invalid_argument("This method requires even spherical harmonics!");
    
    }
   
void LocalQi::Ylm(const double theta, const double phi, std::vector<std::complex<double> > &Y)
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
    
   
    
void LocalQi::compute(const float3 *points, unsigned int Np) 
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
    memset((void*)m_Qlmi.get(), 0, sizeof(complex<double>)*(2*m_l+1)*m_Np);
    memset((void*)m_Qli.get(), 0, sizeof(double)*m_Np);

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
                // rij = rj - ri, from i pointing to j.
                float dx = float(points[j].x - ref.x);
                float dy = float(points[j].y - ref.y);
                float dz = float(points[j].z - ref.z);
                
                float3 delta = m_box.wrap(make_float3(dx, dy, dz));  
                float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;

                if (rsq < rmaxsq && rsq > 1e-6)  
                    {
                    double phi = atan2(delta.y,delta.x);   //0..2Pi
                    double theta = acos(delta.z / sqrt(rsq)); //0..Pi
                    
                    std::vector<std::complex<double> > Y;
                    LocalQi::Ylm(theta, phi,Y);  //Fill up Ylm vector
                    for(unsigned int k = 0; k < (2*m_l+1); ++k)
                        {
                        m_Qlmi[(2*m_l+1)*i+k]+=Y[k]; 
                        }
                    }
                    neighborcount++;
                }
            } //End loop going over neighbor cells (and thus all neighboring particles);
            //Normalize!
            for(unsigned int k = 0; k < (2*m_l+1); ++k)
                {
                m_Qlmi[(2*m_l+1)*i+k]/= neighborcount;
                m_Qli[i]+= abs( m_Qlmi[(2*m_l+1)*i+k]*conj(m_Qlmi[(2*m_l+1)*i+k]) ); //Square by multiplying self w/ complex conj, then take real comp
                }
        m_Qli[i]*=normalizationfactor;
        m_Qli[i]=sqrt(m_Qli[i]);
        } //Ends loop over particles i for Qlmi calcs
    } 
    
/*
void LocalQi::compute(const float3 *points, unsigned int Np)
    {
    m_lc.computeCellList(points,Np);
    m_Np = Np;
    float rmaxsq = m_rmax * m_rmax;
    m_psi_array = boost::shared_array<complex<double> >(new complex<double> [Np]);
    memset((void*)m_psi_array.get(), 0, sizeof(complex<double>)*Np);
    
    for (unsigned int i = 0; i<Np; i++)
        {
        //get cell point is in
        float3 ref = points[i];
        unsigned int ref_cell = m_lc.getCell(ref);
        unsigned int num_adjacent = 0;
        
        //loop over neighboring cells
        const std::vector<unsigned int>& neigh_cells = m_lc.getCellNeighbors(ref_cell);
        for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
            {
            unsigned int neigh_cell = neigh_cells[neigh_idx];
            
            //iterate over particles in cell
            locality::LinkCell::iteratorcell it = m_lc.itercell(neigh_cell);
            for (unsigned int j = it.next(); !it.atEnd(); j = it.next())
                {
                //compute r between the two particles
                float dx = float(ref.x - points[j].x);
                float dy = float(ref.y - points[j].y);
                float dz = float(ref.z - points[j].z);
                float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                
                float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                if (rsq < rmaxsq && rsq > 1e-6)
                    {
                    //compute psi for neighboring particle(only constructed for 2d)
                    //double psi_ij = atan2(delta.y, delta.x);
                    //m_psi_array[i] += exp(complex<double>(0,6*psi_ij));
                    //num_adjacent++;
                    }
                }
            }
	// Don't divide by zero if the particle has no neighbors (this leaves psi at 0)
	if(num_adjacent)
	  m_psi_array[i] /= complex<double>(num_adjacent);  
        }
    }
*/


void LocalQi::computePy(boost::python::numeric::array points)
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
    
void export_LocalQi()
    {
    class_<LocalQi>("LocalQi", init<trajectory::Box&, float, unsigned int>())
        .def("getBox", &LocalQi::getBox, return_internal_reference<>())
        .def("compute", &LocalQi::computePy)
        .def("getQi", &LocalQi::getQliPy)
        ;
    }
    
}; }; // end namespace freud::localqi


