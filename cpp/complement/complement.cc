#include "complement.h"

#include <stdexcept>
#include <complex>
#include <boost/math/special_functions.hpp>

using namespace std;
using namespace boost::python;

namespace freud { namespace complement {   
    
complement::complement(const trajectory::Box& box, float rmax)
    :m_box(box), m_rmax(rmax)
    {
    if (m_rmax < 0.0f)
        throw invalid_argument("rmax must be positive!");   
    }

void complement::compute(const float2 *position,
                         const float *angle,
                         const float2 *polygon,
                         const float *cavity,
                         unsigned int N,
                         unsigned int NV,
                         unsigned int NC
                        ) 
    {
    // There is going to be need of the neighbor list functionality here, otherwise it will be abysmally slow
    } 

void complement::computePy(boost::python::numeric::array position,
                           boost::python::numeric::array angle,
                           boost::python::numeric::array polygon,
                           boost::python::numeric::array cavity
                          )
    {
    //validate input type and rank
    // Will be Nx2 array
    num_util::check_type(position, PyArray_FLOAT);
    num_util::check_rank(position, 2);
    
    // validate that the 2nd dimension is only 2
    num_util::check_dim(position, 1, 2);
    unsigned int N = num_util::shape(position)[0];
    
    //validate input type and rank
    // Will be array of size N
    num_util::check_type(angle, PyArray_FLOAT);
    num_util::check_rank(angle, 1);
    
    // validate that the 1st dimension is only 2
    num_util::check_dim(angle, 0, 2);
    // Should I check that they are the same size
    //unsigned int Np = num_util::shape(angle)[0];
    
    //validate input type and rank
    // Will be array of size NVx2
    num_util::check_type(polygon, PyArray_FLOAT);
    num_util::check_rank(polygon, 2);
    
    // validate that the 1st dimension is only 2
    num_util::check_dim(polygon, 0, 2);
    unsigned int NV = num_util::shape(polygon)[0];
    
    //validate input type and rank
    // Will be array of size N
    num_util::check_type(cavity, PyArray_FLOAT);
    num_util::check_rank(cavity, 1);
    
    // validate that the 1st dimension is only 2
    num_util::check_dim(cavity, 0, 2);
    unsigned int NC = num_util::shape(cavity)[0];
    
    // get the raw data pointers and compute the cell list
    float2* position_raw = (float2*) num_util::data(position);
    float* angle_raw = (float*) num_util::data(angle);
    float2* polygon_raw = (float2*) num_util::data(polygon);
    float* cavity_raw = (float*) num_util::data(cavity);
    compute(position_raw, angle_raw, polygon_raw, cavity_raw, N, NV, NC);
    }
    
void export_complement()
    {
    class_<complement>("complement", init<trajectory::Box&, float>())
        .def("getBox", &complement::getBox, return_internal_reference<>())
        .def("compute", &complement::computePy)
        ;
    }
    
}; }; // end namespace freud::complement


