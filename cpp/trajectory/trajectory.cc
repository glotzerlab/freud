#include <iostream>

#include "num_util.h"
#include "trajectory.h"
#include "DCDLoader.h"
#include "Index1D.h"

using namespace std;
using namespace boost::python;
namespace bnp=boost::python::numeric;

namespace freud { namespace trajectory {

// utility function for taking particle cell data from discmc and returning a coordinate array
void extract_discmc_data(boost::python::numeric::array np_point_data,
                         boost::python::numeric::array np_cell_data,
                         boost::python::numeric::array np_cell_occupancy,
                         unsigned int m,
                         float w)
    {
    // basic checks on the input data
    num_util::check_type(np_point_data, PyArray_FLOAT);
    num_util::check_rank(np_point_data, 2);
    num_util::check_dim(np_point_data, 1, 3);    
    num_util::check_type(np_cell_data, PyArray_FLOAT);
    num_util::check_rank(np_cell_data, 4);
    num_util::check_dim(np_cell_data, 0, m);
    num_util::check_dim(np_cell_data, 1, m);
    num_util::check_dim(np_cell_data, 2, 4);
    num_util::check_dim(np_cell_data, 3, 2);
    num_util::check_type(np_cell_occupancy, PyArray_BYTE);
    num_util::check_rank(np_cell_occupancy, 2);
    num_util::check_dim(np_cell_occupancy, 0, m);
    num_util::check_dim(np_cell_occupancy, 1, m);

    
    // extract pointers from the input arrays
    float2* cell_data = (float2*) num_util::data(np_cell_data);
    unsigned char* cell_occupancy = (unsigned char *)num_util::data(np_cell_occupancy);
    float3* points = (float3*)num_util::data(np_point_data);
    
    // parameters (currently hardcoded to square arrays with 4 slots per cell)
    double L = m * double(w);
    unsigned int count = 0;
    Index2D ci(m);
    unsigned int max_occupancy = 4;
    
    // loop over the cells and fill out points in order
    for (unsigned int j = 0; j < m; j++)
        for (unsigned int i = 0; i < m; i++)
            {
            unsigned int cell_id = ci(i,j);
            unsigned int start = cell_id * max_occupancy;
            unsigned int o = cell_occupancy[cell_id];
            for (unsigned int k = start; k < start+o; k++)
                {
                double x = double(w) * i + double(cell_data[k].x) - L/2.0;
                double y = double(w) * j + double(cell_data[k].y) - L/2.0;
                points[count] = make_float3(x, y, 0.0f);
                count++;
                }
            }
    }
            
            
    

void export_trajectory()
    {
    def("extract_discmc_data", &extract_discmc_data);
    // define functions
    class_<Box>("Box", init<float, optional<bool> >())
        .def(init<float, float, float, optional<bool> >())
        .def("set2D", &Box::set2D)
        .def("is2D", &Box::is2D)
        .def("getLx", &Box::getLx)
        .def("getLy", &Box::getLy)
        .def("getLz", &Box::getLz)
        .def("getVolume", &Box::getVolume)
        .def("wrap", &Box::wrapPy)
        /*.def("unwrap", &Box::unwrapPy)
        .def("makeunit", &Box::makeunitPy)*/
        ;
    export_dcdloader();
    }

}; }; // end namespace freud::trajectory