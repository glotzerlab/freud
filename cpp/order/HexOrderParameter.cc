#include "HexOrderParameter.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#include <complex>

using namespace std;
using namespace boost::python;

namespace freud { namespace order {
    
HexOrderParameter::HexOrderParameter(const trajectory::Box& box, float rmax, float k=6)
    :m_box(box), m_rmax(rmax), m_k(k), m_lc(box, rmax)
    {
    }
    
void HexOrderParameter::compute(const float3 *points, unsigned int Np)
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
                    double psi_ij = atan2(delta.y, delta.x);
                    m_psi_array[i] += exp(complex<double>(0,m_k*psi_ij));
                    num_adjacent++;
                    }
                }
            }
        // Don't divide by zero if the particle has no neighbors (this leaves psi at 0)
	    if(num_adjacent)
	       m_psi_array[i] /= complex<double>(num_adjacent);  
        }
    }

void HexOrderParameter::computePy(boost::python::numeric::array points)
    {
    //validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    
    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];
    
    // get the raw data pointers and compute order parameter
    float3* points_raw = (float3*) num_util::data(points);
        
        // compute the order parameter with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(points_raw, Np);
        }
    }
    
void export_HexOrderParameter()
    {
    class_<HexOrderParameter>("HexOrderParameter", init<trajectory::Box&, float>())
        .def(init<trajectory::Box&, float, float>())
        .def("getBox", &HexOrderParameter::getBox, return_internal_reference<>())
        .def("compute", &HexOrderParameter::computePy)
        .def("getPsi", &HexOrderParameter::getPsiPy)
        ;
    }
    
}; }; // end namespace freud::order


