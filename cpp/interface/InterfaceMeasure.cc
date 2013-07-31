#include "InterfaceMeasure.h"

using namespace std;
using namespace boost::python;

/*! \file InterfaceMeasure.h
    \brief Compute the size of an interface between two point clouds
*/

namespace freud { namespace interface {

InterfaceMeasure::InterfaceMeasure(const trajectory::Box& box, float r_cut)
    : m_box(box), m_rcut(r_cut), m_lc(box, r_cut)
    {
        if (r_cut < 0.0f)
            throw invalid_argument("r_cut must be positive");
    }

unsigned int InterfaceMeasure::compute(const float3 *ref_points,
                                       unsigned int Nref,
                                       const float3 *points,
                                       unsigned int Np)
{
    assert(ref_points);
    assert(points);
    assert(Nref > 0);
    assert(Np > 0);

    // bin the second set of points
    m_lc.computeCellList(points, Np);

    unsigned int interfaceCount = 0;
    float rcutsq = m_rcut * m_rcut;

    // for each reference point
    for( unsigned int i = 0; i < Nref; i++)
    {
        bool inInterface = false;

        // get the cell the point is in
        float3 ref = ref_points[i];
        unsigned int ref_cell = m_lc.getCell(ref);

        // loop over all neighboring cells
        const std::vector<unsigned int>& neigh_cells = m_lc.getCellNeighbors(ref_cell);
        for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
        {
            if(inInterface)
                break;
            unsigned int neigh_cell = neigh_cells[neigh_idx];

            // iterate over the particles in that cell
            locality::LinkCell::iteratorcell it = m_lc.itercell(neigh_cell);
            for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
            {
                if(inInterface)
                    break;
                // compute the distance between the two particles
                float dx = float(ref.x - points[j].x);
                float dy = float(ref.y - points[j].y);
                float dz = float(ref.z - points[j].z);

                float3 delta = m_box.wrap(make_float3(dx, dy, dz));

                // Check if the distance is less than the cutoff
                float deltasq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
                if (deltasq < rcutsq)
                {
                    inInterface = true;
                    break;
                }
            }
        }
        if(inInterface)
            interfaceCount++;
    }
    return interfaceCount;
}

unsigned int InterfaceMeasure::computePy(boost::python::numeric::array ref_points,
                                         boost::python::numeric::array points)
{
    // validate input type
    num_util::check_type(ref_points, PyArray_FLOAT);
    num_util::check_type(points, PyArray_FLOAT);

    // validate input rank
    num_util::check_rank(ref_points, 2);
    num_util::check_rank(points, 2);

    // validate that the second dimension is only 3
    num_util::check_dim(ref_points, 1, 3);
    num_util::check_dim(points, 1, 3);

    // get the number of points in the arrays
    unsigned int Nref = num_util::shape(points)[0];
    unsigned int Np = num_util::shape(ref_points)[0];

    // get the raw data pointers and compute the interface
    float3* ref_points_raw = (float3*) num_util::data(ref_points);
    float3* points_raw = (float3*) num_util::data(points);

    return compute(ref_points_raw, Nref, points_raw, Np);
}

// Export the methods inside the InterfaceMeasure class
void export_InterfaceMeasure()
{
    class_<InterfaceMeasure>("InterfaceMeasure", init<trajectory::Box&, float>())
        .def("getBox", &InterfaceMeasure::getBox, return_internal_reference<>())
        .def("compute",&InterfaceMeasure::computePy)
        ;
}

}; }; // end namespace freud::density
