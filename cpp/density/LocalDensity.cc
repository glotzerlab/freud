#include "LocalDensity.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#include <complex>
#include <tbb/tbb.h>

using namespace std;
using namespace boost::python;
using namespace tbb;

/*! \file LocalDensity.h
    \brief Routines for computing local density around a point
*/

namespace freud { namespace density {

LocalDensity::LocalDensity(const trajectory::Box& box, float rcut, float volume)
    : m_box(box), m_rcut(rcut), m_volume(volume), m_lc(box, rcut), m_Np(0)
    {
    }

//! \internal
/*! \brief Helper class to compute local density in parallel
*/
class ComputeLocalDensity
    {
    private:
        float *m_density_array;
        float *m_num_neighbors_array;
        const trajectory::Box& m_box;
        const float m_rcut;
        const float m_volume;
        const locality::LinkCell& m_lc;
        const float3 *m_points;
    public:
        ComputeLocalDensity(float *density_array,
                            float *num_neighbors_array,
                            const trajectory::Box& box,
                            const float rcut,
                            const float volume,
                            const locality::LinkCell& lc,
                            const float3 *points)
            : m_density_array(density_array), m_num_neighbors_array(num_neighbors_array), m_box(box), m_rcut(rcut),
              m_volume(volume), m_lc(lc), m_points(points)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            float rcutsq = m_rcut * m_rcut;

            for(size_t i=r.begin(); i!=r.end(); ++i)
                {
                unsigned int num_neighbors = 0;

                // get cell point is in
                float3 ref = m_points[i];
                unsigned int ref_cell = m_lc.getCell(ref);

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
                        float dx = float(ref.x - m_points[j].x);
                        float dy = float(ref.y - m_points[j].y);
                        float dz = float(ref.z - m_points[j].z);
                        float3 delta = m_box.wrap(make_float3(dx, dy, dz));

                        float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                        if (rsq < rcutsq && i != j)
                            {
                            num_neighbors++;
                            }
                        }
                    }

                m_num_neighbors_array[i] = num_neighbors;
                if (m_box.is2D())
                    {
                    // local density is area of particles divided by the area of the circle
                    m_density_array[i] = (m_volume * m_num_neighbors_array[i]) / (M_PI * m_rcut * m_rcut);
                    }
                else
                    {
                    // local density is volume of particles divided by the volume of the sphere
                    m_density_array[i] = (m_volume * m_num_neighbors_array[i]) / (4.0f/3.0f * M_PI * m_rcut * m_rcut * m_rcut);
                    }
                }
            }
    };

void LocalDensity::compute(const float3 *points, unsigned int Np)
    {
    // compute the cell list
    m_lc.computeCellList(points,Np);

    // reallocate the output array if it is not the right size
    if (Np != m_Np)
        {
        m_density_array = boost::shared_array<float>(new float[Np]);
        m_num_neighbors_array = boost::shared_array<float>(new float[Np]);
        }

    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,Np), ComputeLocalDensity(m_density_array.get(),
                                                                  m_num_neighbors_array.get(),
                                                                  m_box,
                                                                  m_rcut,
                                                                  m_volume,
                                                                  m_lc,
                                                                  points));

    // save the last computed number of particles
    m_Np = Np;
    }

void LocalDensity::computePy(boost::python::numeric::array points)
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

void export_LocalDensity()
    {
    class_<LocalDensity>("LocalDensity", init<trajectory::Box&, float, float>())
        .def(init<trajectory::Box&, float, float>())
        .def("compute", &LocalDensity::computePy)
        .def("getDensity", &LocalDensity::getDensityPy)
        .def("getNumNeighbors", &LocalDensity::getNumNeighborsPy)
        ;
    }

}; }; // end namespace freud::density
