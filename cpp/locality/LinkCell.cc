#include <boost/python.hpp>
#include <stdexcept>
#include <algorithm>

#include "num_util.h"
#include "LinkCell.h"
#include "../trajectory/trajectory.h"
#include "ScopedGILRelease.h"

using namespace std;
using namespace boost::python;

/*! \file LinkCell.cc
    \brief Build a cell list from a set of points
*/

namespace freud { namespace locality {

// This is only used to initialize a pointer for the new triclinic setup
// this will be phased out when the one true cell list is created
// but until then, enjoy this mediocre hack
LinkCell::LinkCell() : m_box(trajectory::Box()), m_Np(0), m_cell_width(0)
    {
    m_celldim = vec3<unsigned int>(0,0,0);
    }

LinkCell::LinkCell(const trajectory::Box& box, float cell_width) : m_box(box), m_Np(0), m_cell_width(cell_width)
    {
    // check if the cell width is too wide for the box
    m_celldim  = computeDimensions();
    //Check if box is too small!
    bool too_wide =  m_cell_width > m_box.getLx()/2.0 || m_cell_width > m_box.getLy()/2.0;
    if (!m_box.is2D())
        {
        too_wide |=  m_cell_width > m_box.getLz()/2.0;
        }
    if (too_wide)
        {
        throw runtime_error("Cannot generate a cell list where cell_width is larger than half the box.");
        }
    //only 1 cell deep in 2D
    if (m_box.is2D())
        {
        m_celldim.z = 1;
        }
    m_cell_index = Index3D(m_celldim.x, m_celldim.y, m_celldim.z);
    computeCellNeighbors();
    }

void LinkCell::updateBox(const trajectory::Box& box, float cell_width)
    {
    // check if the cell width is too wide for the box
    vec3<unsigned int> celldim  = computeDimensions(box, cell_width);
    //Check if box is too small!
    bool too_wide =  cell_width > box.getLx()/2.0 || cell_width > box.getLy()/2.0;
    if (!box.is2D())
        {
        too_wide |=  cell_width > box.getLz()/2.0;
        }
    if (too_wide)
        {
        throw runtime_error("Cannot generate a cell list where cell_width is larger than half the box.");
        }
    //only 1 cell deep in 2D
    if (box.is2D())
        {
        celldim.z = 1;
        }
    // check if the box is changed
    m_box = box;
    m_cell_width = cell_width;
    if (!((celldim.x == m_celldim.x) && (celldim.y == m_celldim.y) && (celldim.z == m_celldim.z)))
        {
        m_cell_index = Index3D(celldim.x, celldim.y, celldim.z);
        if (m_cell_index.getNumElements() < 1)
            {
            throw runtime_error("At least one cell must be present");
            }
        m_celldim  = celldim;
        computeCellNeighbors();
        }
    }

unsigned int LinkCell::roundDown(unsigned int v, unsigned int m)
    {
    // use integer floor division
    unsigned int d = v/m;
    return d*m;
    }

const vec3<unsigned int> LinkCell::computeDimensions() const
    {
    return computeDimensions(m_box, m_cell_width);
    }

const vec3<unsigned int> LinkCell::computeDimensions(const trajectory::Box& box, float cell_width) const
    {
    vec3<unsigned int> dim;

    //multiple is a holdover from hpmc...doesn't really need to be kept
    unsigned int multiple = 1;
    vec3<float> L = box.getNearestPlaneDistance();
    dim.x = roundDown((unsigned int)((L.x) / (cell_width)), multiple);
    dim.y = roundDown((unsigned int)((L.y) / (cell_width)), multiple);

    // Add a ghost layer on every side where boundary conditions are non-periodic
    if (! box.getPeriodic().x)
        dim.x += 2;
    if (! box.getPeriodic().y)
        dim.y += 2;

    if (box.is2D())
        {
        dim.z = 1;
        }
    else
        {
        dim.z = roundDown((unsigned int)((L.z) / (cell_width)), multiple);
        // add ghost layer if necessary
        if (! box.getPeriodic().z)
            dim.z += 2;
        }

    // In extremely small boxes, the calculated dimensions could go to zero, but need at least one cell in each dimension
    //  for particles to be in a cell and to pass the checkCondition tests.
    // Note: Freud doesn't actually support these small boxes (as of this writing), but this function will return the correct dimensions
    //  required anyways.
    if (dim.x == 0)
        dim.x = 1;
    if (dim.y == 0)
        dim.y = 1;
    if (dim.z == 0)
        dim.z = 1;
    return dim;
    }

void LinkCell::computeCellListPy(boost::python::numeric::array points)
    {
    // validate input type and rank
    num_util::check_type(points, NPY_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute the cell list
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);

        // compute the cell list with the GIL released
        {
        util::ScopedGILRelease gil;
        computeCellList(points_raw, Np);
        }
    }

//Deprecated.  Users should use the modern vec3<float> interfaces
void LinkCell::computeCellList(const float3 *points, unsigned int Np)
    {
        //Copy into appropriate vec3<float>;
        vec3<float>* pointscopy = new vec3<float>[Np];
        for(unsigned int i = 0; i < Np; i++) {
            pointscopy[i].x=points[i].x;
            pointscopy[i].y=points[i].y;
            pointscopy[i].z=points[i].z;
        }
        computeCellList(pointscopy, Np);
        delete[] pointscopy;
    }

void LinkCell::computeCellList(const vec3<float> *points, unsigned int Np)
    {
    if (Np == 0)
        {
        throw runtime_error("Cannot generate a cell list of 0 particles");
        }

    m_Np = Np;

    // determine the number of cells and allocate memory
    unsigned int Nc = getNumCells();
    assert(Nc > 0);
    // shouldn't this be deleted?
    m_cell_list = boost::shared_array<unsigned int>(new unsigned int[Np + Nc]);

    // initialize memory
    for (unsigned int cell = 0; cell < Nc; cell++)
        {
        m_cell_list[Np + cell] = LINK_CELL_TERMINATOR;
        }

    // generate the cell list
    assert(points);

    for (int i = Np-1; i >= 0; i--)
        {
        unsigned int cell = getCell(points[i]);
        m_cell_list[i] = m_cell_list[Np+cell];
        m_cell_list[Np+cell] = i;
        }
    }

void LinkCell::computeCellNeighbors()
    {
    // clear the list
    m_cell_neighbors.clear();
    m_cell_neighbors.resize(getNumCells());

    // for each cell
    for (unsigned int k = 0; k < m_cell_index.getD(); k++)
        for (unsigned int j = 0; j < m_cell_index.getH(); j++)
            for (unsigned int i = 0; i < m_cell_index.getW(); i++)
                {
                // clear the list
                unsigned int cur_cell = m_cell_index(i,j,k);
                m_cell_neighbors[cur_cell].clear();

                // loop over the neighbor cells
                int starti, startj, startk;
                int endi, endj, endk;
                if (m_celldim.x < 3)
                    {
                    starti = (int)i;
                    }
                else
                    {
                    starti = (int)i - 1;
                    }
                if (m_celldim.y < 3)
                    {
                    startj = (int)j;
                    }
                else
                    {
                    startj = (int)j - 1;
                    }
                if (m_celldim.z < 3)
                    {
                    startk = (int)k;
                    }
                else
                    {
                    startk = (int)k - 1;
                    }

                if (m_celldim.x < 2)
                    {
                    endi = (int)i;
                    }
                else
                    {
                    endi = (int)i + 1;
                    }
                if (m_celldim.y < 2)
                    {
                    endj = (int)j;
                    }
                else
                    {
                    endj = (int)j + 1;
                    }
                if (m_celldim.z < 2)
                    {
                    endk = (int)k;
                    }
                else
                    {
                    endk = (int)k + 1;
                    }
                if (m_box.is2D())
                    startk = endk = k;

                for (int neighk = startk; neighk <= endk; neighk++)
                    for (int neighj = startj; neighj <= endj; neighj++)
                        for (int neighi = starti; neighi <= endi; neighi++)
                            {
                            // wrap back into the box
                            int wrapi = (m_cell_index.getW()+neighi) % m_cell_index.getW();
                            int wrapj = (m_cell_index.getH()+neighj) % m_cell_index.getH();
                            int wrapk = (m_cell_index.getD()+neighk) % m_cell_index.getD();

                            unsigned int neigh_cell = m_cell_index(wrapi, wrapj, wrapk);
                            // add to the list
                            m_cell_neighbors[cur_cell].push_back(neigh_cell);
                            }

                // sort the list
                sort(m_cell_neighbors[cur_cell].begin(), m_cell_neighbors[cur_cell].end());
                }
    }

void export_LinkCell()
    {
    class_<LinkCell>("LinkCell", init<trajectory::Box&, float>())
        .def("getBox", &LinkCell::getBox, return_internal_reference<>())
        .def("getCellIndexer", &LinkCell::getCellIndexer, return_internal_reference<>())
        .def("getNumCells", &LinkCell::getNumCells)
        .def("getCell", &LinkCell::getCellPy)
        //.def("getCellCoord", &LinkCell::getCellCoordPy)
        .def("itercell", &LinkCell::itercell)
        .def("getCellNeighbors", &LinkCell::getCellNeighborsPy)
        .def("computeCellList", &LinkCell::computeCellListPy)
        ;

    class_<IteratorLinkCell>("IteratorLinkCell",
        init<boost::shared_array<unsigned int>, unsigned int, unsigned int, unsigned int>())
        .def("next", &IteratorLinkCell::nextPy) //PYthon 2 iterator
        .def("__next__", &IteratorLinkCell::nextPy) //Python3 iterator
        ;
    }

}; }; // end namespace freud::locality
