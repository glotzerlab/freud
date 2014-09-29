#include <boost/python.hpp>
#include <stdexcept>
#include <algorithm>

#include "num_util.h"
#include "LinkCell.h"
#include "ScopedGILRelease.h"

using namespace std;
using namespace boost::python;

/*! \file LinkCell.cc
    \brief Build a cell list from a set of points
*/

namespace freud { namespace locality {

LinkCell::LinkCell(const trajectory::Box& box, float cell_width) : m_box(box), m_Np(0)
    {
    // check if the cell width is too wide for the box
    bool too_wide = cell_width > 1.0f/3.0f * m_box.getLx() || cell_width > 1.0f/3.0f * m_box.getLy();
    if (!m_box.is2D())
        too_wide |= cell_width > 1.0f/3.0f * m_box.getLz();
    if (too_wide)
        {
        throw runtime_error("Cannot generate a cell list where cell_width is larger than 1/3 of any box dimension");
        }

    unsigned int Nx = (unsigned int)(m_box.getLx() / cell_width);
    unsigned int Ny = (unsigned int)(m_box.getLy() / cell_width);
    unsigned int Nz = (unsigned int)(m_box.getLz() / cell_width);

    // only 1 cell deep in 2D
    if (m_box.is2D())
        Nz = 1;

    m_cell_index = Index3D(Nx, Ny, Nz);
    computeCellNeighbors();
    }

void LinkCell::computeCellListPy(boost::python::numeric::array points)
    {
    // validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute the cell list
    // float3* points_raw = (float3*) num_util::data(points);
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);

        // compute the cell list with the GIL released
        {
        util::ScopedGILRelease gil;
        computeCellList(points_raw, Np);
        }
    }

// void LinkCell::computeCellList(const float3 *points, unsigned int Np)
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

                // loop over the 27 neighbor cells (9 in 2d)
                int startk = (int)k-1;
                int endk = (int)k+1;
                if (m_box.is2D())
                    startk = endk = k;

                for (int neighk = startk; neighk <= endk; neighk++)
                    for (int neighj = (int)j-1; neighj <= (int)j+1; neighj++)
                        for (int neighi = (int)i-1; neighi <= (int)i+1; neighi++)
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
        .def("getCellCoord", &LinkCell::getCellCoord)
        .def("itercell", &LinkCell::itercell)
        .def("getCellNeighbors", &LinkCell::getCellNeighborsPy)
        .def("computeCellList", &LinkCell::computeCellListPy)
        ;

    class_<IteratorLinkCell>("IteratorLinkCell",
        init<boost::shared_array<unsigned int>, unsigned int, unsigned int, unsigned int>())
        .def("next", &IteratorLinkCell::nextPy)     // python 2 iteration
        .def("__next__", &IteratorLinkCell::nextPy) // python 3 iteration
        ;
    }

}; }; // end namespace freud::locality
