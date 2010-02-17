#include <stdexcept>
#include <boost/python.hpp>

#include "num_util.h"
#include "LinkCell.h"

using namespace std;
using namespace boost::python;

LinkCell::LinkCell(const Box& box, float cell_width) : m_box(box), m_Np(0)
    {
    if (cell_width > m_box.getLx() ||
        cell_width > m_box.getLy() ||
        cell_width > m_box.getLz())
        {
        throw runtime_error("Cannot generate a cell list where cell_width is larger than a box dimension");
        }
    
    unsigned int Nx = (unsigned int)(m_box.getLx() / cell_width);
    unsigned int Ny = (unsigned int)(m_box.getLy() / cell_width);
    unsigned int Nz = (unsigned int)(m_box.getLz() / cell_width);
    
    m_cell_index = Index3D(Nx, Ny, Nz);
    }

void LinkCell::computeCellListPy(boost::python::numeric::array x,
                                 boost::python::numeric::array y,
                                 boost::python::numeric::array z)
    {
    // validate input type and rank
    num_util::check_type(x, PyArray_FLOAT);
    num_util::check_rank(x, 1);
    num_util::check_type(y, PyArray_FLOAT);
    num_util::check_rank(y, 1);
    num_util::check_type(z, PyArray_FLOAT);
    num_util::check_rank(z, 1);
    
    // validate all inputs are the same size
    unsigned int Np = num_util::size(x);
    num_util::check_size(y, Np);
    num_util::check_size(z, Np);
    
    // get the raw data pointers and compute the cell list
    float* x_raw = (float*) num_util::data(x);
    float* y_raw = (float*) num_util::data(y);
    float* z_raw = (float*) num_util::data(z);
    
    computeCellList(x_raw, y_raw, z_raw, Np);
    }

void LinkCell::computeCellList(float *x, float *y, float *z, unsigned int Np)
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
    assert(x);
    assert(y);
    assert(z);
    
    for (unsigned int i = 0; i < Np; i++)
        {
        unsigned int cell = getCell(x[i], y[i], z[i]);
        m_cell_list[i] = m_cell_list[Np+cell];
        m_cell_list[Np+cell] = i;
        }
    }

void export_LinkCell()
    {
    class_<LinkCell>("LinkCell", init<Box&, float>())
        .def("getBox", &LinkCell::getBox, return_internal_reference<>())
        .def("getCellIndexer", &LinkCell::getCellIndexer, return_internal_reference<>())
        .def("getNumCells", &LinkCell::getNumCells)
        .def("getCell", &LinkCell::getCell)
        .def("getCellCoord", &LinkCell::getCellCoord)
        .def("itercell", &LinkCell::itercell)
        .def("computeCellList", &LinkCell::computeCellListPy)
        ;
    
    class_<IteratorLinkCell>("IteratorLinkCell",
        init<boost::shared_array<unsigned int>, unsigned int, unsigned int, unsigned int>())
        .def("next", &IteratorLinkCell::nextPy)
        ;
    }
