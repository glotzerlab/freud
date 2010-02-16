#include <stdexcept>
using namespace std;

#include "LinkCell.h"

LinkCell::LinkCell(const Box& box, float cell_width) : m_box(box)
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

void LinkCell::computeCellList(float *x, float *y, float *z, unsigned int Np)
    {
    if (Np == 0)
        {
        throw runtime_error("Cannot generate a cell list of 0 particles");
        }
    
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
