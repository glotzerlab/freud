#include "DCDLoader.h"

#include <stdexcept>

using namespace boost::python;
using namespace std;

//! Simple plugin register takes a ** to the pointer to set
static int plugin_register(void *p, vmdplugin_t *plugin)
	{
	vmdplugin_t **plugin_ptr = (vmdplugin_t **)p;
	*plugin_ptr = plugin;
	return VMDPLUGIN_SUCCESS;
	}

/*! \param dcd_fname DCD file to read timestep data from
*/
DCDLoader::DCDLoader(const std::string &dcd_fname) : m_fname(dcd_fname), m_points(num_util::makeNum(1, PyArray_FLOAT))
	{
	// initialize the plugins
	molfile_dcdplugin_init();
	molfile_dcdplugin_register((void**)&dcdplugin, &plugin_register);
	
	// initialize the dcd file handle
	loadDCD();
	}
	
DCDLoader::~DCDLoader()
	{
	// close the dcd file
	dcdplugin->close_file_read((void *)m_dcd);
	}

		
/*! \param frame Frame number to jump to
	The molfile plugins only support skipping forward in the file
	As such, jumpToFrame() must reload the file from scratch if given a previous frame number than the current
*/
void DCDLoader::jumpToFrame(int frame)
	{
    assert(m_dcd);
    
	// figure out where we are in the file:
	int cur_frame = m_dcd->setsread;
	if (cur_frame < frame)
		{
        // cout << "Warning, rewinding DCD file. Expect slow performance" << endl;
        // close and reopen the file to get back to frame 0
        dcdplugin->close_file_read((void *)m_dcd);
        loadDCD();
        cur_frame = 0;
		}
	
    assert(m_dcd);
    
	// calculate the number of steps to make to get to the requested value
	int nskips = frame - cur_frame;
	if (m_dcd->setsread + nskips >= m_dcd->nsets)
		throw runtime_error("DCDLoader::jumpToFrame: asked to read past the end of the file");
		
	// everything checks out, perform the skips
	for (int i = 0; i < nskips; i++)
		{
		int err = dcdplugin->read_next_timestep((void *)m_dcd, m_dcd->natoms, NULL);
		if (err != MOLFILE_SUCCESS)
			{
			throw runtime_error("DCDLoader::jumpToFrame: unexpected error skipping time steps");
			}
		}
	}


/*! Reads the next frame from the DCD file
*/
void DCDLoader::readNextFrame()
	{
	// read the next timestep
	molfile_timestep_t ts;
	ts.coords = (float*)num_util::data(m_points);
    
	int err = dcdplugin->read_next_timestep((void *)m_dcd, m_dcd->natoms, &ts);
	if (err == MOLFILE_EOF)
		{
        throw runtime_error("Reached end of file while reading DCD file");
		}
	if (err != MOLFILE_SUCCESS)
		{
		throw runtime_error("Unknown error while reading DCD file");
		}
    
    // record the box read from this time step
    m_box = Box(ts.A, ts.B, ts.C);
	}
		
/*! Initializes the DCD handle
*/
void DCDLoader::loadDCD()
	{
	int natoms;
	m_dcd = (dcdhandle*)dcdplugin->open_file_read(m_fname.c_str(), "dcd", &natoms);
	if (m_dcd == NULL)
		throw runtime_error("Error loading dcd file");
    
    // allocate the memory for the points
    std::vector<intp> dims(2);
    dims[0] = natoms;
    dims[1] = 3;
    
    m_points = num_util::makeNum(dims, PyArray_FLOAT);
	}		

void export_dcdloader()
	{
	class_<DCDLoader>("DCDLoader", init<const string &>())
		.def("jumpToFrame", &DCDLoader::jumpToFrame)
		.def("readNextFrame", &DCDLoader::readNextFrame)
        .def("getPoints", &DCDLoader::getPoints)
        .def("getBox", &DCDLoader::getBox, return_internal_reference<>())
        .def("getNumParticles", &DCDLoader::getNumParticles)
        .def("getLastFrameNum", &DCDLoader::getLastFrameNum)
        .def("getFrameCount", &DCDLoader::getFrameCount)
		;
	}
