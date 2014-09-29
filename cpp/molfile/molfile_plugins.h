// simple header for accessing the dcd plugins statically compiled in

#ifndef __MOLFILE_PLUGINS_H__
#define __MOLFILE_PLUGINS_H__

#include "vmdplugin.h"
#include "molfile_plugin.h"

extern "C" { 
int molfile_dcdplugin_init();
int molfile_dcdplugin_register(void *, vmdplugin_register_cb);
int molfile_dcdplugin_fini();
}

#endif
