# Need to figure out coverage before CYTHON_FLAGS are set.
option(COVERAGE "Enable coverage" OFF)

find_package(PythonLibs)
find_package(PythonExtensions REQUIRED)
find_package(NumPy REQUIRED)

include_directories(${NumPy_INCLUDE_DIRS})

# Avoid Cython/Python3.8 minor incompatibility warnings, see
# https://github.com/cython/cython/issues/3474. Note that this option is a bit
# expansive, but it's a temporary fix and we'll be testing on other Python
# versions concurrently so it shouldn't hide any real issues. For backwards
# compatibility with older CMake, I'm using PythonInterp; when we drop support
# for CMake < 3.12, we should switch to find_package(Python).
# find_package(PythonInterp REQUIRED) if(${PYTHON_VERSION_MAJOR} EQUAL 3 AND
# ${PYTHON_VERSION_MINOR} EQUAL 8 AND NOT WIN32)
# add_compile_options("-Wno-deprecated-declarations") endif()

set(python_files box.py util.py)
# cluster.py density.py diffraction.py environment.py locality.py order.py
# parallel.py pmft.py)

install(FILES ${python_files} DESTINATION freud)

# THIS WILL NEED TO BE MOVED TO THE BUILD OF THE ORDER MODULE The SolidLiquid
# class has an instance of cluster::Cluster as a member, so including the header
# requires the Cluster.h header. Would prefer to inherit this information from
# the _order library, but that's not possible since we're linking to libfreud.
# target_include_directories(order PUBLIC ${PROJECT_SOURCE_DIR}/cpp/cluster)
