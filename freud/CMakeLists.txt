find_package(PythonExtensions REQUIRED)

# Need to figure out coverage before CYTHON_FLAGS are set.
option(COVERAGE "Enable coverage" OFF)

# Cython flags must be set before we run find_package for Cython since the
# compiler command is created immediately.
set(CYTHON_FLAGS
    "--directive binding=True,boundscheck=False,wraparound=False,embedsignature=True,always_allow_keywords=True"
    CACHE STRING "The directives for Cython compilation."
    )

if (COVERAGE)
    set(CYTHON_FLAGS
        "${CYTHON_FLAGS},linetrace=True"
        CACHE STRING "The directives for Cython compilation."
        )
endif()

find_package(Cython REQUIRED)

find_package(NumPy REQUIRED)
include_directories(${NumPy_INCLUDE_DIRS})

# Avoid Cython/Python3.8 minor incompatibility warnings, see
# https://github.com/cython/cython/issues/3474. Note that this option is a bit
# expansive, but it's a temporary fix and we'll be testing on other Python
# versions concurrently so it shouldn't hide any real issues.
find_package(Python REQUIRED)
if (${Python_VERSION_MAJOR} EQUAL 3 AND ${Python_VERSION_MINOR} EQUAL 8)
    add_compile_options("-Wno-deprecated-declarations")
endif()

if (WIN32)
    # Force Windows to define M_PI in <cmath>
    add_compile_options("_USE_MATH_DEFINES=1")
    # Prevent Windows from defining min/max as macros
    add_compile_options("NOMINMAX=1")
endif()

set(cython_modules_with_cpp
    box
    cluster
    density
    environment
    locality
    order
    parallel
    pmft
    )

set(cython_modules_without_cpp
    diffraction
    interface
    msd
    util
    )

foreach(cython_module ${cython_modules_with_cpp} ${cython_modules_without_cpp})
    add_cython_target(${cython_module} PY3 CXX)
    add_library(${cython_module} MODULE ${${cython_module}})
    # For some reason this command prints the name of the resulting module.
    # Would be good to suppress if possible.
    python_extension_module(${cython_module})

    if ("${cython_module}" IN_LIST cython_modules_with_cpp)
        # Everything links to locality
        target_link_libraries(${cython_module} _locality)

        # Link Cython modules to C++ libraries where appropiate.
        if(NOT ${cython_module} STREQUAL box AND NOT ${cython_module} STREQUAL locality)
            target_link_libraries(${cython_module} _${cython_module})
        endif()

        target_include_directories(
            ${cython_module}
            PUBLIC ${CMAKE_SOURCE_DIR}/cpp/${cython_module}
            )
    endif()

    target_compile_definitions(
        ${cython_module}
        # Avoid deprecation warnings for unsupported NumPy API versions.
        # See https://numpy.org/doc/1.19/reference/c-api/deprecations.html
        PUBLIC "NPY_NO_DEPRECATED_API=NPY_1_10_API_VERSION"
        # Default voro++ verbosity is high.
        PUBLIC "VOROPP_VERBOSE=1"
        )

    if (COVERAGE)
        target_compile_definitions(
            ${cython_module}
            #  Enable line tracing for coverage purposes if requested.
            "CYTHON_TRACE_NOGIL=1"
            )
    endif()

    install(TARGETS ${cython_module} DESTINATION freud)
endforeach(cython_module)

# CMake should allow linking and inheriting linkage from object libraries
# according to
# https://cmake.org/cmake/help/latest/command/target_link_libraries.html#linking-object-libraries
# but that doesn't seem to be working right now because _environment and _order
# are both using target_link_libraries but it's not propagating up. Note that
# TBB linking is being correctly propagated, so I assume it has something to do
# with object library setup (I'm trying to environment to _environment, which
# requires _util; maybe it can't handle two levels of indirection correctly for
# object libraries), but I'm not sure how to fix it right now.  Note that
# because of the dynamic linking of the Python runtime environment, having
# environment linked will technically be sufficient (because the freud library
# will import environment before order), but we can't rely on that detail.
target_link_libraries(environment _util)
target_link_libraries(order _util _cluster)
