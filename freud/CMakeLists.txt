find_package(PythonExtensions REQUIRED)

# Cython flags must be set before we run find_package for Cython since the
# compiler command is created immediately.
set(CYTHON_FLAGS
    "--directive binding=True,boundscheck=False,wraparound=False,embedsignature=True,always_allow_keywords=True"
    CACHE STRING "The directives for Cython compilation."
    )

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
        # Since almost all modules use TBB, linking all of them is easier.
        target_link_libraries(${cython_module} TBB)

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

    install(TARGETS ${cython_module} DESTINATION freud)
endforeach(cython_module)

# Any additional module-specific libraries.

# Not sure why this is necessary; I'm requesting a public link to _diagonalize
# in _environment and _order, so it should propagate up. Something to check
# later. It might be because _environment and _order are object libraries, so
# these settings don't actually propagate meaningfully.
target_link_libraries(environment _diagonalize)
target_link_libraries(order _diagonalize)

target_link_libraries(order _cluster)
