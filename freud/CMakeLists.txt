find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)

include_directories(${NumPy_INCLUDE_DIRS})

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
