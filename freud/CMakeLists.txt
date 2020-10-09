find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)

include_directories(${NumPy_INCLUDE_DIRS})

# Construct all Cython extension modules.
set(cython_modules
    box
    cluster
    density
    environment
    pmft
    locality
    )

foreach(cython_module ${cython_modules})
    add_cython_target(${cython_module} CXX)
    add_library(${cython_module} MODULE ${${cython_module}})
    python_extension_module(${cython_module})

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
    install(TARGETS ${cython_module} DESTINATION freud)
endforeach(cython_module)

# Any additional module-specific libraries.

# Not sure why this is necessary; I'm requesting a public link to _diagonalize
# in _environment, so it should propagate up to environment. Something to check
# later. It might be because _environment is an object library, so these
# settings don't actually propagate meaningfully.
target_link_libraries(environment _diagonalize)
