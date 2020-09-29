# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

cmake_minimum_required(VERSION 3.16)

project(freud VERSION 2.3.0 LANGUAGES C CXX)

set(DEFAULT_BUILD_TYPE "Release")

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})

# Enable warnings for gcc and clang
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

# Enable color output from compiler
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=always")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcolor-diagnostics")
endif()

# Export compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# External C++ dependencies
find_package(TBB REQUIRED)
#add_subdirectory(extern)
include_directories("extern")
#include_directories("${EIGEN_INCLUDE_DIR}"
#                    "${DSET_INCLUDE_DIR}"
#                    "${FSPH_INCLUDE_DIR}"
#                    "${VOROPP_INCLUDE_DIR}")

# Python / Cython dependencies
#find_package(Python3 COMPONENTS NumPy REQUIRED)
#message("${Python3_NumPy_INCLUDE_DIRS}")

# C++ includes
include_directories("${CMAKE_SOURCE_DIR}/cpp"
                    "${CMAKE_SOURCE_DIR}/cpp/box"
                    "${CMAKE_SOURCE_DIR}/cpp/cluster"
                    "${CMAKE_SOURCE_DIR}/cpp/density"
                    "${CMAKE_SOURCE_DIR}/cpp/diffraction"
                    "${CMAKE_SOURCE_DIR}/cpp/environment"
                    "${CMAKE_SOURCE_DIR}/cpp/locality"
                    "${CMAKE_SOURCE_DIR}/cpp/order"
                    "${CMAKE_SOURCE_DIR}/cpp/parallel"
                    "${CMAKE_SOURCE_DIR}/cpp/pmft"
                    "${CMAKE_SOURCE_DIR}/cpp/util")

# Define library
add_library(freud SHARED
            cpp/cluster/Cluster.cc
            cpp/cluster/ClusterProperties.cc
            cpp/density/CorrelationFunction.cc
            cpp/density/GaussianDensity.cc
            cpp/density/LocalDensity.cc
            cpp/density/RDF.cc
            cpp/density/SphereVoxelization.cc
            cpp/diffraction/StructureFactor.cc
            cpp/environment/AngularSeparation.cc
            cpp/environment/BondOrder.cc
            cpp/environment/LocalBondProjection.cc
            cpp/environment/LocalDescriptors.cc
            cpp/environment/MatchEnv.cc
            cpp/locality/AABBQuery.cc
            cpp/locality/LinkCell.cc
            cpp/locality/NeighborComputeFunctional.cc
            cpp/locality/NeighborList.cc
            cpp/locality/NeighborPerPointIterator.cc
            cpp/locality/NeighborQuery.cc
            cpp/locality/PeriodicBuffer.cc
            cpp/locality/Voronoi.cc
            cpp/order/Cubatic.cc
            cpp/order/HexaticTranslational.cc
            cpp/order/Nematic.cc
            cpp/order/RotationalAutocorrelation.cc
            cpp/order/SolidLiquid.cc
            cpp/order/Steinhardt.cc
            cpp/order/Wigner3j.cc
            cpp/parallel/tbb_config.cc
            cpp/pmft/PMFTR12.cc
            cpp/pmft/PMFTXY.cc
            cpp/pmft/PMFTXYT.cc
            cpp/pmft/PMFTXYZ.cc
            cpp/util/diagonalize.cc
)

# Link freud against its dependencies
target_link_libraries(freud tbb)

# TODO: Perhaps not necessary?
#if (APPLE)
#set_target_properties(freud PROPERTIES INSTALL_RPATH "@loader_path")
#else()
#set_target_properties(freud PROPERTIES INSTALL_RPATH "\$ORIGIN")
#endif()

# Install targets for freud
install(TARGETS freud
        DESTINATION lib
        COMPONENT freud)

add_custom_target(install_freud
                  COMMAND "${CMAKE_COMMAND}" -DCOMPONENT=freud -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
                  DEPENDS freud)
