if (WIN32)
    # Use add_compile_definitions when dropping support for CMake < 3.12
    # Force Windows to define M_PI in <cmath>
    add_compile_options(/D_USE_MATH_DEFINES)
    # Prevent Windows from defining min/max as macros
    add_compile_options(/DNOMINMAX)
endif()

add_subdirectory(cluster)
add_subdirectory(density)
add_subdirectory(environment)
add_subdirectory(locality)
add_subdirectory(order)
add_subdirectory(parallel)
add_subdirectory(pmft)
add_subdirectory(util)

add_library(
    libfreud
    SHARED
    $<TARGET_OBJECTS:_cluster>
    $<TARGET_OBJECTS:_density>
    $<TARGET_OBJECTS:_environment>
    $<TARGET_OBJECTS:_locality>
    $<TARGET_OBJECTS:_order>
    $<TARGET_OBJECTS:_parallel>
    $<TARGET_OBJECTS:_pmft>
    $<TARGET_OBJECTS:_util>)

target_link_libraries(
    libfreud
    PUBLIC TBB
    )

target_include_directories(
    libfreud
    PUBLIC ${PROJECT_SOURCE_DIR}/extern/
    )

# Set the name of the library to just freud.
set_target_properties(
    libfreud
    PROPERTIES OUTPUT_NAME freud
    )

# Copy the C++ library into the built version.
install(TARGETS libfreud DESTINATION freud)
