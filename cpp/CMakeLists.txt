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

# CMake will automatically prepend any library name with "lib" when creating
# shared libraries, so a CMake library named libfreud will result in
# liblibfreud.so object. This rename gives the more expected name of
# libfreud.so. Note that we choose not to name the CMake target "freud" to
# avoid conflicting with the project name.
set_target_properties(
    libfreud
    PROPERTIES OUTPUT_NAME freud
    )

# Copy the C++ library into the built version.
install(TARGETS libfreud DESTINATION freud)
