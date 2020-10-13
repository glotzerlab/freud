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

if (WIN32)
    target_compile_definitions(
        libfreud
        # Export API - required for static data members to have external
        # linkage with MSVC. This is in addition to
        # CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS.
        PRIVATE "LIBFREUD_EXPORTS"
        )
endif()

target_link_libraries(
    libfreud
    PUBLIC ${TBB_LIBRARY}
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
