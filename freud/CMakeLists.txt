find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)

# Currently using FindTBB from HOOMD.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_package(TBB REQUIRED)

include_directories(${TBB_INCLUDE_DIR})

add_cython_target(locality CXX)
add_library(locality MODULE ${locality})
python_extension_module(locality)

target_link_libraries(locality TBB)
target_link_libraries(locality _locality)

install(TARGETS locality DESTINATION freud)
