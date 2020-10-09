find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)

include_directories(${NumPy_INCLUDE_DIRS})

add_cython_target(box CXX)
add_library(box MODULE ${box})
python_extension_module(box)

target_link_libraries(box TBB)

add_cython_target(locality CXX)
add_library(locality MODULE ${locality})
python_extension_module(locality)

target_link_libraries(locality TBB)
target_link_libraries(locality _locality)

add_cython_target(cluster CXX)
add_library(cluster MODULE ${cluster})
python_extension_module(cluster)

include_directories(
	${CMAKE_SOURCE_DIR}/cpp/cluster
)
target_link_libraries(cluster _locality)
target_link_libraries(cluster _cluster)

install(TARGETS box cluster locality DESTINATION freud)
