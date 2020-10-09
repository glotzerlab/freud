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

install(TARGETS box locality DESTINATION freud)
