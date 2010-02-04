######################
# Perform setup to make a target into a functional python module
macro(setup_pymodule target)
set_target_properties(${target} PROPERTIES PREFIX "")
target_link_libraries(${target} ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARIES})
endmacro(setup_pymodule)
#####################

add_library(_trajectory MODULE trajectory.h trajectory.cc num_util.cc)
setup_pymodule(_trajectory)
