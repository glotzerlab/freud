# create the target and add the needed properties
pybind11_add_module(box SHARED module-box.cc)
# this probably isn't needed for box, but may be needed by other modules
#target_compile_definitions(
#    box
#    # Avoid deprecation warnings for unsupported NumPy API versions. See
#    # https://numpy.org/doc/1.19/reference/c-api/deprecations.html
#    PRIVATE "NPY_NO_DEPRECATED_API=NPY_1_10_API_VERSION"
#    # Default voro++ verbosity is high.
#    PRIVATE "VOROPP_VERBOSE=1")

# install
install(TARGETS box DESTINATION freud)
if(APPLE)
    set_target_properties(box PROPERTIES INSTALL_RPATH "@loader_path")
else()
    set_target_properties(box PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

if(_using_conda OR DEFINED ENV{CIBUILDWHEEL})
    set_target_properties(box
                          PROPERTIES INSTALL_RPATH_USE_LINK_PATH True)
endif()
