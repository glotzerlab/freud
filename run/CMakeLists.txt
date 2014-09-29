# find the module build dir for development use
get_target_property(_freud_location _freud LOCATION)
get_filename_component(FREUD_MODULE_BUILD_DIR ${_freud_location} PATH)

# install the set build path script
configure_file(set_build_path.sh.in
                ${CMAKE_CURRENT_BINARY_DIR}/set_build_path.sh
                @ONLY
                )

