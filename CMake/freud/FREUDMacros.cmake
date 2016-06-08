macro(fix_tbb_rpath target)
if (APPLE)
get_target_property(_target_exe ${target} LOCATION)
add_custom_command(TARGET ${target} POST_BUILD
                          COMMAND install_name_tool ARGS -change @rpath/libtbb.dylib ${TBB_LIBRARY} ${_target_exe})
endif (ENABLE_CUDA AND APPLE)
endmacro(fix_tbb_rpath)
