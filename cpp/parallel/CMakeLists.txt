add_library(_parallel OBJECT tbb_config.h tbb_config.cc)

target_link_libraries(_parallel PUBLIC TBB::tbb)
