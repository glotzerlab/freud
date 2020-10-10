add_library(
    _util
    OBJECT
    diagonalize.h
    diagonalize.cc
    )
target_include_directories(
    _util
    PUBLIC ${PROJECT_SOURCE_DIR}/extern/
    )
