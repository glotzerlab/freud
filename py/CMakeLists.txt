# install the hoomd_analyze python module
install(DIRECTORY hoomd_analyze
    DESTINATION "."
    PATTERN "*.pyc" EXCLUDE
    PATTERN "*.pyo" EXCLUDE
    )
