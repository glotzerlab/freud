# install the hoomd_analyze python module
install(DIRECTORY freud
    DESTINATION "."
    PATTERN "*.pyc" EXCLUDE
    PATTERN "*.pyo" EXCLUDE
    )
