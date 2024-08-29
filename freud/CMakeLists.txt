set(python_files
    __init__.py
    box.py
    data.py
    errors.py
    locality.py
    parallel.py
    pmft.py
    plot.py
    util.py)
# cluster.py density.py diffraction.py environment.py order.py)

install(FILES ${python_files} DESTINATION freud)
