# Copyright (c) 2010-2023 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import os

from skbuild import setup

version = "2.13.1"

# Read README for PyPI, fallback to short description if it fails.
description = "Powerful, efficient trajectory analysis in scientific Python."
try:
    readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.rst")
    with open(readme_file) as f:
        readme = f.read()
except ImportError:
    readme = description


setup(
    name="freud-analysis",
    version=version,
    packages=["freud"],
    description=description,
    long_description=readme,
    long_description_content_type="text/x-rst",
    keywords=(
        "simulation analysis molecular dynamics soft matter "
        "particle system computational physics"
    ),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
    maintainer="freud Developers",
    maintainer_email="freud-developers@umich.edu",
    # See documentation credits for current and former lead developers
    author="Vyas Ramasubramani et al.",
    author_email="vramasub@umich.edu",
    url="https://github.com/glotzerlab/freud",
    download_url="https://pypi.org/project/freud-analysis/",
    project_urls={
        "Homepage": "https://github.com/glotzerlab/freud",
        "Documentation": "https://freud.readthedocs.io/",
        "Source Code": "https://github.com/glotzerlab/freud",
        "Issue Tracker": "https://github.com/glotzerlab/freud/issues",
    },
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.14",
        "rowan>=1.2.1",
        "scipy>=1.1",
    ],
    tests_require=[
        "ase>=3.16",
        "gsd>=2.0",
        "matplotlib>=3.0",
        "sympy>=1.0",
    ],
)
