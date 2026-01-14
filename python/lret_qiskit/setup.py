"""Setup script for LRET Qiskit integration package.

Installation:
    pip install .                # Basic installation
    pip install .[dev]           # With development dependencies
    
This package provides Qiskit backend integration for the LRET
low-rank quantum simulator. It requires the qlret package to be
installed for actual simulation.
"""

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent.resolve()
readme_path = here / "README.md"
try:
    long_description = readme_path.read_text(encoding="utf-8")
except FileNotFoundError:
    long_description = "LRET Qiskit Backend - Low-Rank Quantum Simulator for Qiskit"

# Read version from version.py
version_dict = {}
exec((here / "version.py").read_text(encoding="utf-8"), version_dict)

setup(
    name="lret-qiskit",
    version=version_dict["__version__"],
    description="LRET Qiskit Backend - Low-Rank Quantum Simulator for Qiskit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LRET Development Team",
    license="Apache 2.0",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "qiskit>=1.0",
        "numpy>=1.20",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
