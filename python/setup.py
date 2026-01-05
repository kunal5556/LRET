"""Setup script for QLRET Python package.

Installation:
    pip install .                      # Pure Python (subprocess backend)
    pip install .[pennylane]           # With PennyLane device support
    
Native bindings (optional):
    cd build && cmake -DUSE_PYTHON=ON .. && cmake --build .
    # This builds _qlret_native.so directly into python/qlret/
"""

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent.resolve()
readme_path = here.parent / "README.md"
try:
    long_description = readme_path.read_text(encoding="utf-8")
except FileNotFoundError:
    long_description = "QLRET - Low-Rank Exact Tensor Quantum Simulator"

setup(
    name="qlret",
    version="1.0.0",
    description="QLRET - Low-Rank Exact Tensor Quantum Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="QLRET Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "pennylane": ["pennylane>=0.30"],
        "jax": ["jax", "jaxlib"],
        "torch": ["torch"],
        "ml": ["jax", "jaxlib", "torch"],
        "dev": [
            "pytest>=7.0",
            "pennylane>=0.30",
            "jax",
            "jaxlib",
            "torch",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    entry_points={
        "pennylane.plugins": [
            "qlret = qlret.pennylane_device:QLRETDevice",
        ],
    },
)
