"""Setup script for litert-torch-qarnux package."""

from setuptools import setup, find_packages

setup(
    name="litert-torch-qarnux",
    version="1.0.0",
    packages=find_packages(include=["litert_torch_qarnux*"]),
    python_requires=">=3.10",
)
