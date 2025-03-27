
from setuptools import setup, find_packages

setup(
    name="ARXLR",
    version="0.1.0",
    author="P.L.Green",
    description="Auto-regressive linear regression with exogeneous inputs",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
)
