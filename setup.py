
from setuptools import setup, find_packages

setup(
    name="ARX",
    version="0.1.0",
    author="P.L.Green",
    description="Auto-regressive machine learning with exogeneous inputs",
    packages=find_packages(),
    install_requires=[
        "numpy","scikit-learn"
    ],
)
