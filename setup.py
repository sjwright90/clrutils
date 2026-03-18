from setuptools import setup, find_packages

setup(
    author="Samuel JS Wright",
    description="Functions for clr transforms and PCA and plotting",
    name="clrutils",
    version="0.1.0",
    packages=find_packages(include=["clrutils", "clrutils.*"]),
    install_requires=[
        "pandas >= 1.5",
        "numpy >= 1.22",
        "matplotlib >= 3.6",  # changed from 3.7 if this breaks change back
        "seaborn >= 0.12",
        "scikit-learn >= 1.2",
        "plotly >= 5.0",
    ],
    python_requires=">=3.9",
)
