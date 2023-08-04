from setuptools import setup, find_packages

setup(
    author="Samuel JS Wright",
    description="Functions for clr transforms and PCA and plotting",
    name="clrutils",
    version="0.1.0",
    packages=find_packages(include=["clrutils", "clrutils.*"]),
    install_requires=[
        "pandas >= 1.5.2",
        "numpy >= 1.23.5",
        "matplotlib >= 3.7.1",
        "seaborn >= 0.12.2",
        "scikit-learn >= 1.2.0",
    ],
    python_requires=">=3.9",
)
