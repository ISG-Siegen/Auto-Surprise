"""
When updating version

- Update version and download URL here
- Update release version in Docs configuration
"""

import setuptools

install_requires = [
    "hyperopt>=0.2.5",
    "numpy",
    "scikit-learn>=0.23.0",
    "scikit-surprise",
    "rich"
]

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("auto_surprise/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

setuptools.setup(
    name="auto-surprise",
    version=version,
    author="Rohan Anand",
    author_email="anandr@tcd.ie",
    description="A python package that automates algorithm selection and hyperparameter tuning for the recommender system library Surprise",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BeelGroup/Auto-Surprise",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
)
