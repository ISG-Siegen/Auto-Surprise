import setuptools

install_requires = [
    "hyperopt",
    "lightgbm",
    "numpy",
    "scikit-learn==0.22.0",
    "scikit-surprise"
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto-surprise",
    version="0.1.4",
    author="Rohan Anand",
    author_email="anandr@tcd.ie",
    description="A python package that automates algorithm selection and hyperparameter tuning for the recommender system library Surprise",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BeelGroup/Auto-Surprise",
    download_url="https://github.com/BeelGroup/Auto-Surprise/archive/v0.1.4.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
)
