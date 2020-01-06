import setuptools

# default required packages
install_requires = ['hyperopt', 'scikit-surprise']

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto-surprise",
    packages="auto-surprise",
    version="0.0.1",
    author="Rohan Anand",
    author_email="anandr@tcd.ie",
    description="A python package that automates machine learning for the recommender system library Surprise",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thededlier/auto-surprise",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
)
