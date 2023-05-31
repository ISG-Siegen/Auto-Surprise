import setuptools

with open("requirements.txt", "r") as fh:
    install_requires =[line.strip() for line in fh.read().split("\n")]

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
    url="https://github.com/ISG-Siegen/Auto-Surprise",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
)
