from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="opac_mixer",
    version="v1.0",
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/aarondavidschneider/opac_mixer",
    license="MIT",
    author="Aaron David Schneider",
    author_email="aaron.schneider@nbi.ku.dk",
    description="opacity mixing - accelerated",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "scikit-learn",
        "numba",
        "scipy",
        "numpy",
        "matplotlib",
        "tqdm",
        "h5py",
        "tensorflow",
        "MITgcmutils",
    ],
)
