import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='opac_mixer',
    version='v0.1',
    packages=setuptools.find_packages(),
    include_package_data=True,
    url='https://github.com/aarondavidschneider/opac_mixer',
    license='MIT',
    author='Aaron David Schneider',
    author_email='aaron.schneider@nbi.ku.dk',
    description='opacity mixing - accelerated',
    long_description=long_description,
    long_description_content_type="text/markdown",
    # scripts=['scripts/chemcomp_main', 'scripts/chemcomp_pipeline', 'scripts/chemcomp_pipeline_slurm'],
    install_requires=[
        "scikit-learn",
        "numba",
        "xgboost",
        "hyperopt",
        "scipy",
        "numpy",
        "matplotlib",
        "pyyaml",
        "tqdm"
    ]
)