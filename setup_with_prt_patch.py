import setuptools

class NoNumpy(Exception):
    pass

try:
    from numpy.distutils.core import Extension
    from numpy.distutils.core import setup
except ImportError:
    raise NoNumpy('Numpy Needs to be installed '
                  'for extensions to be compiled.')


with open("README.md", "r") as fh:
    long_description = fh.read()

fort_bol_flux = Extension('opac_mixer.fort_bol_flux', sources=['opac_mixer/patches/fort_bol_flux.f90'],
                      extra_compile_args=["-O3", "-funroll-loops", "-ftree-vectorize", "-msse", "-msse2", "-m3dnow"])

setup(
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
    ext_modules=[fort_bol_flux],
    install_requires=[
        "scikit-learn",
        "numba",
        "scipy",
        "numpy",
        "matplotlib",
        "tqdm",
        "h5py",
        "tensorflow",
        "MITgcmutils"
    ]
)