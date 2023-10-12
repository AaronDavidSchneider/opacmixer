.. prt_phasecurve documentation master file, created by
   sphinx-quickstart on Mon Jan 11 14:18:49 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to opac_mixer's documentation!
==========================================
There is a pressing need for fast opacity mixing in the correlated-k approximation.
This python package builds a framework for machine learning on grids of k-tables.

To get the most out of this code, start by installing the code (see :ref:`Installation`).
Once installed, head over to :ref:`Tutorial: Quick Start` to see a quick explanation of how to use the code.

Finally, if you want to couple the code to your own radiative transfer, you might need to read in your own k-tables.
Thats easy, and its explained in :ref:`Tutorial: Add custom k-tables`. You also need to write a short code passage to deploy the mixing in the radiative transfer solver (see :ref:`Deployment in code`).

If you find this work useful, please consider to cite the following paper:
Schneider et al. (in review)

WARNING: The paper is not yet published, please do not use this code, before it is published

.. toctree::
   :maxdepth: 0
   :caption: Contents

   Installation
   notebooks/training.ipynb
   Coupling
   notebooks/extend_reader.ipynb
   API
   notebooks/petitRADTRANS.ipynb
   notebooks/hp_tuning.ipynb

Copyright 2023 Aaron Schneider. Feel free to contact me for any questions via `mail <mailto:Aaron.Schneider@nbi.ku.dk>`_.
