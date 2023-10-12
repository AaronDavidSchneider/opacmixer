.. prt_phasecurve documentation master file, created by
   sphinx-quickstart on Mon Jan 11 14:18:49 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deployment in code
==================

Coupling the DeepSet mixing to a radiative transfer solver is easy to do, if you have already some mixing in your radiative transfer solver.
It requires two steps:

1. Reading in your custom k-tables (see :ref:`Tutorial: Add custom k-tables`) from your radiative transfer solver and training the DeepSet on these (see :ref:`Tutorial: Quick Start`).

2. Implementing the inference of the neural network in your code

Here, we will talk about the second step, since the first step is defined at other places in this documentation.

Numpy implementation
--------------------

First, lets start with a simple numpy implementation of the DeepSet:

.. code-block::
   python

   mlp_weights = [weights.numpy() for weights in em.model.weights]
   def simple_mlp(kappas):
       rep = np.tensordot(kappas, mlp_weights[0], axes=(1,0))  # first dense
       rep[rep <= 0.0] = 0.0
       sum_rep = np.sum(rep, axis=1)   # sum
       dec = np.tensordot(sum_rep, mlp_weights[1], axes=(-1,0))  # second dense
       return dec

.. Note::

   We use input and output scaling! It is therefore important to feed the scaled input to the function and to transform the output back using the inverse output scaling.

General remarks
---------------

The exact deployment deployment depends on the code, language and framework you want to couple it to.
Some general recommendations can be found in the paper (Appendix).

Basically, the following things are always needed:

0. (Implement the read in and interpolation of the individual k-tables and the chemistry in your model)

1. Export the weights (there is an ``export`` function on the ``Emulator`` class)

2. Read the weights into your model

3. Implement the scaling functions (see ``opac_mixer/utils/scaling.py``)

4. Implement the DeepSet mixing consisting of:

   - A first matrix vector multiplication for each of the individual species

   - A relu activation

   - A sum over all hidden representations

   - Another matrix vector multiplication

.. Note::

   It would generally be the fastest option to stack the matrix vector multiplications and deploy them using a vectorized matrix vector multiplication. Keep that in mind.