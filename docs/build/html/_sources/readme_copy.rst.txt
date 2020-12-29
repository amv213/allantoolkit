AllanTools
==========

.. image:: https://badge.fury.io/py/AllanTools.svg
    :target: https://badge.fury.io/py/AllanTools
.. image:: https://img.shields.io/conda/vn/conda-forge/allantools.svg
    :target: https://anaconda.org/conda-forge/allantools
.. image:: https://img.shields.io/conda/dn/conda-forge/allantools.svg
    :target: https://anaconda.org/conda-forge/allantools

.. image:: https://travis-ci.org/aewallin/allantools.svg?branch=master
    :target: https://travis-ci.org/aewallin/allantools
.. image:: http://readthedocs.org/projects/allantools/badge/?version=latest
    :target: http://allantools.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://coveralls.io/repos/github/aewallin/allantools/badge.svg?branch=master
    :target: https://coveralls.io/github/aewallin/allantools?branch=master

A python library for calculating Allan deviation and related
time & frequency statistics. `LGPL v3+ license <https://www.gnu.org/licenses/lgpl.html>`_.

* Development at https://github.com/aewallin/allantools
* Installation package at https://pypi.python.org/pypi/AllanTools
* Discussion group at https://groups.google.com/d/forum/allantools
* Documentation available at https://allantools.readthedocs.org


Input data should be evenly spaced observations of either fractional frequency,
or phase in seconds. Deviations are calculated for given tau values in seconds.

=====================================   ====================================================
Function                                Description
=====================================   ====================================================
``adev()``                              Allan deviation
``oadev()``                             Overlapping Allan deviation
``mdev()``                              Modified Allan deviation
``tdev()``                              Time deviation
``hdev()``                              Hadamard deviation
``ohdev()``                             Overlapping Hadamard deviation
``totdev()``                            Total deviation
``mtotdev()``                           Modified total deviation
``ttotdev()``                           Time total deviation
``htotdev()``                           Hadamard total deviation
``theo1()``                             Theo1 deviation
``mtie()``                              Maximum Time Interval Error
``tierms()``                            Time Interval Error RMS
``gradev()``                            Gap resistant overlapping Allan deviation
=====================================   ====================================================

Noise generators for creating synthetic datasets are also included:

* violet noise with f^2 PSD
* white noise with f^0 PSD
* pink noise with f^-1 PSD
* Brownian or random walk noise with f^-2 PSD

More details on available statistics and noise generators : `full list of available functions <functions.html>`_

see /tests for tests that compare allantools output to other
(e.g. Stable32) programs. More test data, benchmarks, ipython notebooks,
and comparisons to known-good algorithms are welcome!


Jupyter notebooks with examples
-------------------------------

Jupyter notebooks are interactive python scripts, embedded in a browser,
allowing you to manipulate data and display plots like easily. For guidance
on installing jupyter, please refer to https://jupyter.org/install.

See /examples for some examples in notebook format.

github formats the notebooks into nice web-pages, for example

* https://github.com/aewallin/allantools/blob/master/examples/noise-color-demo.ipynb
* https://github.com/aewallin/allantools/blob/master/examples/three-cornered-hat-demo.ipynb
