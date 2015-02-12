D\ :sup:`3`\ PO
===============

**D3PO** project homepage: `<http://www.mpa-garching.mpg.de/ift/d3po/>`_

Summary
-------

Description
...........

The D3PO algorithm addresses the inference problem of **D**\enoising, **D**\econvolving, and **D**\ecomposing **P**\hoton **O**\bservations. Its primary goal is the simultaneous but individual reconstruction of the diffuse and point-like photon flux given a single photon count image, where the fluxes are superimposed.

In order to discriminate between these morphologically different signal components, a probabilistic algorithm is derived in the language of `information field theory <http://www.mpa-garching.mpg.de/ift/>`_ based on a hierarchical Bayesian parameter model. The signal inference exploits prior information on the spatial correlation structure of the diffuse component and the brightness distribution of the spatially uncorrelated point-like sources.
Since the derivation of the solution is not dependent on the underlying position space, the implementation of the D3PO algorithm uses the `NIFTY <http://www.mpa-garching.mpg.de/ift/nifty/>`_ package to ensure applicability to various spatial grids and at any resolution.

*Parts of this summary are taken from* [1]_ *without marking them explicitly as
quotations.*

Installation
------------

Requirements
............

*   `Python <http://www.python.org/>`_ (v2.7.x)

    *   `NumPy <http://www.numpy.org/>`_ and `SciPy <http://www.scipy.org/>`_
    *   `matplotlib <http://matplotlib.org/>`_
    *   `multiprocessing <http://docs.python.org/2/library/multiprocessing.html>`_
        (standard library)

*   `NIFTY <https://github.com/information-field-theory/nifty>`_ (v1.0.6) - Numerical Information
    Field Theory

Download
........

The latest release is tagged **v1.0.1** and is available as a source package
at `<https://github.com/information-field-theory/d3po/tags>`_. The current version can be
obtained by cloning the repository::

    git clone git://github.com/information-field-theory/d3po.git

Installation
............

*   D3PO can be installed using `PyPI <https://pypi.python.org/pypi>`_ and
    **pip** by running the following command::

        pip install ift_d3po

    Alternatively, a private or user specific installation can be done by::

        pip install --user ift_d3po

*   D3PO can be installed using **Distutils** by running the following command::

        cd d3po
        python setup.py install

    Alternatively, a private or user specific installation can be done by::

        python setup.py install --user
        python setup.py install --install-lib=/SOMEWHERE

First Steps
...........

To get started, you can browse through the
`how-to guide <http://www.mpa-garching.mpg.de/ift/d3po/HOWTO.html>`_
or simply run the demonstration::

        >>> run -m d3po.demo

Acknowledgement
---------------

Please, acknowledge the use of D3PO in your publication(s) by using a phrase
such as the following:

    *"Some of the results in this publication have been derived using the D3PO
    algorithm [Selig et al., 2014]."*

References
..........

.. [1] Selig et. al.,
    "Denoising, Deconvolving, and Decomposing Photon Observations", accepted by
    Astronomy & Astrophysics,
    `A&A, vol. 574, id. A74 <http://dx.doi.org/10.1051/0004-6361/201323006>`_,
    2014; `arXiv:1311.1888 <http://www.arxiv.org/abs/1311.1888>`_

Release Notes
-------------

The D3PO module is licensed under the
`GPLv3 <http://www.gnu.org/licenses/gpl.html>`_ and is distributed *without any
warranty*.

----

**D3PO** project homepage: `<http://www.mpa-garching.mpg.de/ift/d3po/>`_

