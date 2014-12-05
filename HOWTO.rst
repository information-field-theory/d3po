HOWTO -- An informal step-by-step guide
=======================================

.. ONLINE VERSION AT http://www.mpa-garching.mpg.de/ift/nifty/HOWTO.html

D3PO is a `NIFTY <http://www.mpa-garching.mpg.de/ift/nifty/>`_
inference algorithm capable of denoising, deconvolving, and
decomposing photon observations. This how-to guides you through all essential
steps to analyze your photon count images using D3PO.

.. tip:: Browsing the D3PO demo files in parallel might be advantageous.

.. currentmodule:: d3po

1. Data & Signal
----------------

First, you have to specify the spatial dimensionality of the signal you intend
to infer and its response to the data at hand.

Signal domain
.............

D3PO can reconstruct the diffuse and point-like photon flux by means of a signal
field describing the logarithmic flux. At this point, all you have to
specify is the spatial grid and desired resolution of the reconstructions.
Therefore, define a
`NIFTY space <http://www.mpa-garching.mpg.de/ift/nifty/start.html#spaces>`_
to serve as signal domain.

>>> from nifty import *
>>> x_space = rg_space([100, 100]) # two-dimensional regular 100 x 100 grid

Response
........

The mapping from signal to data space is described by the response operator and
incorporates all aspects of the observation. This might include a convolution
with the instrument's response functions, an exposure mask, and more.

The response needs to be a
`NIFTY operator <http://www.mpa-garching.mpg.de/ift/nifty/operator.html>`_.

>>> R = response_operator(x_space, den=True, target=None) # identity-like

.. warning:: It is crucial that :py:meth:`R.times` and
    :py:meth:`R.adjoint_times` are correctly adjoint. Verify that
    ``a.dot(R.times(b))`` equals ``R.adjoint_times(a).dot(b)`` for all ``a`` and
    ``b``!

Additionally, you can augment the response operator with following two
attributes:

  * The Boolean or
    `NIFTY switch <http://www.mpa-garching.mpg.de/ift/nifty/setup_classes.html#the-switch-class-on-off>`_
    :py:attr:`R.mask_out_pixels` controls the use of an additional (binary)
    mask that excludes pixels hosting bright point sources. This attribute is
    only used (switched on or off by D3PO itself) when raising an initial guess
    for the diffuse signal field.

  * The Boolean :py:attr:`R.identify_io_pixels` indicates whether there is an
    equivalence between data and signal space; i.e., you can uniquely
    associate pixels by their raveled indices. Although signal and data space
    need to have the same total dimensionality for this, they are not required
    to be equal. This attribute is used when raising an initial guess for the
    point-like signal field.

Data
....

The raw photon count data can be provided in any array-like form. If you create
a `NIFTY field <http://www.mpa-garching.mpg.de/ift/nifty/start.html#fields>`_,
the response's codomain provides the correct data domain.

>>> d = np.load("MY_PHOTON_COUNTS.npy")
>>> d = field(R.target, val=d)

.. note:: The default data space is a :py:class:`point_space` with the data type
     ``np.float64``.

-- Although the data are supposed to be integer photon counts, the expected
number of photons that is also defined in data space is a floating-point number.

2. Configuration
----------------

Second, you can configure parameters and settings for D3PO by editing a
configuration file. This file is executed in Python and can thus be quite
versatile including imports and functions, as well as comments preceded by
``#``.

When initialized, the :py:class:`configuration` class searches the configuration
file for certain parameters and processes them to its own attributes. These
parameters are discussed below.

.. note:: If you intend to implement additional parameters for your application,
    it is recommended to extend the :py:class:`configuration` class.

Model parameters
................

The input model parameters for the D3PO algorithm need to be specified.

  * ``alpha`` describes the spectral index of the *a priori* assumed
    inverse-Gamma distribution for the logarithmic harmonic power spectrum
    parameters of the diffuse component. The chosen value must be larger than
    1. Choosing a value of 1 would result in a asymptotically log-flat prior and
    is recommended.

  * ``q`` describes the low-value cut-off of this inverse-Gamma distribution.
    The chosen value must be larger than 0. Choosing a value of 0 would result
    in a asymptotically log-flat prior. It is recommended to choose a small,
    non-zero value for numerical stability.

  * ``sigma`` describes the Gaussian standard deviation of the smoothness prior
    assumed for the logarithmic harmonic power spectrum parameters of the
    diffuse component. This model parameter controls the freedom of the power
    spectrum to deviate from a strict power-law. The chosen value must be larger
    than 0. Choosing a value around 1 allows for a slightly bend, higher values
    give the power spectrum more freedom. In order to remove the smoothness
    prior assumption, set ``apply_smoothness = False`` (see below), which is
    equivalent to choosing an infinite standard deviation.

  * ``beta`` describes the spectral index of the *a priori* assumed
    inverse-Gamma distribution for point-like component. The chosen value must
    be larger than 1. Choosing a value of 1 would result in a asymptotically
    log-flat prior. For the analysis of 2D images a value of 1.5 can be
    recommended by geometrical and statistical arguments.

  * ``eta`` describes the low-value cut-off of this inverse-Gamma distribution.
    The chosen value must be larger than 0. Choosing a value of 0 would result
    in a asymptotically log-flat prior. It is recommended to choose a small,
    non-zero value that is appropriate for the chosen resolution; i.e., refining
    (or coarsening) the resolution by splitting (or merging) 2 pixels should be
    accounted for by decreasing (or increasing) this parameter by a factor of 4.

Primary settings
................

With the primary setting you can specify options for the D3PO task.

  * ``MAP_s`` indicates whether the maximum *a posteriori* solution for the
    diffuse signal should be computed or, alternatively, whether the Gibbs free
    energy approach should be used. Since the obtaining the MAP solution is
    computationally less expensive, it is recommended to try it at first. The
    Gibbs solution (eventually initialized with the MAP solution) might provide
    second order corrections.

  * ``MAP_t`` indicates whether the maximum *a posteriori* solution for the
    power spectrum of the diffuse signal should be computed or, alternatively,
    whether the covariance of the diffuse signal should be included. In other
    words, whether a classical or critical filter shall be applied. Confer to
    the ``perception`` parameter below.

  * ``MAP_u`` indicates whether the maximum *a posteriori* solution for the
    point-like signal should be computed or, alternatively, whether the Gibbs
    free energy approach should be used. This parameter is currently ignored.

  * ``NO_t`` indicates whether the power spectrum of the diffuse signal is to be
    kept as given or inferred from the data. In the former case, the initial
    power spectrum ``p0`` or ``t0`` (see below) remains untouched.

  * ``NO_u`` indicates whether the point-like signal is to be ignored or not.
    In the former case, **D3PO "light"** only infers the diffuse signal (and its
    power spectrum).

  * ``notes`` indicates whether intermediate printouts are displayed or not.

  * ``saves`` indicates whether intermediate results are saved or not. This is
    recommended for tests as it might help debugging problems.

  * ``saves`` indicates whether intermediate results are saved or not. This is
    recommended for tests as it might help debugging problems.

  * ``map_tol`` specifies the convergence tolerance for the solvers of the
    diffuse and point-like maps. The chosen value must be positive number. This
    parameter can be overwritten by ``runSD_s`` and ``runSD_u`` (see below).

  * ``tau_tol`` specifies the convergence tolerance for the solver of the power
    spectrum of the diffuse signal. The chosen value must be positive number.

  * ``aftermath`` indicates whether the uncertainty is to be computed after
    convergence of the maps or not. Instead of a Boolean, it can be an integer
    specifying the maximum number of iterations.

  * ``ncpu`` and ``nper`` are general parameters used by multiprocessing that
    specify the number of CPUs to used and the number of tasks performed by each
    individual worker threat. D3PO uses multiprocessing when probing operators.
    You can this off by globally setting ``nifty.about.multiprocessing.off()``.

  * ``seed`` fixes the used random seed. Choosing ``seed = None`` will use a
    unfixed, "random" random seed.

Secondary settings
..................

The secondary settings allow you to control advanced options for the D3PO task.

  * You can give starting values for all inference quantities:

      - ``s0`` the diffuse signal
      - ``p0`` the power spectrum of the diffuse signal
      - ``t0`` its logarithm (overwrites ``p0``)
      - ``u0`` the point-like signal
      - ``D0`` the variance of the diffuse signal
      - ``F0`` the variance of the point-like signal

    The input starting values are supposed to be valid inputs for NIFTY.

  * ``apply_smoothness`` indicates whether to incorporate the smoothness prior
    or not.

  * ``force_smoothness`` indicates whether to enforce the given ``sigma`` or
    not. It is not recommended to enforce too small values of sigma.

  * ``perception`` characterizing the applied filter. A tuple (0,0) describes
    a classical, (1,0) a critical filter. For details see
    `critical Wiener filtering <http://www.mpa-garching.mpg.de/ift/nifty/demo_excaliwir.html#critical-wiener-filtering-a-priori-unknown-signal-correlation>`_.

  * You can manipulate the indexing of the harmonic space through

      - ``log`` a flag indicating whether binning is performed on logarithmic
        scale or not
      - ``nbin`` the number of used bins
      - ``binbounds`` user specific inner boundaries of the bins

Numerical settings
..................

The following set of parameters manipulate the numerical effectiveness and
efficiency of D3PO.

  * It has proven useful to ensure the convergence of the diffuse signal
    on large scales at first. Smaller scales are sequentially included.

      - ``nb`` controls the inclusion of smaller harmonic bands. If zero, no
        sequencing is applied; if given as a positive integer, it specifies the
        number logarithmically binned groups of bands; if given as a list, it
        specifies the separating bands. The result is stored in the
        :py:attr:`configuration.kbands` attribute.

      - ``sb`` specifies an additional Gaussian-like supression of small bands.
        The smallest scale is suppressed by a factor ``10**-sb``. The resulting
        array is stored in the :py:attr:`configuration.gaussmooth` attribute.

  * The signal fields are optimized using
    `steepest descent <http://www.mpa-garching.mpg.de/ift/nifty/tools.html#nifty.nifty_tools.steepest_descent>`_
    (SD).

      - ``iniSD_s`` is a dictionary with parameters for initializing the SD
        optimization of the diffuse signal field.
      - ``runSD_s`` is a tuple of dictionaries for running this SD, where the
        first applies in the initial cycles 1 and 2, and the second in the
        following cycles until global convergence.
      - ``iniSD_u`` and ``runSD_u`` correspond to the above but for the
        point-like signal.

  * The (co)variances of the signal fields are optimized by probing using
    `conjugate gradient <http://www.mpa-garching.mpg.de/ift/nifty/tools.html#conjugated-gradients-the-conjugate-gradient-class>`_
    (CG).

      - The dictionaries ``iniCGprobing`` and ``runCGprobing`` are used in the
        same way as the SD dictionaries.
      - ``precondition_D`` indicates whether the prior covariance is to be used
        as preconditioner or not.
      - There are additional dictionaries for the diffuse and point-like
        covariances (``D`` and ``F``) in the position or harmonic basis (``xx``
        or ``kk``). Their items overwrite the above ones.

3. Solve your problem
---------------------

Third, you have to initialize your problem and solve it.

Therefore, import D3PO and create a problem.

>>> import d3po
>>> problem = d3po.problem(R, configfile="./MY_CONFIG_FILE", workingdirectory="./test_1337")

This will create the following directory tree.

.. code::

    ./MY_CONFIG_FILE           # original configuration file
    ./test_1337/               # working directory (for final results)
    ./test_1337/MY_CONFIG_FILE # a copy (in case the original is modified for test_1338)
    ./test_1337/d3po_tmp/      # subdirectory (for intermediate results)

.. tip:: At this point, you can also check and modify the configuration
    by accessing the :py:attr:`problem.config` attribute.

>>> problem.config.map_mode, problem.config.tau_mode # reconstruction modes
(1, -1)
>>> problem.config.p0 # initial power spectrum
array([ 1.00000000e+00,  1.00000000e+00, ... ])

Since the problem is now set up, you can solve it for your data set.

>>> problem.solve(d)

The basic results are saved to disc after completion. Nonetheless, you can
retrieve fields and arrays from the :py:class:`problem` class for further usage.

>>> s, t, u = problem.get(s=True, u=True, t=True)

Do you want to know more?
-------------------------

As a continuation of this HOWTO guide, you can play around with the
`D3PO Demo <http://www.mpa-garching.mpg.de/ift/d3po/demo.html>`_.

