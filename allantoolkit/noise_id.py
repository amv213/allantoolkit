import logging
import numpy as np
import scipy.special
from . import tables
from . import stats
from . import utils
from . import bias

# Spawn module-level logger
logger = logging.getLogger(__name__)

# Confidence Intervals
ONE_SIGMA_CI = scipy.special.erf(1/np.sqrt(2))
#    = 0.68268949213708585

# shorten type hint to save some space
Array = np.ndarray


# Noise Identification using ACF


def acf(z: Array, k: int) -> float:
    """Lag-k autocorrelation function.

    The autocorrelation function (ACF) is a fundamental way to describe a time
    series by multiplying it by a delayed version of itself, thereby showing
    the degree by which its value at one time is similar to its value at a
    certain later time.

    # TODO: check this gives exactly Stable32 results and maybe switch to
    Fourier method?

    References:
         [RileyStable32]_ (5.5.3, pg.53-54)

    Args:
        z:  timeseries for which to calculate lag-k autocorrelation
        k:  lag interval

    Returns:
        timeseries autocorrelation factor at lag-k
    """

    # Mean value of the timeseries
    zbar = np.nanmean(z)
    z0 = z - zbar

    # Calculate autocorrelation factor
    r = np.nansum(z0[:-k]*z0[k:])/np.nansum(z0**2)

    # Scale by (N-1)/N, this was suggested in one of the Stable32 manuals
    # but cannot find the reference anymore. Turns out Stable32 does indeed
    # do this.
    r *= (z.size - 1)/z.size

    return r


def acf_noise_id_core(z: Array, dmax: int, dmin: int = 0):
    """Core algorithm for the lag1 autocorrelation power law noise
    identification algorithm.

    References:
        [RileyStable32]_ (5.5.6 pg.56)
        http://www.stable32.com/Auto.pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.503.9864&rep=rep1&type=pdf
        Power law noise identification using the lag 1 autocorrelation
        Riley,W.J. et al.
        18th European Frequency and Time Forum (EFTF 2004)
        https://ieeexplore.ieee.org/document/5075021

    Args:
        z:                  timeseries for which to calculate lag-1
                            autocorrelation. Input data should be at the
                            particular averaging time tau of interest.
        dmax:               maximum order of differencing
        dmin (optional):    minimum order of differencing


    Returns:
        an estimate p of the `alpha` of the dominant power law noise type.
    """

    d = 0
    done = False
    while not done:

        # Calculate lag-1 autocorrelation function
        r1 = acf(z=z, k=1)

        delta = r1/(1 + r1)

        if d >= dmin and (delta < 0.25 or d >= dmax):

            p = -round(2*delta) - 2*d
            done = True

        else:

            z = np.diff(z)
            d += 1

    return p


def acf_noise_id(data: Array, data_type: str, m: int, dev_type: str) -> int:
    """Effective method for identifying power law noise factor `alpha` at
    given averaging time.

    Noise identification based on Lag-1 autocorrelation function. Excellent
    discrimination for all common power law noises for both phase and
    frequency data, including difficult cases with mixed noises.

    References:
        [RileyStable32]_ (5.5.3-6 pg.53-7)

    Args:
        data:       array of input timeseries data. Before analysis, the data
                    should be preprocessed to remove outliers, discontinuities,
                    and  deterministic components
        data_type:  input data type. Either `phase` or `freq`.
        m:          averaging factor at which to estimate dominant noise type
        dev_type:   type of deviation used for analysis, e.g. `adev`.

    Returns:
        estimate of the `alpha` exponent, the dominant power law noise type.
    """

    # The input data should be for the particular averaging time, τ, of
    # interest, and it is therefore be necessary to decimate the phase
    # data or average the frequency data by the appropriate averaging factor
    # before applying the noise identification algorithm
    z = utils.decimate(data=data, m=m, data_type=data_type)

    # The dmax parameter should be set to 2 or 3 for an Allan or Hadamard (2
    # or 3-sample) variance analysis, respectively.
    dmax = tables.D_ORDER.get(dev_type, None)

    if dmax is None:
        raise KeyError(f"Could not find ACF differencing factor d for"
                       f" {dev_type}. Maybe ACF does not support this dev "
                       f"type?")

    # Run lag1 autocorrelation noise id algorithm
    p = acf_noise_id_core(z=z, dmax=dmax)

    # The alpha result is equal to p+2 or p for phase or frequency data,
    # respectively, and may be rounded to an integer (although the fractional
    # part is useful for estimated mixed noises).
    alpha = p+2 if data_type == 'phase' else p

    if alpha > 2:
        raise ValueError(f"Estimated noise type out of bounds (alpha="
                         f"{alpha}). Are you sure that your data "
                         f"matches the given data_type?")

    return alpha


# Noise Identification using B1

def b1_expected(N: int, mu: int, r: float = 1) -> float:
    """ Expected B1 ratio for a timeseries of N fractional frequency
    datapoints and phase noise exponent mu

    References:
        Howe, Beard, Greenhall, Riley,
        A TOTAL ESTIMATOR OF THE HADAMARD FUNCTION USED FOR GPS OPERATIONS
        32nd PTTI, 2000
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a484835.pdf
        (Table 3, pg. 261-2)

    Args:
        N:  number of fractional frequency datapoints. This is M-1,
            for M phase data points.
        mu: phase noise power-law exponent for which to calculate expected
            B1 ratio
        r:  dead time ratio. Set to 1.

    Returns:
        B1 ratio for given phase noise exponent.
    """

    if mu == 2:  # FWFM
        return N * (N + 1) / 6.0

    elif mu == 1:  # RWFM
        return N / 2.

    elif mu == 0:  # FLFM
        return N*np.log(N) / (2*(N-1)*np.log(2))

    elif mu == -1:  # WHFM
        return 1

    elif mu == -2:  # WHPM or FLPM
        return (N**2 - 1) / (1.5*N*(N-1.))

    else:
        logger.warning("Calculating B1 ratio for phase noise exponent outside "
                       "bounds")
        return N * (1 - N**mu) / (2 * (N-1) * (1 - 2**mu))


def b1_noise_id(measured: float, N: int, r: float = 1) -> int:
    """ Identify phase noise power-law exponent type `mu` from measured B1
    ratio.

    References:
        Howe, Beard, Greenhall, Riley,
        A TOTAL ESTIMATOR OF THE HADAMARD FUNCTION USED FOR GPS OPERATIONS
        32nd PTTI, 2000
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a484835.pdf
        (Table 4, pg. 263)

    Args:
        measured:   measured B1 ratio
        N:          number of fractional frequency datapoints on which
                    measured B1 ratio was calculated. This is M-1,
                    for M phase data points.
        r:          dead time ratio. Set to 1.

    Returns:
        identified `mu` phase noise power-law exponent
    """

    mus = [2, 1, 0, -1, -2]

    # Expected B1 ratio for each phase noise type
    d = {mu: b1_expected(N=N, mu=mu, r=r) for mu in mus}

    # Boundaries between noise types
    b = {}
    for mu, b1 in d.items():

        if mu > -2:

            bndry = 0.5*(b1 + d[mu-1]) if mu == 2 else np.sqrt(b1 * d[mu-1])
            b[mu] = bndry


    #print(f"\tBoundaries {b}")

    # Assign measured b1 to most plausible noise type:
    # the actual measured ratio is tested against mu values downwards from
    # the largest applicable mu
    for mu, bndry in b.items():

        if measured > bndry:

            return mu

    # If the above didn't return anything, give lowest possible noise type
    return -2


# Noise Identification using R(n)

def rn_expected(m: int, alpha: int) -> float:
    """R(n) ratio expected from theory for given noise type and number of
    phase samples

    References:

        D. B. Sullivan, D. W. Allan, D. A. Howe, and F. L. Walls (editors)
        1990, “Characterization of Clocks and  Oscillators, ” National
        Institute of Standards and Technology  Technical Note 1337, Sec. A-6.
        Table 2.

    Args:
        m:      averaging factor
        alpha:  fractional frequency noise power-law exponent for which to
                calculate expected R(n) ratio

    Returns:
        R(n) ratio for given fractional frequency noise exponent
    """

    if alpha == 1:

        # assume measurement bandwidth `f_h` is 1/(2*tau_0)
        # then `w_h` == 2*pi*f_h = pi / tau_0
        # which means w_h*tau == w_h*m*tu_0 = pi * m

        return 3.37 / (1.04 + 3 * np.log(np.pi*m))

    elif alpha == 2:

        return 1. / m

    else:

        raise ValueError('Use B1 ratio instead for other noise types')


def rn_noise_id(measured: float, m: int) -> int:
    """ Identify fractional frequency noise power-law exponent type `alpha`
    from measured R(n) ratio.

    References:
        Howe, Beard, Greenhall, Riley,
        A TOTAL ESTIMATOR OF THE HADAMARD FUNCTION USED FOR GPS OPERATIONS
        32nd PTTI, 2000
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a484835.pdf
        (Table 4, pg. 263)

    Args:
        measured:   measured R(n) ratio
        m:          averaging factor at which ratio was calculated

    Returns:
        identified `alpha` fractional frequency noise power-law exponent
    """

    alphas = [1, 2]

    # Expected R(n) ratio for each phase noise type
    d = {alpha: rn_expected(m=m, alpha=alpha) for alpha in alphas}

    # Boundary between FLPM (alpha=1) and WHPM (alpha=2)
    bndry = np.sqrt(d[1] * d[2])

    # Compare measured result against boundary
    alpha = 1 if measured > bndry else 2

    return alpha


# Noise ID algorithms

def noise_id(data: Array, data_type: str, m: int, rate: float,
             dev_type: str, n: int) -> int:
    """Noise identification pipeline, following what prescribed by Stable32.

    References:
        http://www.wriley.com/CI2.pdf
        [RileyStable32Manual]_ (Table pg. 97) (missing htotvar)
        [RileyStable32Manual]_ (Confidence Intervals pg. 89-93)
        [RileyStable32]_ (5.5.2-6 pg.53-7)
        Howe, Beard, Greenhall, Riley,
        A TOTAL ESTIMATOR OF THE HADAMARD FUNCTION USED FOR GPS OPERATIONS
        32nd PTTI, 2000
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a484835.pdf

        # FIXME: find a reference for how to handle htotdev noiseid

    Args:
        data:       array of input timeseries data. Before analysis, the data
                    should be preprocessed to remove outliers, discontinuities,
                    and  deterministic components
        data_type:  input data type. Either `phase` or `freq`.
        m:          averaging factor at which to estimate dominant noise type
        rate:   sampling rate of the input data, in Hz.
        dev_type:   type of deviation used for analysis, e.g. `adev`.
        n:          number of analysis points on which deviation was calculated

    Returns:
        estimate of the `alpha` exponent, the dominant power law noise type.
    """

    use_acf = ['adev', 'oadev', 'mdev', 'tdev', 'hdev', 'ohdev', 'totdev',
               'mtotdev', 'ttotdev', 'htotdev', 'theoh']
    use_b1 = ['adev', 'oadev', 'mdev', 'tdev', 'hdev', 'ohdev', 'totdev',
               'mtotdev', 'ttotdev', 'htotdev', 'theoh']  # hdev?
    use_b1_star = ['hdev']  # TODO: any others?
    use_rn = ['mdev', 'tdev', 'mtotdev', 'ttotdev', 'htotdev',
              'theoh']

    # Stable32 uses two methods for power law noise identification, based
    # respectively on the lag 1 autocorrelation and the B1 bias factor. The
    # former method is preferred, and is used when there are at least 30
    # analysis  data  points.

    # Size of dataset at given averaging factor:
    nn = utils.decimate(data=data, m=m, data_type=data_type).size

    if nn >= 30 and dev_type in use_acf:

        # print(f"AF: {m} | TAU: {m/rate} - Using ACF noise id")

        return acf_noise_id(data=data, data_type=data_type, m=m,
                            dev_type=dev_type)

    # Estimate alpha when there are less than 30 analysis datapoints
    elif dev_type in use_b1:

        # print(f"AF: {m} | TAU: {m/rate} - Using B1 noise id")

        # B1 ratios expect phase data
        x = utils.input_to_phase(data=data, rate=rate, data_type=data_type)
        y = utils.phase2frequency(x=x, rate=rate)

        # compare b1 bias factor = standard variance / allan variance vs
        # expected value of this same ratio for pure noise types

        # Actual
        svar, _ = stats.calc_svar(x=x, m=m, rate=rate)
        avar, _ = stats.calc_avar(x=x, m=m, rate=rate)
        b1 = svar / avar

        # B1 noise_id
        mu = b1_noise_id(measured=b1, N=n)  # this should be number of
        # frequency samples

        # If modified family of variances MVAR, TVAR or TOTMVAR
        # distinguish between WPM vs FPM by:
        # Supplement with R(n) ratio = mod allan / allan variance
        if mu == -2 and dev_type in use_rn:  # find if alpha = 1 or 2

            # print("Using Rn ratio")

            # Actual
            mvar, _ = stats.calc_mvar(x=x, m=m, rate=rate)
            rn = mvar / avar

            # Rn noise_id
            alpha = rn_noise_id(measured=rn, m=m)
            return alpha

        # For the Hadamard variance, for which RRFM noise can apply (mu=3,
        # alpha=-4) the B1 ratio can be applied to frequency (rather than
        # phase) data, and adding 2 to the resulting mu
        # FIXME: implement this correctly and then remove the False
        #  statement to make it run
        elif False and mu == 2 and dev_type in use_b1_star:  # find if alpha =
            # -3 or -4

            # print("Using *B1 ratio")

            # *B1 ratio applies to frequency data
            y = data if data_type == 'freq' else utils.phase2frequency(x=data,
                                                                       rate=rate)
            svar, _ = stats.calc_svar_freq(y=y, m=m, rate=rate)
            avar, _ = stats.calc_avar_freq(y=y, m=m, rate=rate)
            b1star = svar / avar

            # B1 noise_id
            mu = b1_noise_id(measured=b1star, N=n)
            mu += 2

            assert mu <= 3, f"Invalid phase noise type mu: {mu}"

        # Get alpha value corresponding to identified mu
        alpha = [a for a, m in tables.ALPHA_TO_MU.items() if m == mu][0]

        return alpha


def noise_id_theoBR(kf: float, afs: Array) -> Array:
    """ Use Theo1 bias factor to determine noise type. As done by Stable32
    `Sigma` tool.

    Thresholds for noise sorting are geometric means of nominal Theo1
    variance bias values.

    PRELIMINARY - REQUIRES FURTHER TESTING.

    This function should probably only be used to emulate the behaviour of
    Stable32 `Sigma` tool. Although in that case the kf provided here should
    be the one averaged over /30 and not /6

    References:

        http://www.wriley.com/Fast%20Bias-Removed%20Theo1%20Calculation%20with%20R.pdf

        [RileyStable32Manual]_ (TheoBR and TheoH, pg.80)

    Args:
        kf: theoBR correction factor

    Returns:
        estimate of the `alpha` exponents, the dominant power law noise type
        for the whole run.
    """

    # Initialise array
    alphas = np.full(afs.size, np.NaN)

    # Get nominal Theo1 var bias factors for various power law noise types
    # at each averaging factor
    for i, m in enumerate(afs):

        noise_types = [2, 1, 0, -1, -2]

        # Nominal Theo1 bias values for each phase noise type
        d = {alpha: bias.calc_bias_theo1(None, m=m, alpha=alpha) for alpha
             in noise_types}

        # Boundaries between noise types
        b = {}
        for alpha, b1 in d.items():

            if alpha > -2:

                bndry = np.sqrt(b1 * d[alpha-1])
                b[alpha] = bndry

        # Assign measured b1 to most plausible noise type:
        alphas[i] = -2  # prefill with lowest possible noise type
        for alpha, bndry in b.items():

            if kf < bndry:

                alphas[i] = alpha
                break

    return alphas


def noise_id_theoBR_fixed(kf) -> float:
    """ Use Theo1 bias factor to determine noise type, as done by Stable32
    `Run` and `Plot` tools.

    There is one bias factor value and one noise type determination for the
    whole run. Thresholds for noise sorting are geometric means of nominal
    Theo1 variance bias values.

    PRELIMINARY - REQUIRES FURTHER TESTING.

    This function should be used to emulate the behaviour of Stable32 `Run`
    tool. The kf provided here should be the standard one i.e. calculated
    over the average / 6 (See Notes)

    References:

        http://www.wriley.com/Fast%20Bias-Removed%20Theo1%20Calculation%20with%20R.pdf

        [RileyStable32Manual]_ (TheoBR and TheoH, pg.80)

    Args:
        kf: theoBR correction factor

    Returns:
        estimate of the `alpha` exponent, the dominant power law noise type
        for the whole run.

    Notes:
        Slightly different versions of Theo1-like statistics are present in
        the literature, all stemming from the same core Thêo1 definition. They
        vary in averaging factor ranges, bias correction methods and other
        details. Notably, the ThêoBR bias corrections and ThêoH presentations
        may differ between implementations.

        The correction factor kf to use here can be obtained from
        allantoolkit.bias.calc_bias_theobr(). Check its documentation for
        details.
    """

    # Nominal Theo1 bias values for each phase noise type
    d = tables.BIAS_THEO1_FIXED

    # Boundaries between noise types
    b = {}
    for alpha, b1 in d.items():

        if alpha > -2:
            bndry = np.sqrt(b1 * d[alpha - 1])
            b[alpha] = bndry

    # Assign measured b1 to most plausible noise type:
    for alpha, bndry in b.items():

        if kf < bndry:

            return alpha

    # If the above didn't return anything, give lowest possible noise type
    return -2



