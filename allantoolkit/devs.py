import logging
import numpy as np
from . import ci
from . import noise_id
from . import bias
from . import stats
from . import utils
from typing import List, Tuple, NamedTuple, Union

# Spawn module-level logger
logger = logging.getLogger(__name__)

# shorten type hint to save some space
Array = np.ndarray

# group allowed taus types to save some space
Taus = Union[str, float, List, Array]


# define named tuple to hold dev results
class DevResult(NamedTuple):
    """Represents a statistical stability analysis result.

    Args:
        afs:        array of averaging factors for which deviations were
                    computed
        taus:       array of corresponding averaging times, in seconds
        ns:         array with number of analysis points used to compute each
                    deviation.
        alphas:     array of estimated dominant noise type at each deviation
        devs_lo:    array of estimated statistical lower bounds for each
                    deviation
        devs:       array with deviations computed at each averaging time
        devs_hi:    array of estimated statistical higher bounds for each
                    deviation
    """

    afs: Array
    taus: Array
    ns: Array
    alphas: Array
    devs_lo: Array
    devs: Array
    devs_hi: Array

    def __str__(self) -> str:
        """A human-friendly pretty-print representation of the object.

        Returns:
            pretty-print of the DevResult object.
        """

        # A bit of formatting to get all rows aligning nicely...
        afs = '[ ' + '   '.join(f"{x:<10}" for x in self.afs) + ' ]'
        taus = '[ ' + '   '.join(f"{x:<10}" for x in self.taus) + ' ]'
        ns = '[ ' + '   '.join(f"{x:<10}" for x in self.ns) + ' ]'
        alphas = '[ ' + '   '.join(f"{x:<10}" for x in self.alphas) + ' ]'
        devs_lo = '[ ' + \
                  '   '.join(f"{np.format_float_scientific(x, 4, trim='k')}"
                             for x in self.devs_lo) + ' ]'
        devs = '[ ' + \
               '   '.join(f"{np.format_float_scientific(x, 4, trim='k')}"
                          for x in self.devs) + ' ]'
        devs_hi = '[ ' + \
                  '   '.join(f"{np.format_float_scientific(x, 4, trim='k')}"
                             for x in self.devs_hi) + ' ]'

        return ("\nSTABILITY ANALYSIS RESULTS:\n"
                "AFS:     \t{afs}\n"
                "TAUS (s):\t{taus}\n"
                "#:       \t{ns}\n"
                "ALPHAS:  \t{alphas}\n"
                "DEVS_LO: \t{devs_lo}\n"
                "DEVS:    \t{devs}\n"
                "DEVS_HI: \t{devs_hi}\n").format(
            afs=afs, taus=taus, ns=ns, alphas=alphas,
            devs_lo=devs_lo, devs=devs, devs_hi=devs_hi)


def dev(dev_type: str, data: Array, rate: float, data_type: str,
        taus: Taus, max_af: int) -> DevResult:
    """Core pipeline processing the input data and returning the appropriate
    frequency stability analysis results for the given deviation type.

    Args:
        dev_type:   type of deviation to be computed, e.g. `adev`.
        data:       array of phase (in units of seconds) or fractional
                    frequency data for which to calculate deviation.
        rate:       sampling rate of the input data, in Hz.
        data_type:  input data type. Either `phase` or `freq`.
        taus:       array of averaging times for which to compute deviation.
                    Can also be one of the keywords: `all`, `many`, `octave`,
                    `decade`.
        max_af:     maximum averaging factor for which to compute deviation.
                    Defaults to length of dataset.

    Returns:
        (afs, taus, ns, alphas, devs_lo, devs, devs_hi) DevResult named tuple
    """

    # Easier to work with phase data, in units of seconds
    x = utils.input_to_phase(data=data, rate=rate, data_type=data_type,
                             normalize=True)

    # ---------------------
    # SET AVERAGING TIMES
    # ---------------------
    # generates [taus, afs]
    # ---------------------

    # Build/Select averaging factors at which to calculate deviations
    taus, afs = utils.tau_generator(data=x, rate=rate, dev_type=dev_type,
                                    taus=taus, maximum_m=max_af)

    # ---------------------
    # CALC VARIANCES
    # ---------------------
    # generates [vars, ns]
    # ---------------------

    if dev_type != 'theo1':

        # Initialise arrays
        ns, vars = np.zeros(afs.size, dtype=int), np.zeros(afs.size)

        # Set dispatcher to appropriate variance calculator for this dev_type:
        # should be function of this signature: func(x, m, rate) -> var, n
        func = getattr(stats, 'calc_' + dev_type.replace('dev', 'var'))

        # Calculate variance at each averaging time
        for i, m in enumerate(afs):

            # Calculate variance, and number of analysis points it is based on
            var, n = func(x=x, m=m, rate=rate)

            ns[i], vars[i] = n, var

    else:  # Fast batch calculation for theo1

        vars, ns = stats.calc_theo1_fast(x=x, rate=rate, explode=True)
        vars, ns = vars[afs], ns[afs]  # index out only selected AFS

    # Get rid of averaging times where dev calculated on too few samples (<= 1)
    afs, taus, ns, vars = utils.remove_small_ns(afs, taus, ns, vars)

    if dev_type in ['mtie', 'tierms']:  # Stop here

        devs = np.sqrt(vars)

        nan_array = np.full(afs.size, np.NaN)
        return DevResult(afs=afs, taus=taus,  ns=ns, alphas=nan_array,
                         devs_lo=nan_array,  devs=devs, devs_hi=nan_array)

    # -------------------------
    # DE-BIAS VARS AND NOISE ID
    # -------------------------
    # generates [alphas]
    # -------------------------

    if dev_type != 'theo1':

        # Initialise arrays
        alphas, devs = np.zeros(afs.size, dtype=int), np.zeros(afs.size)

        for i, (m, n, var) in enumerate(zip(afs, ns, vars)):

            # Estimate Noise ID
            if i < afs.size - 1:  # Only estimate if not last averaging time
                alpha = noise_id.noise_id(data=x, m=m, rate=rate,
                                          data_type='phase',
                                          dev_type=dev_type, n=n)

            else:
                # Use previous estimate at longest averaging time
                alpha = alphas[i-1]

            # Apply Bias Corrections

            # Dispatch to appropriate bias calculator for this dev_type:
            # should be function of this signature: func(x, m, alpha) -> b

            bfunc = getattr(bias, 'calc_bias_' + dev_type.replace('dev', 'var'))
            b = bfunc(data=data, m=m, alpha=alpha)

            var *= b  # correct variance

            alphas[i], vars[i] = alpha, var

    else:  # batch de-biasing using theoBR

        # Note that if we ever switch from letting users provide an `alpha` to
        # the API, Stable32 calculates the bias using the hard-coded
        # biasing factors in bias.calc_bias_theo1() instead

        kf = bias.calc_bias_theobr(x=x, rate=rate)
        vars = kf*vars

        # ID noise type for the whole run
        alpha = noise_id.noise_id_theoBR_fixed(kf=kf)
        alphas = np.full(afs.size, alpha)

    # ---------------------------------------------
    # CALCULATE CONFIDENCE INTERVALS AND OUTPUT DEV
    # ---------------------------------------------
    # generates [devs, devs_lo, devs_hi]
    # ---------------------------------------------

    # Initialise arrays
    devs = np.zeros(afs.size)
    devs_lo, devs_hi = np.zeros(afs.size), np.zeros(afs.size)

    for i, (m, n, alpha, var) in enumerate(zip(afs, ns, alphas, vars)):

        # Calculate ci bounds
        var_lo, var_hi = ci.get_error_bars(x=x, m=m, var=var, n=n, alpha=alpha,
                                           dev_type=dev_type)

        # Convert variances to deviations
        dev, dev_lo, dev_hi = np.sqrt(var), np.sqrt(var_lo), np.sqrt(var_hi)

        devs[i], devs_lo[i], devs_hi[i] = dev, dev_lo, dev_hi

    return DevResult(afs=afs, taus=taus,  ns=ns, alphas=alphas,
                     devs_lo=devs_lo, devs=devs, devs_hi=devs_hi)


def adev(data: Array, rate: float = 1., data_type: str = "phase",
         taus: Taus = None, max_af: int = None) -> DevResult:
    """Calculates the Allan deviation (ADEV) of phase or fractional
    frequency data.

    .. hint:
    
        classic - use only if required - relatively poor confidence.

    The Allan deviation - :math:`\\sigma_y(\\tau)` - is the square root of
    the Allan variance. The Allan variance is the same as  the  ordinary
    variance  for  white  FM  noise, but  has  the  advantage,  for  more
    divergent  noise  types  such  as flicker  noise,  of  converging  to  a
    value  that  is  independent  on  the  number  of  samples.

    In terms of `phase` data, the Allan variance may be calculated as:

    .. math::

        \\sigma^2_y(\\tau) = { 1 \\over 2 (N-2) \\tau^2 }
        \\sum_{i=1}^{N-2} \\left[ x_{i+2} - 2x_{i+1} + x_{i} \\right]^2

    where :math:`x_i` is the :math:`i^{th}` of :math:`N` phase
    values
    spaced by an averaging time :math:`\\tau`.

    For a time-series of fractional frequency values, the Allan variance is
    defined as:

    .. math::

        \\sigma^{2}_y(\\tau) =  { 1 \\over 2 (M - 1) } \\sum_{i=1}^{M-1}
        \\left[ \\bar{y}_{i+1} - \\bar{y}_i \\right]^2

    where :math:`\\bar{y}_i` is the :math:`i^{th}` of :math:`M`
    fractional frequency values averaged over the averaging time
    :math:`\\tau`.

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [SP1065]_ (pg.14-15)
    """

    return dev(dev_type='adev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def oadev(data: Array, rate: float = 1., data_type: str = "phase",
          taus: Taus = None, max_af: int = None) -> DevResult:
    """ overlapping Allan deviation.
        General purpose - most widely used - first choice

    .. math::

        \\sigma^2_{OADEV}(m\\tau_0) = { 1 \\over 2 (m \\tau_0 )^2 (N-2m) }
        \\sum_{n=1}^{N-2m} ( {x}_{n+2m} - 2x_{n+1m} + x_{n} )^2

    where :math:`\\sigma^2_x(m\\tau_0)` is the overlapping Allan
    deviation at an averaging time of :math:`\\tau=m\\tau_0`, and
    :math:`x_n` is the time-series of phase observations, spaced by the
    measurement interval :math:`\\tau_0`, with length :math:`N`.

    NIST [SP1065]_ eqn (11), page 16.

    Args:
        See documentation for allantoolkit.allantools.dev
    """

    return dev(dev_type='oadev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def mdev(data: Array, rate: float = 1., data_type: str = "phase",
         taus: Taus = None, max_af: int = None) -> DevResult:
    """  Modified Allan deviation.
         Used to distinguish between White and Flicker Phase Modulation.

    .. math::

        \\sigma^2_{MDEV}(m\\tau_0) = { 1 \\over 2 (m \\tau_0 )^2 (N-3m+1) }
        \\sum_{j=1}^{N-3m+1} \\lbrace
        \\sum_{i=j}^{j+m-1} {x}_{i+2m} - 2x_{i+m} + x_{i} \\rbrace^2

    see http://www.leapsecond.com/tools/adev_lib.c

    NIST [SP1065]_ eqn (14), page 17.

    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.

    Returns
    -------
    (taus2, md, mde, ns): tuple
          Tuple of values
    taus2: np.array
        Tau values for which td computed
    md: np.array
        Computed mdev for each tau value
    mde: np.array
        mdev errors
    ns: np.array
        Values of N used in each mdev calculation

    """

    return dev(dev_type='mdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def tdev(data: Array, rate: float = 1., data_type: str = "phase",
         taus: Taus = None, max_af: int = None) -> DevResult:
    """ Time deviation.
        Based on modified Allan variance.

    .. math::

        \\sigma^2_{TDEV}( \\tau ) = { \\tau^2 \\over 3 }
        \\sigma^2_{MDEV}( \\tau )

    Note that TDEV has a unit of seconds.

    NIST [SP1065]_ eqn (15), page 18.

    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.

    Returns
    -------
    (taus, tdev, tdev_error, ns): tuple
          Tuple of values
    taus: np.array
        Tau values for which td computed
    tdev: np.array
        Computed time deviations (in seconds) for each tau value
    tdev_errors: np.array
        Time deviation errors
    ns: np.array
        Values of N used in mdev_phase()

    Notes
    -----
    http://en.wikipedia.org/wiki/Time_deviation
    """
    return dev(dev_type='tdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def hdev(data: Array, rate: float = 1., data_type: str = "phase",
         taus: Taus = None, max_af: int = None) -> DevResult:
    """ Hadamard deviation.
        Rejects frequency drift, and handles divergent noise.

    .. math::

        \\sigma^2_{HDEV}( \\tau ) = { 1 \\over 6 \\tau^2 (N-3) }
        \\sum_{i=1}^{N-3} ( {x}_{i+3} - 3x_{i+2} + 3x_{i+1} - x_{i} )^2

    where :math:`x_i` is the time-series of phase observations, spaced
    by the measurement interval :math:`\\tau`, and with length :math:`N`.

    NIST [SP1065]_ eqn (17) and (18), page 20

    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.
    """
    return dev(dev_type='hdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def ohdev(data: Array, rate: float = 1., data_type: str = "phase",
          taus: Taus = None, max_af: int = None) -> DevResult:
    """ Overlapping Hadamard deviation.
        Better confidence than normal Hadamard.

    .. math::

        \\sigma^2_{OHDEV}(m\\tau_0) = { 1 \\over 6 (m \\tau_0 )^2 (N-3m) }
        \\sum_{i=1}^{N-3m} ( {x}_{i+3m} - 3x_{i+2m} + 3x_{i+m} - x_{i} )^2

    where :math:`x_i` is the time-series of phase observations, spaced
    by the measurement interval :math:`\\tau_0`, and with length :math:`N`.

    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.

    Returns
    -------
    (taus2, hd, hde, ns): tuple
          Tuple of values
    taus2: np.array
        Tau values for which td computed
    hd: np.array
        Computed hdev for each tau value
    hde: np.array
        hdev errors
    ns: np.array
        Values of N used in each hdev calculation

    """
    return dev(dev_type='ohdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def totdev(data: Array, rate: float = 1., data_type: str = "phase",
           taus: Taus = None, max_af: int = None) -> DevResult:
    """ Total deviation.
        Better confidence at long averages for Allan deviation.

    .. math::

        \\sigma^2_{TOTDEV}( m\\tau_0 ) = { 1 \\over 2 (m\\tau_0)^2 (N-2) }
            \\sum_{i=2}^{N-1} ( {x}^*_{i-m} - 2x^*_{i} + x^*_{i+m} )^2


    Where :math:`x^*_i` is a new time-series of length :math:`3N-4`
    derived from the original phase time-series :math:`x_n` of
    length :math:`N` by reflection at both ends.

    FIXME: better description of reflection operation.
    the original data x is in the center of x*:
    x*(1-j) = 2x(1) - x(1+j)  for j=1..N-2
    x*(i)   = x(i)            for i=1..N
    x*(N+j) = 2x(N) - x(N-j)  for j=1..N-2
    x* has length 3N-4
    tau = m*tau0

    NIST [SP1065]_ eqn (25) page 23

    FIXME: bias correction http://www.wriley.com/CI2.pdf page 5

    Parameters
    ----------
    phase: np.array
        Phase data in seconds. Provide either phase or frequency.
    frequency: np.array
        Fractional frequency data (nondimensional). Provide either
        frequency or phase.
    rate: float
        The sampling rate for phase or frequency, in Hz
    taus: np.array
        Array of tau values for which to compute measurement

    """
    return dev(dev_type='totdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def mtotdev(data: Array, rate: float = 1., data_type: str = "phase",
            taus: Taus = None, max_af: int = None) -> DevResult:
    """ PRELIMINARY - REQUIRES FURTHER TESTING.
        Modified Total deviation.
        Better confidence at long averages for modified Allan

    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.

    NIST [SP1065]_ eqn (27) page 25

    """
    return dev(dev_type='mtotdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def ttotdev(data: Array, rate: float = 1., data_type: str = "phase",
            taus: Taus = None, max_af: int = None) -> DevResult:
    """ Time Total Deviation

        Modified total variance scaled by tau^2 / 3

        NIST [SP1065]_ eqn (28) page 26.  Note that [SP1065]_ erroneously has tau-cubed here (!).
    """

    return dev(dev_type='ttotdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def htotdev(data: Array, rate: float = 1., data_type: str = "phase",
            taus: Taus = None, max_af: int = None) -> DevResult:
    """ PRELIMINARY - REQUIRES FURTHER TESTING.
        Hadamard Total deviation.
        Better confidence at long averages for Hadamard deviation
        
        Computed for N fractional frequency points y_i with sampling
        period tau0, analyzed at tau = m*tau0
        1. remove linear trend by averaging first and last half and divide by interval
        2. extend sequence by uninverted even reflection
        3. compute Hadamard for extended, length 9m, sequence.


    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.

    """
    return dev(dev_type='htotdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def theo1(data: Array, rate: float = 1., data_type: str = "phase",
          taus: Taus = None, max_af: int = None) -> DevResult:
    """ PRELIMINARY - REQUIRES FURTHER TESTING.
        Theo1 is a two-sample variance with improved confidence and
        extended averaging factor range. [Howe_theo1]_

        .. math::

            \\sigma^2_{THEO1}(m\\tau_0) = { 1 \\over  (m \\tau_0 )^2 (N-m) }
                \\sum_{i=1}^{N-m}   \\sum_{\\delta=0}^{m/2-1}
                {1\\over m/2-\\delta}\\lbrace
                    ({x}_{i} - x_{i-\\delta +m/2}) +
                    (x_{i+m}- x_{i+\\delta +m/2}) \\rbrace^2


        Where :math:`10<=m<=N-1` is even.

        FIXME: bias correction

        NIST [SP1065]_ eq (30) page 29

    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.

    """
    return dev(dev_type='theo1', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


# FIXME: implement TheoBR and TheoH


def mtie(data: Array, rate: float = 1., data_type: str = "phase",
         taus: Taus = None, max_af: int = None) -> DevResult:
    """ Maximum Time Interval Error.

    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.

    Notes
    -----
    this seems to correspond to Stable32 setting "Fast(u)"
    Stable32 also has "Decade" and "Octave" modes where the
    dataset is extended somehow?
    """
    return dev(dev_type='mtie', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def tierms(data: Array, rate: float = 1., data_type: str = "phase",
           taus: Taus = None, max_af: int = None) -> DevResult:
    """ Time Interval Error RMS.

    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.

    """
    return dev(dev_type='tierms', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


# TODO: eliminate by making all deviations gap resistant
def gradev(data, rate=1.0, data_type="phase", taus=None,
           ci=0.9, noisetype='wp'):
    """ gap resistant overlapping Allan deviation

    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional). Warning : phase data works better (frequency data is
        first trantformed into phase using numpy.cumsum() function, which can
        lead to poor results).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.
    ci: float
        the total confidence interval desired, i.e. if ci = 0.9, the bounds
        will be at 0.05 and 0.95.
    noisetype: string
        the type of noise desired:
        'wp' returns white phase noise.
        'wf' returns white frequency noise.
        'fp' returns flicker phase noise.
        'ff' returns flicker frequency noise.
        'rf' returns random walk frequency noise.
        If the input is not recognized, it defaults to idealized, uncorrelated
        noise with (N-1) degrees of freedom.

    Returns
    -------
    taus: np.array
        list of tau vales in seconds
    adev: np.array
        deviations
    [err_l, err_h] : list of len()==2, np.array
        the upper and lower bounds of the confidence interval taken as
        distances from the the estimated two sample variance.
    ns: np.array
        numper of terms n in the adev estimate.

    """
    if data_type == "freq":
        print("Warning : phase data is preferred as input to gradev()")
    phase = utils.input_to_phase(data, rate, data_type)
    (taus_used, m) = utils.tau_generator(data=phase, rate=rate,
                                               dev_type='gradev', taus=taus)

    ad = np.zeros_like(taus_used)
    ade_l = np.zeros_like(taus_used)
    ade_h = np.zeros_like(taus_used)
    adn = np.zeros_like(taus_used)

    for idx, mj in enumerate(m):
        (dev, deverr, n) = stats.calc_gradev(data,
                                             rate,
                                             mj,
                                             1,
                                             ci,
                                             noisetype)
        # stride=1 for overlapping ADEV
        ad[idx] = dev
        ade_l[idx] = deverr[0]
        ade_h[idx] = deverr[1]
        adn[idx] = n

    # Note that errors are split in 2 arrays
    return utils.remove_small_ns(taus_used, ad, ade_l, ade_h, adn)



