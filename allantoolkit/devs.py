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
    """Container for stability analysis results. Defines the following fields:
    """

    afs: Array
    "array of averaging factors for which deviations were computed."
    taus: Array
    "array of corresponding averaging times, in seconds."
    ns: Array
    "array with number of analysis points used to compute each deviation."
    alphas: Array
    "array of estimated dominant noise type at each deviation."
    devs_lo: Array
    "array of estimated statistical lower bounds for each deviation."
    devs: Array
    "array with deviations computed at each averaging time."
    devs_hi: Array
    "array of estimated statistical higher bounds for each deviation."

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
        taus: Taus, max_af: int, alpha: int = None) -> DevResult:
    """This function implements the core pipeline common to all frequency
    stability analyses. It processes the input data and returns the appropriate
    frequency stability analysis results for the given deviation type.

    The preferred usage of this function is through one of its many wrappers
    offered in this module, each tailored to the calculation of a specific
    deviation type.

    Example:

        .. code-block:: python
            :emphasize-lines: 8

            import allantoolkit

            # Generate some white noise 1Hz phase data
            noise = allantoolkit.noise.white(1000)
            x = noise.data

            # Compute deviation of choice (e.g. oadev)
            out = allantoolkit.devs.oadev(x)

            print(out)

    Args:
        dev_type:   type of deviation to be computed, e.g. ``adev``.
        data:       array of phase (in units of seconds) or fractional
                    frequency data for which to calculate deviation.
        rate:       sampling rate of the input data, in Hz.
        data_type:  input data type. Either ``phase`` or ``freq``.
        taus:       array of averaging times for which to compute deviation.
                    Can also be one of the keywords: ``all``, ``many``,
                    ``octave``, ``decade``.
        max_af:     maximum averaging factor for which to compute deviation.
                    Defaults to length of dataset.
        alpha:      global dominant noise type. If ``None``, it is
                    automatically estimated at each averaging time.

    Returns:
        frequency stability analysis results, stored in a
        :class:`allantoolkit.devs.DevResult` NamedTuple.

    References:
        [RileyStable32Manual]_
    """

    # Make copy to avoid accidental input data modifications
    data = data.copy()

    # Easier to work with phase data, in units of seconds
    x = utils.input_to_phase(data=data, rate=rate, data_type=data_type,
                             normalize=True)

    # ---------------------
    # SET AVERAGING TIMES
    # ---------------------
    # generates [taus, afs]
    # ---------------------

    # Set a flag if calculating mtie on many taus
    flag_fastu = True if dev_type == 'mtie' and taus == 'all' else False

    # Build/Select averaging factors at which to calculate deviations
    taus, afs = utils.tau_generator(data=x, rate=rate, dev_type=dev_type,
                                    taus=taus, maximum_m=max_af)

    # ---------------------
    # CALC VARIANCES
    # ---------------------
    # generates [vars, ns]
    # ---------------------

    if dev_type == 'theo1':  # Fast batch calculation for theo1

        vars, ns = stats.calc_theo1_fast(x=x, rate=rate, explode=True)
        vars, ns = vars[afs], ns[afs]  # index out only selected AFS

    elif flag_fastu:  # Fast batch calculation for many-tau (Fast `u`) mtie
        vars, ns = stats.calc_mtie_fast(x=x, afs=afs, rate=rate)

    else:  # individual calculations at each averaging time for everyone else

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

    # Get rid of averaging times where dev calculated on too few samples (<= 1)
    afs, taus, ns, vars = utils.remove_small_ns(afs, taus, ns, vars)

    # Stop here if calculating MTIE or TIERMS. We don't need confidence
    # noise estimation, de-biasing, or confidence intervals
    if dev_type in ['mtie', 'tierms']:

        devs = np.sqrt(vars)

        nan_array = np.full(afs.size, np.NaN)
        return DevResult(afs=afs, taus=taus,  ns=ns, alphas=nan_array,
                         devs_lo=nan_array,  devs=devs, devs_hi=nan_array)

    # ----------------------------------
    # NOISE ID and DE-BIASING
    # ----------------------------------
    # generates [alphas], scales [vars]
    # ----------------------------------

    if dev_type == 'theo1' and alpha is None:  # Stable32 `TheoBR` conditions
        # batch noise id and de-biasing for theo1

        # Note that if users provide an `alpha` to the API, Stable32
        # calculates the bias using the hard-coded biasing factors in
        # bias.calc_bias_theo1() instead (so the else case below applies)

        # Calculate TheoBR correction factor
        kf = bias.calc_bias_theobr(x=x, rate=rate)

        # ID noise type for the whole run
        alpha = noise_id.noise_id_theoBR_fixed(kf=kf)
        alphas = np.full(afs.size, alpha)

        # De-bias theo1 to make TheoBR
        vars = kf*vars

    else:  # individual noise id at each averaging time for everyone else

        if alpha is None:  # need to auto ID noise

            # Initialise array for noise ID
            alphas = np.zeros(afs.size, dtype=int)

            for i, (m, n, var) in enumerate(zip(afs, ns, vars)):

                # Estimate Noise ID
                if i < afs.size - 1:  # Only estimate if not last averaging time
                    alphas[i] = noise_id.noise_id(data=x, m=m, rate=rate,
                                                  data_type='phase',
                                                  dev_type=dev_type, n=n)

                else:
                    # Use previous estimate at longest averaging time
                    alphas[i] = alphas[i-1]

        else:  # global dominant noise has been provided

            alphas = np.full(afs.size, fill_value=alpha, dtype=int)

        # Apply Bias Corrections based on noise type

        for i, (m, var, alpha) in enumerate(zip(afs, vars, alphas)):

            # Dispatch to appropriate bias calculator for this dev_type:
            # should be function of this signature: func(data, m, alpha) -> b

            bfunc = getattr(bias, 'calc_bias_' + dev_type.replace('dev', 'var'))
            b = bfunc(data=data, m=m, alpha=alpha)

            var *= b  # correct variance

            vars[i] = var

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
         taus: Taus = None, max_af: int = None, alpha: int = None) -> \
        DevResult:
    """Calculates the Allan deviation (ADEV) of phase or fractional
    frequency data.

    .. hint::

        Classic - use only if required - relatively poor confidence.

    The Allan deviation - :math:`\\sigma_y(\\tau)` - is the square root of
    the Allan variance. The Allan variance is the same as  the  ordinary
    variance  for  white  FM  noise, but  has  the  advantage,  for  more
    divergent  noise  types  such  as flicker  noise,  of  converging  to  a
    value  that  is  independent  on  the  number  of  samples.

    In terms of phase data, the Allan variance may be calculated as:

    .. math::

        \\sigma^2_y(\\tau) = { 1 \\over 2 (N-2) \\tau^2 }
        \\sum_{i=1}^{N-2} \\left[ x_{i+2} - 2x_{i+1} + x_{i} \\right]^2

    where :math:`x_i` is the :math:`i^{th}` of :math:`N` phase
    values spaced by an averaging time :math:`\\tau`.

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
       [RileyStable32]_ (5.2.2. Allan Variance, pg.14-5)
    """

    return dev(dev_type='adev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af, alpha=alpha)


def oadev(data: Array, rate: float = 1., data_type: str = "phase",
          taus: Taus = None, max_af: int = None, alpha: int = None) -> DevResult:
    """Calculates the overlapping Allan deviation (OADEV) of phase or
    fractional frequency data.

    .. hint::

        General Purpose - most widely used – first choice.

    The fully overlapping Allan deviation is the square root of the fully
    overlapping Allan variance: a form of the standard Allan variance that
    makes maximum use of a data set by forming all possible overlapping
    samples at each averaging time :math:`\\tau`.

    The overlapping Allan variance can be estimated from a set of :math:`N`
    phase measurements for averaging time :math:`\\tau = m\\tau_0`, where
    :math:`m` is the averaging factor and :math:`\\tau_0` is the basic
    data sampling period, by the following expression:

    .. math::

        \\sigma^{2}_y(\\tau) = { 1 \\over 2 \\tau^2 (N-2m) }
        \\sum_{i=1}^{N-2m} \\left[ {x}_{i+2m} - 2x_{i+m} + x_{i} \\right]^2

    For a time-series of :math:`M` fractional frequency values,
    the overlapping Allan variance is instead defined as:

    .. math::

        \\sigma^{2}_y(\\tau) = { 1 \\over 2 m^2 (M-2m+1) }
        \\sum_{j=1}^{M-2m+1} \\left\\{
        \\sum_{i=j}^{j+m-1} \\left[ y_{i+m} - y_i \\right]
        \\right\\}^2

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [RileyStable32]_ (5.2.4. Overlapping Allan Variance, pg.21-2)
    """

    return dev(dev_type='oadev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af, alpha=alpha)


def mdev(data: Array, rate: float = 1., data_type: str = "phase",
         taus: Taus = None, max_af: int = None, alpha: int = None) -> DevResult:
    """Calculates the modified Allan deviation (MDEV) of phase or
    fractional frequency data.

    .. hint::

        Used to distinguish WHPM and FLPM noise.

    The modified Allan deviation is the square root of the modified Allan
    variance (MVAR). The modified Allan variance is the same as the standard
    Allan variance at unity averaging factors, but includes an additional
    phase averaging operation, and has the advantage of being able to
    distinguish between white and flicker PM noise.

    The modified Allan variance is estimated from a set of
    :math:`N` phase measurements for averaging time :math:`\\tau =
    m\\tau_0`, where :math:`m` is the averaging factor and :math:`\\tau_0`
    is the basic data sampling period, by the following expression:

    .. math::

        \\sigma^{2}_y(\\tau) = { 1 \\over 2 m^2 \\tau^2 (N-3m+1) }
        \\sum_{j=1}^{N-3m+1} \\left\\{
        \\sum_{i=j}^{j+m-1} \\left[ {x}_{i+2m} - 2x_{i+m} + x_{i} \\right]
        \\right\\}^2

    For a time-series of :math:`M` fractional frequency values, the modified
    Allan variance is instead defined as:

    .. math::

        \\sigma^{2}_y(\\tau) = { 1 \\over 2 m^4 (M-3m+2) }
        \\sum_{j=1}^{M-3m+2} \\left\\{
        \\sum_{i=j}^{j+m-1} \\left(
        \\sum_{k=i}^{i+m-1} \\left[ y_{k+m} - y_k \\right]
        \\right)
        \\right\\}^2

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [RileyStable32]_ (5.2.5. Modified Allan Variance, pg.22-3)
    """

    return dev(dev_type='mdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af, alpha=alpha)


def tdev(data: Array, rate: float = 1., data_type: str = "phase",
         taus: Taus = None, max_af: int = None, alpha: int = None) -> DevResult:
    """Calculates the Time Allan deviation (TDEV) of phase or fractional
    frequency data.

    .. hint::

        Based on modified Allan variance. Use the Time Allan deviation to
        characterise the time error of a time source (clock) or distribution
        system.

    The Time Allan deviation is the square root of the Time Allan
    variance (TVAR). The Time Allan variance is a measure of stability based on
    the modified Allan variance, and is equal to the standard variance of the
    time deviations for white PM noise.

    The Time Allan variance is defined as:

    .. math::

        \\sigma^2_x( \\tau ) = { \\tau^2 \\over 3 } {\\textrm{MVAR}(\\tau)}

    where :math:`\\textrm{MVAR}(\\tau)` is the modified Allan variance of the
    data at averaging time :math:`\\tau`.

    Note that the Time Allan variance has units of seconds, and not fractional
    frequency.

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [RileyStable32]_ (5.2.6. Time Variance, pg.23-4)
    """

    return dev(dev_type='tdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af, alpha=alpha)


def hdev(data: Array, rate: float = 1., data_type: str = "phase",
         taus: Taus = None, max_af: int = None, alpha: int = None) -> DevResult:
    """Calculates the Hadamard deviation (HDEV) of phase or fractional
    frequency data.

    .. hint::

        Rejects frequency drift, and handles divergent noise. Use the Hadamard
        deviation to characterise frequency sources with divergent noise
        and/or frequency drift.

    The Hadamard deviation is the square root of the Hadamard Variance (HVAR).
    The Hadamard variance is a three-sample variance similar to the
    two-sample Allan variance, and is commonly applied for the analysis of
    frequency stability data that has highly divergent noise or linear
    frequency drift.

    In terms of phase data, the Hadamard variance may be calculated as:

    .. math::

        \\sigma^2_y(\\tau) = { 1 \\over 6 \\tau^2 (N-3)}
        \\sum_{i=1}^{N-3} \\left[
        x_{i+3} - 3x_{i+2} + 3x_{i+1} - x_i
        \\right]^2

    where :math:`x_i` is the :math:`i^{th}` of :math:`N` phase
    values spaced by an averaging time :math:`\\tau`.

    For a time-series of fractional frequency values, the Hadamard variance is
    defined as:

    .. math::

        \\sigma^{2}_y(\\tau) =  { 1 \\over 6 (M - 2) }
        \\sum_{i=1}^{M-2} \\left[
        \\bar{y}_{i+2} - 2\\bar{y}_{i+1} + \\bar{y}_i \\right]^2

    where :math:`\\bar{y}_i` is the :math:`i^{th}` of :math:`M`
    fractional frequency values averaged over the averaging time
    :math:`\\tau`.

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [RileyStable32]_ (5.2.8. Hadamard Variance, pg.25-6)
    """

    return dev(dev_type='hdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af, alpha=alpha)


def ohdev(data: Array, rate: float = 1., data_type: str = "phase",
          taus: Taus = None, max_af: int = None, alpha: int = None) -> DevResult:
    """Calculates the overlapping Hadamard deviation (OHDEV) of phase or
    fractional frequency data.

    .. hint::

        Better confidence than normal Hadamard.

    The overlapping Hadamard deviation is the square root of the
    overlapping Hadamard Variance (OHVAR). In the same way that the
    overlapping Allan variance makes the maximum use of a data set by forming
    all possible fully overlapping 2-sample pairs at each averaging time
    :math:`\\tau`, the overlapping Hadamard variance uses all 3-sample
    combinations.

    The overlapping Hadamard variance can be estimated from a set of :math:`N`
    phase measurements for averaging time :math:`\\tau = m\\tau_0`, where
    :math:`m` is the averaging factor and :math:`\\tau_0` is the basic
    data sampling period, by the following expression:

    .. math::

        \\sigma^{2}_y(\\tau) = { 1 \\over 6 \\tau^2 (N-3m) }
        \\sum_{i=1}^{N-3m} \\left[
        x_{i+3m} - 3x_{i+2m} + 3x_{i+m} - x_{i}
        \\right]^2

    For a time-series of :math:`M` fractional frequency values,
    the overlapping Allan variance is instead defined as:

    .. math::

        \\sigma^{2}_y(\\tau) = { 1 \\over 6 m^2 (M-3m+1) }
        \\sum_{j=1}^{M-3m+1} \\left\\{
        \\sum_{i=j}^{j+m-1} \\left[
        y_{i+2m} - 2y_{i+m} + y_i \\right]
        \\right\\}^2

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [RileyStable32]_ (5.2.9. Overlapping Hadamard Variance, pg.26-7)
    """

    return dev(dev_type='ohdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af, alpha=alpha)


def totdev(data: Array, rate: float = 1., data_type: str = "phase",
           taus: Taus = None, max_af: int = None, alpha: int = None) -> DevResult:
    """Calculates the Total deviation (TOTDEV) of phase or
    fractional frequency data.

    .. hint::

        Offers improved confidence at large averaging factors by extedning
        the data set by reflection at both ends.

    The total deviation is the square root of the total variance (TOTVAR).
    The total variance is similar to the two-sample or Allan variance,
    and has the same expected value, but offers improved confidence at long
    averaging times.

    The total variance can be estimated from a set of :math:`N` phase
    measurements for averaging time :math:`\\tau = m\\tau_0`, where
    :math:`m` is the averaging factor and :math:`\\tau_0` is the basic
    data sampling period, by the following expression:

    .. math::

        \\sigma^{2}_y(\\tau) = { 1 \\over 2 \\tau^2 (N-2) }
        \\sum_{i=2}^{N-1} \\left[
        x^*_{i-m} - 2x^*_{i} + x^*_{i+m}
        \\right]^2

    where the original :math:`N` phase values are extended by reflection
    about both endpoints to form a virtual sequence :math:`x^*` of length
    :math:`3N-4`, from :math:`i=3-N` to :math:`i=2N-2`. That is,
    the reflected portions added at each end have a 2-sample overlap each
    with the original dataset.

    TODO: Find and add definition for frequency data

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [RileyStable32]_ (5.2.11. Total Variance, pg.29-31)
    """

    return dev(dev_type='totdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af, alpha=alpha)


def mtotdev(data: Array, rate: float = 1., data_type: str = "phase",
            taus: Taus = None, max_af: int = None, alpha: int = None) -> DevResult:
    """Calculates the modified total deviation (MTOTDEV) of phase or
    fractional frequency data.

    .. hint::

        Better confidence at long averaging times for modified Allan
        deviation. The modified total deviation combines the features of the
        modified Allan and total deviations.

    The modified total deviation is the square root of the modified total
    variance (MTOTVAR). The modified total variance is similar to the
    modified Allan variance (MVAR), and has the same expected value,
    but offers improved confidence at long averaging times. It uses the same
    averaging technique as MVAR to distinguish between white and flicker PM
    noise.

    The modified total variance can be estimated from a set of :math:`N` phase
    measurements for averaging time :math:`\\tau = m\\tau_0`, where
    :math:`m` is the averaging factor and :math:`\\tau_0` is the basic
    data sampling period, by the following expression:

    .. math::

        \\sigma^{2}_y(\\tau) = { 1 \\over 2 \\tau^2 (N-3m+1) }
        \\sum_{n=1}^{N-3m+1} \\left\\{
        { 1 \\over 6m } \\sum_{i=n-3m}^{N+3m-1}
        \\left[ z^{\\#}_i(m) \\right]^2
        \\right\\}

    where the :math:`z^{\\#}_i(m)` terms are linear trend removed phase
    averages from triply-extended subsequences of the original phase data.

    TODO: Find and add definition for frequency data

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [RileyStable32]_ (5.2.12. Modified Total Variance, pg.31-2)
    """

    return dev(dev_type='mtotdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af, alpha=alpha)


def ttotdev(data: Array, rate: float = 1., data_type: str = "phase",
            taus: Taus = None, max_af: int = None, alpha: int = None) -> DevResult:
    """Calculates the time total deviation (TTOTDEV) of phase or fractional
    frequency data.

    .. hint::

        Better confidence at long averaging times. The time total deviation
        is a measure of time stability based on the modified total deviation.

    The time total deviation is the square root of the time total variance
    (TTOTVAR). The time total variance is a measure of stability based on
    the modified total variance, and is defined as:

    .. math::

        \\sigma^2_x( \\tau ) = { \\tau^2 \\over 3 } {\\textrm{MTOTVAR}(\\tau)}

    where :math:`\\textrm{MTOTVAR}(\\tau)` is the modified total variance of
    the data at averaging time :math:`\\tau`.

    Note that the time total variance has units of seconds, and not
    fractional frequency.

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [RileyStable32]_ (5.2.13. Time Total Variance, pg.33)
    """

    return dev(dev_type='ttotdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af, alpha=alpha)


def htotdev(data: Array, rate: float = 1., data_type: str = "phase",
            taus: Taus = None, max_af: int = None, alpha: int = None) -> DevResult:
    """Calculates the Hadamard total deviation (HTOTDEV) of phase or fractional
    frequency data.

    .. hint::

        The Hadamard total deviation combines the features of the Hadamard
        and total variances by rejecting linear frequency drift, handling
        more divergent noise types, and providing better confidence at large
        averaging factors.

    The Hadamard total deviation is the square root of the Hadamard total
    variance (HTOTVAR). The Hadamard total variance is a total version of
    the Hadamard variance. As such, it rejects linear frequency drift while
    offering improved confidence at large averaging times.

    The Hadamard total variance can be estimated from a set of :math:`M`
    fractional frequency measurements for averaging time :math:`\\tau =
    m\\tau_0`, where :math:`m` is the averaging factor and :math:`\\tau_0`
    is the basic data sampling period, by the following expression:

    .. math::

        \\sigma^{2}_y(\\tau) = { 1 \\over 6 (M-3m+1) }
        \\sum_{n=1}^{M-3m+1} \\left\\{
        { 1 \\over 6m } \\sum_{i=n-3m}^{N+3m-1}
        \\left[ H_i(m) \\right]^2
        \\right\\}

    where the :math:`H_i(m)` terms are the :math:`z^{\\#}_i(m)` linear trend
    removed Hadamard second differences from triply-extended subsequences of
    the original factional frequency data.

    TODO: Find and add definition for phase data

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [RileyStable32]_ (5.2.14. Hadamard Total Variance, pg.33-7)
    """

    return dev(dev_type='htotdev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af, alpha=alpha)


# PHASE-ONLY STATISTICS


def theo1(data: Array, rate: float = 1., data_type: str = "phase",
          taus: Taus = None, max_af: int = None, alpha: int = None) -> DevResult:
    """Calculates the (Bias Removed) Thêo1 deviation (THEOBR) of phase data.
    If fractional frequency data is provided, it is integrated to phase
    before processing.

    .. hint::

        Thêo1 is a 2-sample variance with improved confidence and extended
        averaging factor range.

    The Thêo1 deviation is the square root of the Thêo1 variance. The Thêo1
    statistic is a two-sample variance similar to the Allan variance that
    provides improved confidecne, and the ability to obtain a result for a
    maximum averaging time equal to :math:`75\\%` of the record length.

    The Thêo1 variance can be estimated from a set of :math:`N`
    phase measurements for even averaging factor :math:`m` where
    :math:`10 \\leq m \\leq N-1`, by the following expression:

    .. math::

        \\sigma^{2}_y(\\tau^*) =
        { 0.75 \\over \\tau^{*^2} (N-m) }
        \\sum_{i=1}^{N-m} \\sum_{\\delta=0}^{m/2 - 1}
        { 1 \\over m/2 - \\delta }
        \\left[
        \\left( x_i - x_{i-\\delta+m/2} \\right) +
        \\left( x_{i+m} - x_{i+\\delta+m/2} \\right)
        \\right]^2

    which applies to an effective averaging time :math:`\\tau^* =
    0.75m\\tau_0`, where :math:`\\tau_0` is the basic data sampling period.

    Automatic bias correction for a Thêo1 estimation is based on the average
    ratio of the Allan and Thêo1 variances over a range of averaging times.
    The de-biasing factor :math:`k_f` is given by:

    .. math::

        k_f = { 1 \\over n+1 }
        \\sum_{\\tau^*=9\\tau_0}^{(9+n)\\tau_0} { \\mathrm{AVAR(\\tau^*)}
        \\over \\mathrm{THEO1(\\tau^*)} }

    where :math:`n = \\lfloor {N \\over 6} - 3 \\rfloor`. Note that some
    implementations in the literature use instead
    :math:`n = \\lfloor {N \\over 30} - 3 \\rfloor`.

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [RileyStable32]_ (5.2.15-6. Thêo1, pg.37-41)
    """

    return dev(dev_type='theo1', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af, alpha=alpha)


# TODO: Implement TheoH (not really necessary as a Fast THEO1 (BR) removes the
#  fundamental reason for which TheoH was created)


def mtie(data: Array, rate: float = 1., data_type: str = "phase",
         taus: Taus = None, max_af: int = None, alpha: int = None) -> \
        DevResult:
    """Calculates the maximum time interval error (MTIE) of phase data. If
    fractional frequency data is provided, it is integrated to phase
    before processing.

    .. hint::

        MTIE is a measure of clock error commonly used in the
        telecommunications industry.

    The maximum time interval error is a measure of the maximum time error
    of a clock over a particular time interval. This statistic is very
    commonly used in the telecommunications industry. MTIE is a measure of
    the peak time deviation of a clock and is therefore very sensitive to a
    single extreme value, transient or outlier.

    MTIE is calculated by moving a `m`-point window (`m` being the
    averaging time of interest) through phase (time-error) data and finding
    the difference between the maximum and minimum values at each window
    position. MTIE is the overall maximum of this time interval error over
    the entire data set.

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [RileyStable32]_ (5.2.17. MTIE, pg.41-2)
    """

    return dev(dev_type='mtie', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def tierms(data: Array, rate: float = 1., data_type: str = "phase",
           taus: Taus = None, max_af: int = None, alpha: int = None) -> \
        DevResult:
    """Calculates rms time interval error (TIE rms) of phase data. If
    fractional frequency data is provided, it is integrated to phase
    before processing.

    The rms time interval error is another clock statistic commonly used by
    the telecommunication industry. For no frequency offset, TIE rms is
    approximately equal to the standard deviation of the integrated
    fractional frequency fluctuations. It is therefore similar in behaviour
    to the time Allan deviation (TDEV), although the latter properly
    identifies divergent noise types.

    The rms time interval error can be estimated from a set of
    :math:`N` phase measurements for averaging time :math:`\\tau =
    m\\tau_0`, where :math:`m` is the averaging factor and :math:`\\tau_0`
    is the basic data sampling period, by the following expression:

    .. math::

        \\sigma_y(\\tau) = \\sqrt{
        {1 \\over N-m} \\sum_{i=1}^{N-m} \\left[ x_{i+m} - x_i \\right]^2
        }

    .. seealso::
        Function :func:`allantoolkit.devs.dev` for detailed usage.

    References:
       [RileyStable32]_ (5.2.18. TIE rms, pg.42-3)
    """

    return dev(dev_type='tierms', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


