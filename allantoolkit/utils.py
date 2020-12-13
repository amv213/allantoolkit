import logging
import warnings
import numpy as np
from typing import List, Tuple, Union, Callable
from . import tables


# Spawn module-level logger
logger = logging.getLogger(__name__)

# shorten type hint to save some space
Array = np.ndarray


def trim_data(data: Array) -> Array:
    """Trim leading and trailing NaNs from dataset.

    This is done by browsing the array from each end and store the index of the
    first non-NaN in each case, the return the appropriate slice of the array

    Args:
        data:   data array with possible NaNs

    Returns:
        data array without leading and trailing NaNs
    """

    # input dataset may be empty or full of NaNs
    try:

        # Find indices for first and last valid data
        first = 0
        while np.isnan(data[first]):
            first += 1

        last = len(data)
        while np.isnan(data[last - 1]):
            last -= 1

        data = data[first:last]

    # in that case trimming means giving back empty array
    except IndexError:
        logger.exception('Error raised when trimming trailing and leading '
                         'gaps from data')
        data = np.array([])

    return data


def nan_helper(data: Array) -> Tuple[Array, Callable]:
    """Helper to handle indices and logical indices of NaNs.

    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array#6520696

    Args:
        data:   1d numpy array with possible NaNs

    Returns:
        logical indices of NaNs and a function, with signature indices = index(
        logical_indices), to convert logical indices of NaNs to 'equivalent'
        indices

    Usage:

        .. code-block:: python

           # linear interpolation of NaNs
           nans, x = nan_helper(y)
           y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    """

    return np.isnan(data), lambda z: z.nonzero()[0]


def fill_gaps(data: Array) -> Array:
    """Fill internal gaps in phase or frequency data by interpolation.

    Gaps may be filled in phase or frequency data by replacing them with
    interpolated values, by first removing any leading and trailing gaps,
    and then using the two values immediately before and after any interior
    gaps to determine linearly interpolated values within the gap
    [RileyStable32]_ (pg. 107).

    Args:
        data:   data array of frequency or phase measurements, with gaps
                represented as NaNs

    Returns:
        data array with interior gaps filled via linear interpolation.
    """

    # remove leading and trailing gaps
    data = trim_data(data)

    # locate 'inner' gaps
    nans, x = nan_helper(data)

    # try filling inner gaps
    try:
        data[nans] = np.interp(x=x(nans), xp=x(~nans), fp=data[~nans],
                               left=np.NaN, right=np.NaN)

    # No concept of 'gaps' (empty input dataset), return same dataset
    except ValueError:
        logger.exception("Error raised when interpolating gaps in data")

    return data


def phase2frequency(x: Array, rate: float) -> Array:
    """Convert phase data in units of seconds to fractional frequency data.

    Phase to frequency conversion is done by dividing the first differences
    of the phase points by the averaging time [RileyStable32Manual]_ (pg. 174):

    .. math:: y_i = ( x_{i+1} - x_i ) \\over \\tau

    Phase to frequency conversion is straightforward for data having
    gaps. Because two phase points  are  needed  to  determine  each
    frequency  point a single phase gap will cause two frequency gaps, and a
    gap of N phase points causes N+1 frequency gaps [RileyStable32]_ (pg. 108).

    Params:
        x:      data array of phase measurements, in seconds
        rate:   sampling rate of the input data, in Hz

    Returns:
        data array converted to fractional frequency. Size: x.size - 1
    """

    y = np.diff(x) * rate

    return y


def frequency2phase(y: Array, rate: float) -> Array:
    """Integrates fractional frequency data to output phase data (with
    arbitrary initial value), in units of second.

    Frequency to phase conversion is done by piecewise  integration  using
    the  averaging  time  as  the  integration  interval
    [RileyStable32Manual]_ (pg. 174):

    .. math:: x_{i+1} = x_i + y_i \\tau

    Any gaps in the frequency data are filled to obtain phase continuity.

    Args:
        y:      data array of fractional frequency measurements
        rate:   sampling rate of the input data, in Hz

    Returns:
        time integral of fractional frequency data, i.e. phase (time) data
        in units of seconds. For phase in units of radians, see
        `phase2radians()`. Size: y.size + 1 - leading_and_trailing_gaps.size
    """

    y = fill_gaps(y)

    # if meaningful data to convert (not empty)...
    if y.size > 0:

        x = np.cumsum(y) / rate

        # insert arbitrary 0 phase point for x_0
        x = np.insert(x, 0, 0)

    else:
        x = y

    return x


def frequency2fractional(f: Array, v0: float = None) -> Array:
    """ Convert frequency in Hz to fractional frequency [Wikipedia]_.

    .. math:: y(t) =  ( \\nu(t) - \\nu_0 )  \\over \\nu_0

    Args:
        f:              data array of frequency in Hz.
        v0 (optional):  nominal oscillator frequency, in Hz. Defaults to mean
                        frequency of dataset.

    Returns:
        array of fractional frequency data, y.
    """

    # If data has only NaNs it might rise a warning if calculating the mean
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        mu = np.nanmean(f) if v0 is None else v0

    y = (f - mu) / mu

    return y


def phase2radians(x: Array, v0: float) -> Array:
    """ Convert array of phases in seconds to equivalent phases in radians.

    Args:
        x:  data array of phases, in seconds.
        v0: nominal oscillator frequency, in Hz.

    Returns:
        array of phase data, in radians.
    """

    phase_data = 2*np.pi*v0 * x

    return phase_data


def input_to_phase(data: Array, rate: float, data_type: str) -> Array:
    """Takes either phase or fractional frequency data as input and returns
    phase.

    Args:
        data:       data array of input measurements.
        rate:       sampling rate of the input data, in Hz
        data_type:  input data type. Either `phase` or `freq`.

    Returns:
        array of phase data.
    """

    if data_type == "phase":
        return data

    elif data_type == "freq":
        return frequency2phase(data, rate)

    else:
        raise ValueError(f"Invalid data_type value: {data_type}. Should be "
                         f"`phase` or `freq`.")


def tau_generator(data: Array, rate: float, dev_type: str,
                  taus: Union[Array, str] = None,
                  even: bool = False,
                  maximum_m: int = None) -> Tuple[Array, Array, Array]:
    """Pre-processing of the list of averaging times requested by the user, at
    which to compute statistics.

    Does consistency checks, sorts data, removes duplicates and invalid values.
    Generates a tau-list based on keywords 'all', 'decade', 'octave'.
    Uses 'octave' by default if no taus= argument is given.

    Averaging times may be integer multiples of the data sampling interval
    [RileyStable32]_ (Pg.3):

    .. math:: \\tau =  m \\tau_0

    where `m` is the averaging factor.

    Args:
        data:               data array of phase or fractional frequency
                            measurements.
        rate:               sampling rate of the input data, in Hz.
        dev_type:           type of deviation is going to be computed.
        taus (optional):    array of averaging times for which to compute
                            measurement. Can also be one of the keywords:
                            `all`, `octave`, `decade`. Defaults to `octave`.
        even (optional):    If True, allows only averaging times which are
                            even multiples of the sampling interval.
        maximum_m (optional):  maximum averaging factor for which to compute
                            measurement. Defaults to length of dataset

    Returns:
        (data, afs, taus) Tuple of sanitised input data, averaging factors,
        and sanitised averaging times.
    """

    N = data.size

    maximum_m = N if maximum_m is None else maximum_m
    taus = 'octave' if taus is None else taus

    # Consistency check sampling interval
    try:
        tau_0 = 1. / rate
    except ZeroDivisionError:
        logger.exception("Invalid data sampling rate %s Hz", rate)
        raise

    # Calculate requested averaging factors
    if isinstance(taus, str):

        if taus == "all":
            # 1, 2, 3, 4, 5, 6, 7, ...
            afs = np.linspace(start=1, stop=N, num=N + 1, dtype=int)

        elif taus == "octave":
            # 1, 2, 4, 8, 16, 32, 64, ...
            maxn = int(np.floor(np.log2(N)))
            afs = np.logspace(start=0, stop=maxn, num=maxn + 1, base=2.0,
                              dtype=int)

        elif taus == "decade":
            # 1, 2, 4, 10, 20, 40, 100, ...
            maxn = int(np.floor(np.log10(N)))
            pwrs = 10 ** np.arange(maxn + 1)
            afs = np.outer(np.array([1, 2, 4]), pwrs).flatten(
                order='F').astype(int)

        else:
            raise ValueError(f"Invalid averaging mode selected: {taus}. "
                             f"Should be either `all`, `octave` or `decade`.")

    # Get closest integer averaging factors for requested averaging times
    else:
        print(taus)
        afs = np.array(taus) // tau_0
        afs = afs.astype(int)

    # FIXME: technically these should be ints, but then toolkit's devs
    #  fail to calculate with the correct precision. Find where the type
    #  casting fails and then come back here and keep them as ints
    afs = afs.astype(float)

    afs = afs[afs < N]  # make sure averaging time smaller than size of dataset
    afs = afs[afs > 0]  # make sure minimum averaging time is at least 1
    afs = afs[afs <= maximum_m]  # make sure afs within maximum allowed
    afs = afs[afs % 2 == 0] if even else afs  # only keep even afs if requested
    afs = np.unique(afs)    # remove duplicates and sort

    # FIXME: should we use a "stop-ratio" like Stable32
    if taus == 'octave' or taus == 'decade':
        stop_ratio = tables.STOP_RATIOS.get(dev_type)

        if not stop_ratio:
            raise NotImplementedError(f"You provided a {dev_type} dev_type")

        af_max = N / stop_ratio
        #afs = afs[afs <= af_max]

    logger.debug("tau_generator: averaging factors are %s", afs)

    if afs.size == 0:
        print("Warning: sanity-check on tau failed!")
        print("   len(data)=", len(data), " rate=", rate, "taus= ", taus)

    taus = tau_0*afs

    return data, afs, taus


def tau_reduction(ms: np.ndarray, rate: float, n_per_decade: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """Reduce the number of taus to maximum of n per decade (Helper function)

    takes in a tau list and reduces the number of taus to a maximum amount per
    decade. This is only useful if more than the "decade" and "octave" but
    less than the "all" taus are wanted. E.g. to show certain features of
    the data one might want 100 points per decade.

    NOTE: The algorithm is slightly inaccurate for ms under n_per_decade, and
    will also remove some points in this range, which is usually fine.

    Typical use would be something like:
    (data,m,taus)=tau_generator(data,rate,taus="all")
    (m,taus)=tau_reduction(m,rate,n_per_decade)

    Parameters
    ----------
    ms: array of integers
        List of m values (assumed to be an "all" list) to remove points from.
    rate: float
        Sample rate of data in Hz. Time interval between measurements
        is 1/rate seconds. Used to convert to taus.
    n_per_decade: int
        Number of ms/taus to keep per decade.

    Returns
    -------
    m: np.array
        Reduced list of m values
    taus: np.array
        Reduced list of tau values
    """
    ms = np.int64(ms)
    keep = np.bool8(np.rint(n_per_decade*np.log10(ms[1:])) -
                    np.rint(n_per_decade*np.log10(ms[:-1])))
    # Adjust ms size to fit above-defined mask
    ms = ms[:-1]
    assert len(ms) == len(keep)
    ms = ms[keep]
    taus = ms/float(rate)

    return ms, taus


def remove_small_ns(taus: np.ndarray, devs: np.ndarray,
                    deverrs: Union[List[np.ndarray], np.ndarray],
                    ns: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                             Union[List[np.ndarray], np.ndarray],
                                             np.ndarray]:
    """ Remove results with small number of samples.
        If n is small (==1), reject the result

    Parameters
    ----------
    taus: array
        List of tau values for which deviation were computed
    devs: array
        List of deviations
    deverrs: array or list of arrays
        List of estimated errors (possibly a list containing two arrays :
        upper and lower values)
    ns: array
        Number of samples for each point

    Returns
    -------
    (taus, devs, deverrs, ns): tuple
        Identical to input, except that values with low ns have been removed.

    """
    ns_big_enough = ns > 1

    o_taus = taus[ns_big_enough]
    o_devs = devs[ns_big_enough]
    o_ns = ns[ns_big_enough]
    if isinstance(deverrs, list):
        assert len(deverrs) < 3
        o_deverrs = [deverrs[0][ns_big_enough], deverrs[1][ns_big_enough]]
    else:
        o_deverrs = deverrs[ns_big_enough]
    if len(o_devs) == 0:
        print("remove_small_ns() nothing remains!?")
        raise UserWarning

    return o_taus, o_devs, o_deverrs, o_ns


def three_cornered_hat_phase(phasedata_ab, phasedata_bc,
                             phasedata_ca, rate, taus, function):
    """
    Three Cornered Hat Method

    Given three clocks A, B, C, we seek to find their variances
    :math:`\\sigma^2_A`, :math:`\\sigma^2_B`, :math:`\\sigma^2_C`.
    We measure three phase differences, assuming no correlation between
    the clocks, the measurements have variances:

    .. math::

        \\sigma^2_{AB} = \\sigma^2_{A} + \\sigma^2_{B}

        \\sigma^2_{BC} = \\sigma^2_{B} + \\sigma^2_{C}

        \\sigma^2_{CA} = \\sigma^2_{C} + \\sigma^2_{A}

    Which allows solving for the variance of one clock as:

    .. math::

        \\sigma^2_{A}  = {1 \\over 2} ( \\sigma^2_{AB} +
        \\sigma^2_{CA} - \\sigma^2_{BC} )

    and similarly cyclic permutations for :math:`\\sigma^2_B` and
    :math:`\\sigma^2_C`

    Parameters
    ----------
    phasedata_ab: np.array
        phase measurements between clock A and B, in seconds
    phasedata_bc: np.array
        phase measurements between clock B and C, in seconds
    phasedata_ca: np.array
        phase measurements between clock C and A, in seconds
    rate: float
        The sampling rate for phase, in Hz
    taus: np.array
        The tau values for deviations, in seconds
    function: allantoolkit deviation function
        The type of statistic to compute, e.g. allantoolkit.oadev

    Returns
    -------
    tau_ab: np.array
        Tau values corresponding to output deviations
    dev_a: np.array
        List of computed values for clock A

    References
    ----------
    http://www.wriley.com/3-CornHat.htm
    """

    (tau_ab, dev_ab, err_ab, ns_ab) = function(phasedata_ab,
                                               data_type='phase',
                                               rate=rate, taus=taus)
    (tau_bc, dev_bc, err_bc, ns_bc) = function(phasedata_bc,
                                               data_type='phase',
                                               rate=rate, taus=taus)
    (tau_ca, dev_ca, err_ca, ns_ca) = function(phasedata_ca,
                                               data_type='phase',
                                               rate=rate, taus=taus)

    var_ab = dev_ab * dev_ab
    var_bc = dev_bc * dev_bc
    var_ca = dev_ca * dev_ca
    assert len(var_ab) == len(var_bc) == len(var_ca)
    var_a = 0.5 * (var_ab + var_ca - var_bc)

    var_a[var_a < 0] = 0 # don't return imaginary deviations (?)
    dev_a = np.sqrt(var_a)
    err_a = [d/np.sqrt(nn) for (d, nn) in zip(dev_a, ns_ab)]

    return tau_ab, dev_a, err_a, ns_ab
