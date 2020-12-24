import math
import logging
import warnings
import numpy as np
from typing import List, Tuple, NamedTuple, Union, Callable
from . import tables
from . import allantools


# Spawn module-level logger
logger = logging.getLogger(__name__)

# shorten type hint to save some space
Array = np.ndarray

# group allowed taus types to save some space
Taus = Union[str, float, List, Array]

# define named tuple to hold averaging times and factors
TausResult = NamedTuple('TausResult', [('taus', Array), ('afs', Array)])


def binom(n: int, k: int) -> int:
    """Calculate binomial coefficient (n k)

    Args:
        n:  binomial power exponent
        k:  binomial term index

    Returns:
        binomial coefficient
    """

    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def decimate(data: Array, m: int, data_type: str) -> Array:
    """ Average phase or fractional frequency data at given averaging factor.

    Args:
        data:       data array of input measurements.
        m:          averaging factor for which to resample data.
        data_type:  input data type. Either `phase` or `freq`.

    Returns:
        array of input data sampled at the given averaging factor.
    """

    if data_type == 'phase':

        # Decimate
        z = data[::m]

    elif data_type == 'freq':

        # Trim to length if necessary
        z = data[:-(len(data) % m)] if data.size % m != 0 else data

        # Take average of every m measurements
        z = np.nanmean(z.reshape(-1, m), axis=1)

    else:

        raise ValueError(f"Invalid data_type value: {data_type}. Should be "
                         f"`phase` or `freq`.")

    return z


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


def frequency2phase(y: Array, rate: float, normalize: bool = False) -> Array:
    """Integrates fractional frequency data to output phase data (with
    arbitrary initial value), in units of second.

    Frequency to phase conversion is done by piecewise  integration  using
    the  averaging  time  as  the  integration  interval
    [RileyStable32Manual]_ (pg. 173-174):

    .. math:: x_{i+1} = x_i + y_i \\tau

    Any gaps in the frequency data are filled to obtain phase continuity.

    Args:
        y:          data array of fractional frequency measurements
        rate:       sampling rate of the input data, in Hz
        normalize:  remove average frequency before conversion

    Returns:
        time integral of fractional frequency data, i.e. phase (time) data
        in units of seconds. For phase in units of radians, see
        `phase2radians()`. Size: y.size + 1 - leading_and_trailing_gaps.size
    """

    # if meaningful data to convert (not empty)...
    if y.size > 0:

        if normalize:
            y -= np.nanmean(y)

        y = fill_gaps(y)

        x = np.cumsum(y) / rate

        # insert arbitrary 0 phase point for x_0
        x = np.insert(x, 0, 0.)

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


def input_to_phase(data: Array, rate: float, data_type: str,
                   normalize: bool = False) -> Array:
    """Takes either phase or fractional frequency data as input and returns
    phase.

    Args:
        data:       data array of input measurements.
        rate:       sampling rate of the input data, in Hz
        data_type:  input data type. Either `phase` or `freq`.
        normalize:  remove average frequency before conversion

    Returns:
        array of phase data.
    """

    if data_type == "phase":
        return data

    elif data_type == "freq":
        return frequency2phase(data, rate, normalize=normalize)

    else:
        raise ValueError(f"Invalid data_type value: {data_type}. Should be "
                         f"`phase` or `freq`.")


def tau_generator(data: Array, rate: float, dev_type: str,
                  taus: Taus = None,
                  maximum_m: int = None) -> TausResult:
    """Pre-processing of the list of averaging times requested by the user, at
    which to compute statistics.

    Does consistency checks, sorts, removes duplicates and invalid values.
    Generates a tau-list based on keywords 'all', 'decade', 'octave'.
    Uses 'octave' by default if no taus= argument is given.

    Averaging times may be integer multiples of the data sampling interval
    [RileyStable32]_ (Pg.3):

    .. math:: \\tau =  m \\tau_0

    where `m` is the averaging factor.

    This function implements a Stable32 stop-ratio for `octave` and `decade`
    stability runs, which automatically calculates the maximum averaging
    factor to use for the given deviation [RileyEvolution]_ (pg.9 Table III).

    If requesting averaging times for a `theo1` statistic, the resulting
    averaging times will be `effective` times tau = 0.75*tau

    Notes:

        With the Many Tau option, a selectable subset of the possible tau
        values is used to provide a quasi-uniform distribution of points on
        the stability plot adequate to give the appearance of a complete
        set, which can provide much faster calculating, plotting and printing

        The Stable32 implementation of `many`-taus was inspired by
        http://www.leapsecond.com/tools/adev3.c.

    Args:
        data:                   data array of phase or fractional frequency
                                measurements.
        rate:                   sampling rate of the input data, in Hz.
        dev_type:               type of deviation is going to be computed.
        taus (optional):        array of averaging times for which to compute
                                measurement. Can also be one of the keywords:
                                `all`, `many`, `octave`, `decade`. Defaults to
                                `octave`.
        maximum_m (optional):   maximum averaging factor for which to compute
                                measurement. Defaults to length of dataset,
                                and might not take effect if bigger than
                                automatic Stable32 stop-ratio.

    Returns:
        (taus, afs) NamedTuple of sanitised averaging times and averaging
        factors.
    """

    N = data.size

    if not N:
        raise ValueError("Cannot calculate averaging times on empty data.")

    maximum_m = N if maximum_m is None else maximum_m

    taus = 'octave' if taus is None else taus
    taus = taus if isinstance(taus, str) else np.array(taus)  # force to numpy

    # Consistency check sampling interval
    try:
        tau_0 = 1. / rate
    except ZeroDivisionError:
        logger.exception("Invalid data sampling rate %s Hz", rate)
        raise

    # Calculate requested averaging factors
    if isinstance(taus, str):

        if taus == "octave":
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

        elif taus == "many":
            # 1, 2, 3, 4, ... 71, 73, 75, 77, ..., 141, 143, 146, 149, 152, ...

            # FIXME: add effect of the many-tau `size` (=500) parameter

            density = 72  # afs per epoch

            afs = np.linspace(1, density, density)
            maxn = max(afs)

            i = 0
            while maxn < N:

                new = np.linspace((i + 1) * density + 1, (i + 2) * density,
                                  density)[i::i + 2]
                afs = np.concatenate([afs, new])

                maxn = max(afs)
                i += 1

            afs = afs.astype(int)

        elif taus == "all":

            # Too memory intensive for mtie. Set Stable32 `fastu` mode instead
            # to support algorithm based on binary decomposition
            if dev_type == "mtie":
                # 1, 3, 7, 15, 31, 63, 127, ...
                maxn = int(np.floor(np.log2(N)))
                afs = np.logspace(start=1, stop=maxn, num=maxn, base=2.0,
                                  dtype=int) - 1

            # Everyone else can use the real `all`
            else:
                # 1, 2, 3, 4, 5, 6, 7, ...
                afs = np.linspace(start=1, stop=N, num=N, dtype=int)

        else:
            raise ValueError(f"Invalid averaging mode selected: {taus}. "
                             f"Should be either `all`, `octave`, `decade`.")

    # Get closest integer averaging factors for requested averaging times
    else:

        taus = taus[~np.isnan(taus)]  # remove NaNs in input taus
        afs = np.array(taus) // tau_0
        afs = afs.astype(int)

    # Consistency checks for averaging factors:

    afs = afs[afs < N]  # make sure averaging time smaller than size of dataset
    afs = afs[afs > 0]  # make sure minimum averaging time is at least 1
    afs = afs[afs <= maximum_m]  # make sure afs within maximum allowed
    # Technically theo1 should have afs > 10. So fo example if `octave`: 16,
    # 32... Stable 32 decides instead to multiply by 10 the whole sequence
    # so we get 10, 20, 40, ...
    afs = afs*10 if (dev_type == 'theo1' and isinstance(taus, str)) else afs
    afs = afs[afs % 2 == 0] if dev_type == 'theo1' else afs

    # Apply a Stable32 `stop-ratio`. Only applies to 'octave' and 'decade'
    if isinstance(taus, str) and (taus == 'octave' or taus == 'decade'):

        stop_ratio = tables.STOP_RATIOS.get(dev_type)

        if not stop_ratio:
            raise KeyError(f"You provided an invalid {dev_type} "
                           f"dev_type for Stable32 stop-ratio.")

        stop_m = N // stop_ratio
        afs = afs[afs <= stop_m]

    afs = np.unique(afs)  # remove duplicates and sort

    # Recalculate averaging times, now sanitised.
    taus = tau_0*afs if dev_type != 'theo1' else 0.75*tau_0*afs

    if not afs.size:
        logger.warning("Could not generate valid averaging factors at which "
                       "to calculate deviation!")

    logger.debug("Averaging times: %s", taus)
    logger.debug("Averaging factors: %s", afs)

    return TausResult(taus=taus, afs=afs)


def tau_reduction(afs: Array, rate: float, n_per_decade: int) -> TausResult:
    """Reduce the number of averaging factors to maximum of n per decade.

    This is only useful if more than the "decade" and "octave" but
    less than the "all" taus are wanted. E.g. to show certain features of
    the data one might want 100 points per decade.

    NOTE: The algorithm is slightly inaccurate for ms under n_per_decade, and
    will also remove some points in this range, which is usually fine.

    Typical use would be something like:
    (data,m,taus)=tau_generator(data,rate,taus="all")
    (m,taus)=tau_reduction(m,rate,n_per_decade)

    Args:
        afs:            integer array of averaging factors (assumed to be an
                        "all" list) from which to remove points.
        rate:           sampling rate of the input data, in Hz.
        n_per_decade:   number of averaging factors to keep per decade.

    Returns:
        (taus, afs) NamedTuple of reduced averaging times and averaging
        factors.
    """

    # Consistency check sampling interval
    try:
        tau_0 = 1. / rate
    except ZeroDivisionError:
        logger.exception("Invalid data sampling rate %s Hz", rate)
        raise

    afs = afs.astype(int)

    keep = np.bool8(np.rint(n_per_decade*np.log10(afs[1:])) -
                    np.rint(n_per_decade*np.log10(afs[:-1])))

    afs = afs[:-1]     # Adjust ms size to fit above-defined mask

    assert len(afs) == len(keep)
    afs = afs[keep]

    taus = tau_0*afs

    return TausResult(taus=taus, afs=afs)


def remove_small_ns(afs: Array, taus: Array, ns: Array, vars: Array) \
        -> Tuple[Array, Array, Array, Array]:
    """ Remove results calculated on one or less samples.

    Args:
        afs:        array of averaging factors for which deviations were
                    computed
        taus:       array of averaging times for which deviation were computed
        ns:         array with number of analysis points for each deviation
        devs:       array of computed variances

    Returns:
        (afs, taus, ns, devs) identical to input, except that values with
        low ns have been removed.
    """

    mask = ns > 1

    # Filter out results
    vars = vars[mask]

    if not vars.size:
        logger.warning("Deviation calculated on too little samples. All "
                       "results have been filtered out!")

        raise UserWarning

    # Filter supporting arrays as well
    afs = afs[mask]
    taus = taus[mask]
    ns = ns[mask]

    return afs, taus, ns, vars


def three_cornered_hat_phase(x_ab: Array, x_bc: Array, x_ca: Array,
                             rate: float, taus: Union[Array, str],
                             dev_type: str) -> \
        Tuple[Array, Array, Array, Array, Array]:
    """
    Three Cornered Hat Method [Riley3CHat]_

    Given three clocks A, B, C, we seek to find their variances

    :math:`\\sigma^2_A`, :math:`\\sigma^2_B`, :math:`\\sigma^2_C`

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

    Args:
        x_ab:       phase measurements between clock A and B, in seconds
        x_bc:       phase measurements between clock B and C, in seconds
        x_ca:       phase measurements between clock C and A, in seconds
        rate:       sampling rate of the input data, in Hz.
        taus:       array of averaging times for which to compute measurement.
                    Can also be one of the keywords: `all`, `octave`, `decade`.
                    Defaults to `octave`.
        dev_type:   type of deviation to compute e.g. `oadev`

    Returns:
        (taus_ab, devs_a, err_a_lo, err_a_hi, ns_ab) Tuple of averaging times
        corresponding to output deviations, clock A deviations, clock A
        estimated errors, number of samples used to calculate each deviaiton.
    """

    tau_ab, dev_ab, _, _, ns_ab = getattr(allantools, dev_type)(
        data=x_ab, data_type='phase', rate=rate, taus=taus)

    _, dev_bc, _, _, _ = getattr(allantools, dev_type)(
        data=x_bc, data_type='phase', rate=rate, taus=taus)

    _, dev_ca, _, _, _ = getattr(allantools, dev_type)(
        data=x_ca, data_type='phase', rate=rate, taus=taus)

    var_ab = dev_ab * dev_ab
    var_bc = dev_bc * dev_bc
    var_ca = dev_ca * dev_ca
    assert len(var_ab) == len(var_bc) == len(var_ca)

    var_a = 0.5 * (var_ab + var_ca - var_bc)
    var_a[var_a < 0] = 0  # don't return imaginary deviations (?)

    dev_a = np.sqrt(var_a)
    err_a_lo = np.array([d/np.sqrt(nn) for (d, nn) in zip(dev_a, ns_ab)])
    err_a_hi = err_a_lo

    return tau_ab, dev_a, err_a_lo, err_a_hi, ns_ab


def rolling_grad(x: Array, n: int):
    """Adapted from:
    https://stackoverflow.com/questions/43288542/max-in-a-sliding-window-in-numpy-array"""

    def each_value(width=n):

        w = x[:width].copy()
        mx = np.nanmax(w)
        mn = np.nanmin(w)
        yield mx-mn

        i = 0
        j = width
        while j < x.size:

            oldValue = w[i]
            newValue = w[i] = x[j]

            if newValue > mx:
                mx = newValue
            elif oldValue == mx:
                mx = np.nanmax(w)

            if newValue < mn:
                mn = newValue
            elif oldValue == mn:
                mn = np.nanmin(w)

            yield mx-mn

            i = (i + 1) % width
            j += 1

    return np.array(list((each_value())))


def detrend(data: Array, data_type: str):
    """Detrend phase or factional frequency data.

    Args:
        data:       data array of input measurements.
        data_type:  input data type. Either `phase` or `freq`.

    Returns:
        detrended data.
    """

    if data_type == 'freq':
        deg = 1  # remove frequency drift
    elif data_type == 'phase':
        deg = 2  # remove frequency offset and drift)
    else:
        raise ValueError(f"Invalid data_type value: {data_type}. Should be "
                         f"`phase` or `freq`.")

    t = range(data.size)
    p = np.polyfit(t, data, deg)
    detrended = data - np.polyval(p, t)

    return detrended