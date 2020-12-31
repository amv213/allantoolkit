import math
import gzip
import logging
import warnings
import numpy as np
from . import tables
from . import noise_id
from . import devs
from scipy import signal
from pathlib import Path
from typing import List, Tuple, NamedTuple, Union, Callable

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


def is_power(z: Array) -> Array:
    """Checks whether the input array is an array of powers of 2.

    Args:
        z:  input array

    Returns:
        boolean array flagging powers of two as ``True``.

    References:
        https://stackoverflow.com/questions/29480680/finding-if-a-number-is-a-power-of-2-using-recursion
    """
    return np.logical_and(np.bitwise_and(z, (z - 1)) == 0, z != 0)


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

    # if meaningful data to convert (not empty or all gaps)...
    if y.size > 0 and y[~np.isnan(y)].size > 0:

        if normalize:
            y -= np.nanmean(y)

        y = fill_gaps(y)

        x = np.cumsum(y) / rate

        # insert arbitrary 0 phase point for x_0
        x = np.insert(x, 0, 0.)

    else:
        x = np.array([])

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

            # TODO: this is all calibrated for the many-tau `size` (=500)
            #  parameter (in Stable32 config options). I don't know how
            #  other values play out

            # max_af per epoch
            density = 53 if 'tot' in dev_type else 71

            afs = np.arange(1, stop=density, step=1)

            epoch = 1
            while epoch * density < N:

                last = afs[-1]
                new = np.arange(start=last + epoch, stop=(epoch + 1) * density,
                                step=epoch + 1)
                afs = np.concatenate([afs, new])

                epoch += 1

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
    afs = afs[afs > 0]  # make sure minimum averaging time is at least 1

    if dev_type == 'theo1':

        # Match Stable32 in-builts...
        if isinstance(taus, str):
            afs = afs*2 if taus == 'many' else afs*10

        afs = afs[afs >= 10]
        afs = afs[afs % 2 == 0]

    afs = afs[afs < N]  # make sure averaging time smaller than size of dataset
    afs = afs[afs <= maximum_m]  # make sure afs within maximum allowed

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


def scale(data: Array, addend: float = 0., multiplier: float = 1.,
              slope: float = 0., reverse: bool = False) -> Array:
    """Scales the selected data by an additive or multiplicative factor,
    by adding a linear slope, or by reversing the data.

    References:
        [RileyStable32Manual]_ (Fill Function, pg.181-2)

    Args:
        data:       data array of phase or frequency measurements.
        addend:     additive factor to be added to the data.
        multiplier: multiplicative factor by which to scale the data.
        slope:      linear slope by which to scale the data.
        reverse:    if `True` reverses the data, after scaling it.
    """

    # Scale
    scaled_data = (data*multiplier) + addend + slope*np.arange(data.size)

    # Reverse
    scaled_data = scaled_data[::-1] if reverse else scaled_data

    return scaled_data


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

    out_ab = getattr(devs, dev_type)(
        data=x_ab, data_type='phase', rate=rate, taus=taus)

    out_bc = getattr(devs, dev_type)(
        data=x_bc, data_type='phase', rate=rate, taus=taus)

    out_ca = getattr(devs, dev_type)(
        data=x_ca, data_type='phase', rate=rate, taus=taus)

    var_ab = out_ab.devs**2
    var_bc = out_bc.devs**2
    var_ca = out_ca.devs**2
    assert len(var_ab) == len(var_bc) == len(var_ca)

    var_a = 0.5 * (var_ab + var_ca - var_bc)
    var_a[var_a < 0] = 0  # don't return imaginary deviations (?)

    dev_a = np.sqrt(var_a)
    err_a_lo = np.array([d/np.sqrt(nn) for (d, nn) in zip(dev_a, out_ab.ns)])
    err_a_hi = err_a_lo

    return out_ab.taus, dev_a, err_a_lo, err_a_hi, out_ab.ns


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


def replace_outliers(data: Array, sigmas: float = 3.5,
                     replace: str = None) -> Array:
    """Check for and remove outliers from frequency data. Outliers are
    replaced by gaps (NaNs).

    Outliers are detected using the median absolute deviation (MAD). The
    MAD is a robust statistic based on the median of the data. It is the
    median of the scaled absolute deviation of the data points from their
    median value.

    References:

        [RileyStable32Manual]_ (Check Function, pg.189-90)

        [RileyStable32]_ (10.11, pg.108-9)

        https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list

        https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

    Args:
        data:       data array of phase or frequency measurements.
        sigmas:     desired number of deviations for which a point is to be
                    classified as outlier. Defaults to 3.5
        replace:    whether to replace the detected outliers, and how.
                    If set to `all`, all outliers are replaced. If set to
                    `largest`, only the largest outlier is removed. If
                    not set, outliers are not removed but only logged.

    Returns:
        input data with (optionally) outliers replaced with NaNs.
    """

    # Absolute deviation of the data points from their median
    d = np.abs(data - np.median(data))

    # 1-sigma MAD
    mad = np.median(d / 0.6745)

    # Flag outliers
    outliers_idxs = np.flatnonzero(d > sigmas*mad)

    logger.info("Detected %i outliers (#%s)", np.sum(outliers_idxs.size),
                outliers_idxs+1)

    # Replace outliers with gaps
    if replace is not None:

        if replace == 'all':

            data[outliers_idxs] = np.NaN

            logger.info("\tRemoved all outliers")

        elif replace == 'largest':

            idx_max_outlier = outliers_idxs[np.argmax(d[outliers_idxs])]

            data[idx_max_outlier] = np.NaN

            logger.info("\tRemoved largest outliers (#%i)",
                        idx_max_outlier+1)

        else:
            raise ValueError(f"Gap replacement method `{replace}` not "
                             f"recognised.")

    return data


def filter(data: Array, rate: float, type: str, f_low: float = None,
           f_high: float = None) -> Array:
    """Applies a first-order Butterworth filter to an array of phase or
    frequency data.

    TODO: check scaling of frequencies for Wn filter parameter is correct

    References:
        [RileyStable32Manual]_ (Filter Function, pg.185-6)

    Args:
        data:       data array of phase or frequency measurements.
        rate:           sampling rate of the input data, in Hz.
        type:   the type of filter. Can be any of {'lowpass', 'highpass',
                'bandpass', 'bandstop'}
        f_low:  for a 'highpass' or 'band*' filter, the lower cutoff
                frequency.
        f_high: for a 'lowpass' or 'band*' filter, the higher cutoff
                frequency.

    Returns:
        filtered input data
    """

    # Check user provided relevant cutoff frequencies
    if type != 'lowpass' and f_low is None:
        raise ValueError(f"Need to provide a `f_low` cutoff frequency for "
                         f"{type} filter.")
    elif type != 'highpass' and f_high is None:
        raise ValueError(f"Need to provide a `f_high` cutoff frequency "
                         f"for {type} filter.")

    # Zero-pad data to next power of two
    N = data.size
    next_pow = int(np.ceil(np.log2(N)))
    padded = np.concatenate((data, np.zeros(2**next_pow - N)))
    N = padded.size

    # Range of Fourier frequencies: frequency bin size to Nyquist frequency
    f_min, f_max = rate / N, rate / 2

    # Check cutoff frequencies are in right range
    if f_low is not None:
        if f_low >= f_max:
            raise ValueError(f"`f_low` must be smaller than max Nyquist "
                             f"frequency {f_max} Hz")

    if f_high is not None:
        if f_high <= f_min:
            raise ValueError(f"`f_high` must be bigger than min "
                             f"frequency bin {f_min} Hz")
        elif f_high >= f_max:
            raise ValueError(f"`f_high` must be smaller than max Nyquist "
                             f"frequency {f_max} Hz")

    # Critical frequencies for filter
    if type == 'lowpass':

        freqs = f_high

    elif type == 'highpass':

        freqs = max(f_min, f_low)

    else:
        freqs = np.array([max(f_min, f_low), f_high])

    # It is recommended to use second-order sections format (ref. scipy)
    # Also need to normalise frequencies to Nyquist frequency
    sos = signal.butter(N=1, Wn=freqs/f_max, btype=type, output='sos')
    filtered = signal.sosfilt(sos=sos, x=data)

    logger.info("\n\n%s FILTER:\n"
                "\tMax:      \t%.5e Hz\n"
                "\tMin:      \t%.5e Hz\n"
                "\tApplied:  \t%s Hz\n", type.upper(), f_max, f_min,
                freqs)

    return filtered


# TODO: Finish implementing
def drift(data: Array, rate: float, data_type: str, type: str = None,
          m: int = 1, remove: bool = False) -> Array:
    """Analyzes phase or frequency data for frequency drifts, or finds
    frequency offsets in phase data.

    Four frequency drift methods and three frequency offset methods are
    available for phase data:

    - `quadratic`:  least-squares quadratic fit. Optimal for WHPM noise.
    - `avg 2diff`:  average of 2nd differences. Optimal for RWFM noise.
    - `3-point`:    three-points average. Equivalent of the bisection model
                    for frequency data. Good for WHFM and RWFM noise.
    - `greenhall`:  Greenhall 4-points average. Applicable to all noise types.


    - `linear`:     least-squares linear fit. Optimal for WHPM noise.
    - `avg 1diff`:  average of 1st differences. Optimal for WHFM noise.
    - `endpoints`:  difference between the first and last points of the
                    data. This method is intended mainly to match the two
                    endpoints to condition data for a TOTVAR analysis.

    Five drift methods and one autoregression frequency drift method
    are available for frequency data:

    - `linear`:         least-squares linear fit. Optimal for WHFM noise.
    - `bisection`:      drift computation from the frequency averages over
                        the first and last halves of the data. Optimal for
                        WHFM and RWFM noise.
    - `log`:            log model fit.
    - `diffusion`:      diffusion model.

    - `autoregressive`: fit and remove AR(1) autoregressive noise from the
                        data. Useful for prewhitening data before a jump
                        analysis.

    FIXME: Need to make this robust to floating point numerical precision
    errors. This happens in particular when calculating the mean of small
    values - resulting in erroneous results at the e-32 level

    References:
        [RileyStable32Manual]_ (Drift Function, pg.191-4)

        [RileyStable32]_ (5.18-21, pg.67-70)

        http://www.wriley.com/Frequency%20Drift%20Characterization%20in%20Stable32.pdf

        C. A. Greenhall, "A frequency-drift estimator and its removal
        from modified Allan variance," Proceedings of International
        Frequency Control Symposium, Orlando, FL, USA, 1997,
        pp. 428-432, doi: 10.1109/FREQ.1997.638639.

    Args:
        data:       data array of phase or frequency measurements.
        rate:       sampling rate of the input data, in Hz
        data_type:  input data type. Either `phase` or `freq`.
        type:       drift analysis type. Defaults to `quadratic` for phase
                    data, and `linear` for frequency data.
        m:          averaging factor to be used for the log, diffusion or
                    autoregression drift analysis models
        remove:     if `True` remove the drift from the data. If `False`
                    only logs drift analysis results

    Returns:
        input data with (optionally) drift removed.
    """

    logger.warning("`Drift` function might be affected by numerical precision "
                   "limits. Apologies if that is the case.")

    if data_type not in ['phase', 'freq']:
        raise ValueError(f"Invalid data_type value: {data_type}. Should be "
                         f"`phase` or `freq`.")

    # Assign default drift analysis method for phase and frequency data
    if type is None:
        type = 'quadratic' if data_type == 'phase' else 'linear'

    N = data.size

    # 'x' support vector onto which to fit drifts
    t = np.arange(1, data.size + 1)

    # PHASE DATA DRIFT ANALYSIS
    if data_type == 'phase':

        if type == 'quadratic':

            coeffs = np.polyfit(t, data, deg=2)
            a, b, c = coeffs[::-1]
            slope = 2 * c * rate

            logger.info("\nQuadratic")
            logger.info("a=%.7g\nb=%.7g\nc=%.7g", a, b, c)
            logger.info("%.7g", slope)

            if remove:
                data -= np.polyval(coeffs, t)

        elif type == 'avg 2diff':

            slope = data[2:] - 2 * data[1:-1] + data[:-2]
            slope = np.mean(slope)

            logger.info("\nAvg of 2nd Diff")
            logger.info("%.7g", slope)

        elif type == '3-point':

            M = data.size
            slope = 4 * (data[-1] - 2 * data[M // 2] + data[0])
            slope /= (M - 1) ** 2

            logger.info("\n3-point")
            logger.info("%.7g", slope)

        elif type == 'greenhall':  # see reference for details

            def w(n, w0=0):
                return w0 + np.sum(data[:n])

            w_N = np.sum(data)
            n1 = int(N / 10)
            r1 = n1 / N

            slope = 6. / (N ** 3 * (1/rate) ** 2 * r1 * (1 - r1)) * (
                    w_N - (w(N - n1) - w(n1)) / (1 - 2 * r1))

            logger.info("\nGreenhall")
            logger.info("%.7g", slope)

        # FREQUENCY OFFSETS

        elif type == 'linear':

            coeffs = np.polyfit(t, data, deg=1)
            a, b = coeffs[::-1]
            slope = b
            f_offset = slope * rate

            logger.info("\nLinear")
            logger.info("a=%.7g\nb=%.7g", a, b)
            logger.info("slope=%.7g", slope)
            logger.info("f_offset=%.7g", f_offset)

            if remove:  # removing frequency offset
                data -= slope * t

        elif type == 'avg 1diff':

            slope = np.diff(data)

            slope = np.mean(slope)

            logger.info("\nAvg of 1st Diff")
            logger.info("slope=%.7g", slope)
            logger.info("f_offset=%.7g", slope * rate)

            if remove:  # removing frequency offset
                data -= slope * t

        elif type == 'endpoints':

            slope = (data[-1] - data[0]) / (N - 1)

            logger.info("\nEndpoints")
            logger.info("slope=%.7g", slope)
            logger.info("f_offset=%.7g", slope * rate)

            if remove:  # removing frequency offset
                data -= slope * t

        else:

            raise ValueError(f"`{type}` drift analysis method is not "
                             f"available for phase data")

    # FREQUENCY DATA DRIFT ANALYSIS
    else:

        if type == 'linear':

            coeffs = np.polyfit(t, data, deg=1)
            a, b = coeffs[::-1]
            slope = b

            logger.info("\nLinear")
            logger.info("a=%.7g\nb=%.7g", a, b)
            logger.info("%.7g", slope)

            if remove:
                data -= np.polyval(coeffs, t)

        elif type == 'bisection':

            # calc mean of halves
            w = N // 2  # width of a `half`

            # mean of `halves`
            mu1, mu2 = np.nanmean(data[:w]), np.nanmean(data[-w:])

            # calc slope (extra sep if odd n samples)
            slope = (mu2 - mu1) / (w + (N % 2))

            logger.info("\nBisection")
            logger.info("%.7g", slope)

        # FIXME: this doesn't work yet. Probably need better param
        #  guessing to give matching results
        elif type == 'log':

            raise NotImplementedError("Log drift analysis still "
                                      "needs improvements...")

            # Fitting function
            def func(t, a, b, c):
                return a * np.log(b * t + 1) + c

            z = decimate(data=data, data_type=data_type, m=m)

            t = np.arange(1, z.size + 1)

            popt, pcov = curve_fit(func, xdata=t, ydata=z)

            logger.info("\nLog")
            logger.info("a=%.7g\nb=%.7g\nc=%.7g", *popt)

            # TODO: remove drift from data

        # FIXME: this doesn't work yet. Probably need better param
        #  guessing to give matching results
        elif type == 'diffusion':

            raise NotImplementedError("Diffusion drift analysis still "
                                      "needs improvements...")

            # Fitting function
            def func(t, a, b, c):
                return a + b * np.sqrt(t + c)

            z = decimate(data=data, data_type=data_type, m=m)

            t = np.arange(1, z.size + 1)

            popt, pcov = curve_fit(func, xdata=t, ydata=z)

            logger.info("\nDiffusion")
            logger.info("a=%.7g\nb=%.7g\nc=%.7g", *popt)

            # TODO: remove drift from data

        # FIXME: this doesn't give exactly the same results as Stable32 (
        #  but it's quite close)
        elif type == 'autoregression':

            z = decimate(data=data, data_type=data_type, m=m)

            r1 = noise_id.acf(z=z, k=1)

            logger.info("\nAutoregression")
            logger.info("%.3f", r1)

            if remove:
                data = data[1:] - r1 * data[:-1]

        else:

            raise ValueError(f"`{type}` drift analysis method is not "
                             f"available for frequency data")

    return data


def read_datafile(fn: Union[str, Path]) -> Array:
    """Extract phase or frequency data from an input .txt file (optionally
    compressed to .gz) or .DAT file.

    If present, a first column with associated timestamps will be omitted.
    Lines to omit should be commented out with `#`.

    Args:
        fn:   path of the datafile from which to extract data

    Returns:
        array of input data.
    """

    if fn.suffix == '.gz':

        x = []
        with gzip.open(fn, mode='rt') as f:
            for line in f:

                if not line.startswith("#"):  # skip comments

                    data = line.split(" ")
                    val = data[0] if len(data) == 1 else data[1]
                    x.append(float(val))

    elif fn.suffix == '.txt' or fn.suffix == '.DAT':
        data = np.genfromtxt(fn, comments='#')
        x = data if data.ndim == 1 else data[:, 1]

    else:
        raise ValueError("Input data should be a `.txt`, `.DAT` or `.txt.gz` "
                         "file.")

    return np.array(x)