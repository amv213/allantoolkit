import logging
import numpy as np
from typing import List, Tuple, Union

# Spawn module-level logger
logger = logging.getLogger(__name__)

# shorten type hint to save some space
Array = np.ndarray


def frequency2phase(frequency_data: Array, rate: float) -> Array:
    """Integrates fractional frequency data to output phase data.

    Frequency to phase conversion is done by piecewise  integration  using
    the  averaging  time  as  the  integration  interval
    [RileyStable32Manual]_:

    .. math:: x_{i+1} = x_i + y_i \\tau

    Args:
        frequency_data: data array of fractional frequency measurements
        rate:           sampling rate of the input data, in Hz

    Returns:
        time integral of fractional frequency data, i.e. phase (time) data
        in units of seconds. For phase in units of radians, see
        `phase2radians()`.
    """

    sampling_period = 1.0 / rate

    # Protect against NaN values in input array (issue #60)
    # Reintroduces data trimming as in commit 503cb82
    frequency_data = trim_data(frequency_data)

    # Erik Benkler (PTB): Subtract mean value before cumsum in order to
    # avoid precision issues when we have small frequency fluctuations on
    # a large average frequency
    frequency_data = frequency_data - np.nanmean(frequency_data)

    phasedata = np.cumsum(frequency_data) * sampling_period

    # insert arbitrary 0 phase point for x_0
    phasedata = np.insert(phasedata, 0, 0)

    return phasedata


def input_to_phase(data: Array, rate: float, data_type: str) -> Array:
    """Takes either phase or frequency data as input and returns phase.

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


def tau_generator(data, rate, taus=None, v=False, even=False, maximum_m=-1):
    """ pre-processing of the tau-list given by the user (Helper function)

    Does sanity checks, sorts data, removes duplicates and invalid values.
    Generates a tau-list based on keywords 'all', 'decade', 'octave'.
    Uses 'octave' by default if no taus= argument is given.

    Parameters
    ----------
    data: np.array
        data array
    rate: float
        Sample rate of data in Hz. Time interval between measurements
        is 1/rate seconds.
    taus: np.array
        Array of tau values for which to compute measurement.
        Alternatively one of the keywords: "all", "octave", "decade".
        Defaults to "octave" if omitted.
    v:
        verbose output if True
    even:
        require even m, where tau=m*tau0, for Theo1 statistic
    maximum_m:
        limit m, where tau=m*tau0, to this value.
        used by mtotdev() and htotdev() to limit maximum tau.

    Returns
    -------
    (data, m, taus): tuple
        List of computed values
    data: np.array
        Data
    m: np.array
        Tau in units of data points
    taus: np.array
        Cleaned up list of tau values
    """

    if rate == 0:
        raise RuntimeError("Warning! rate==0")

    if taus is None:  # empty or no tau-list supplied
        taus = "octave"  # default to octave
    elif isinstance(taus, list) and taus == []:
        taus = "octave"

    if isinstance(taus, str):

        if taus == "all":
            taus = (1.0/rate)*np.linspace(1.0, len(data), len(data))
        elif taus == "octave":
            maxn = np.floor(np.log2(len(data)))
            taus = (1.0/rate)*np.logspace(0, int(maxn), int(maxn+1), base=2.0)
        elif taus == "decade": # 1, 2, 4, 10, 20, 40, spacing similar to Stable32
            maxn = np.floor(np.log10(len(data)))
            taus = []
            for k in range(int(maxn+1)):
                taus.append(1.0*(1.0/rate)*pow(10.0, k))
                taus.append(2.0*(1.0/rate)*pow(10.0, k))
                taus.append(4.0*(1.0/rate)*pow(10.0, k))

    data, taus = np.array(data), np.array(taus)
    rate = float(rate)
    m = [] # integer averaging factor. tau = m*tau0

    if maximum_m == -1: # if no limit given
        maximum_m = len(data)
    # FIXME: should we use a "stop-ratio" like Stable32
    # found in Table III, page 9 of "Evolution of frequency stability analysis software"
    # max(AF) = len(phase)/stop_ratio, where
    # function  stop_ratio
    # adev      5
    # oadev     4
    # mdev      4
    # tdev      4
    # hdev      5
    # ohdev     4
    # totdev    2
    # tierms    4
    # htotdev   3
    # mtie      2
    # theo1     1
    # theoH     1
    # mtotdev   2
    # ttotdev   2

    taus_valid1 = taus < (1 / float(rate)) * float(len(data))
    taus_valid2 = taus > 0
    taus_valid3 = taus <= (1 / float(rate)) * float(maximum_m)
    taus_valid = taus_valid1 & taus_valid2 & taus_valid3
    m = np.floor(taus[taus_valid] * rate)
    m = m[m != 0]       # m is tau in units of datapoints
    m = np.unique(m)    # remove duplicates and sort

    if v:
        print("tau_generator: ", m)

    if len(m) == 0:
        print("Warning: sanity-check on tau failed!")
        print("   len(data)=", len(data), " rate=", rate, "taus= ", taus)

    taus2 = m / float(rate)

    if even:  # used by Theo1
        m_even_mask = ((m % 2) == 0)
        m = m[m_even_mask]
        taus2 = taus2[m_even_mask]

    return data, m, taus2


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


def trim_data(x):
    """
    Trim leading and trailing NaNs from dataset
    This is done by browsing the array from each end and store the index of the
    first non-NaN in each case, the return the appropriate slice of the array
    """
    # Find indices for first and last valid data
    first = 0
    while np.isnan(x[first]):
        first += 1
    last = len(x)
    while np.isnan(x[last - 1]):
        last -= 1
    return x[first:last]


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

########################################################################
#
# simple conversions between frequency, phase(seconds), phase(radians)
#


def phase2radians(phasedata, v0):
    """ Convert phase in seconds to phase in radians

    Parameters
    ----------
    phasedata: np.array
        Data array of phase in seconds
    v0: float
        Nominal oscillator frequency in Hz

    Returns
    -------
    fi:
        phase data in radians
    """
    fi = [2*np.pi*v0*xx for xx in phasedata]
    return fi


def phase2frequency(phase, rate):
    """ Convert phase in seconds to fractional frequency

    Parameters
    ----------
    phase: np.array
        Data array of phase in seconds
    rate: float
        The sampling rate for phase, in Hz

    Returns
    -------
    y:
        Data array of fractional frequency
    """
    y = rate*np.diff(phase)
    return y


def frequency2fractional(frequency, mean_frequency=-1):
    """ Convert frequency in Hz to fractional frequency

    Parameters
    ----------
    frequency: np.array
        Data array of frequency in Hz
    mean_frequency: float
        (optional) The nominal mean frequency, in Hz
        if omitted, defaults to mean frequency=np.mean(frequency)

    Returns
    -------
    y:
        Data array of fractional frequency
    """
    if mean_frequency == -1:
        mu = np.mean(frequency)
    else:
        mu = mean_frequency
    y = [(x-mu)/mu for x in frequency]
    return y