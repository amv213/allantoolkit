import logging
import numpy as np
from . import ci
from . import stats
from . import utils
from typing import List, Tuple, NamedTuple, Union

# Spawn module-level logger
logger = logging.getLogger(__name__)

# shorten type hint to save some space
Array = np.ndarray

# define named tuple to hold dev results
DevResult = NamedTuple('DevResult', [('taus', Array),
                                     ('devs', Array),
                                     ('errs', Array),
                                     ('ns', Array)])


def dev(dev_type: str, data: Array, rate: float, data_type: str,
        taus: Union[str, Array], max_af: int) -> DevResult:
    """Dispatches the input data and parameters to the appropriate statistical
    algorithm computing the requested deviation.

    Args:
        dev_type:   type of deviation to be computed, e.g. `adev`.
        data:       array of phase (in units of seconds) or fractional
                    frequency data for which to calculate deviation.
        rate:       sampling rate of the input data, in Hz.
        data_type:  input data type. Either `phase` or `freq`.
        taus:       array of averaging times for which to compute deviation.
                    Can also be one of the keywords: `all`, `octave`, `decade`.
        max_af:     maximum averaging factor for which to compute deviation.
                    Defaults to length of dataset.

    Returns:
        (taus, devs, errs, ns) NamedTuple of results:

        .taus:  array of averaging times for which deviation was computed.
        .devs:  array with deviation computed at each averaging time.
        .errs:  array with estimated error in each computed deviation.
        .ns:    array with number of values used to compute each deviation.
    """

    # Work with phase data, in units of seconds
    x = utils.input_to_phase(data=data, rate=rate, data_type=data_type)

    # Build/Select averaging factors at which to calculate deviations
    taus, afs = utils.tau_generator(data=x, rate=rate, dev_type=dev_type,
                                    taus=taus, maximum_m=max_af)

    # CALC DEV

    # Should be function of this signature: func(x, m, tau) -> var, n
    func = getattr(stats, 'calc_' + dev_type)  # e.g. stats.calc_adev

    # Initialise arrays
    devs, errs = np.zeros(len(afs)), np.zeros(len(afs))
    ns = np.zeros(len(afs), dtype=int)

    # Calculate metrics at each averaging time / factor
    for i, (tau, m) in enumerate(zip(taus, afs)):

        # Calculate variance, and number of samples it is based on
        var, n = func(x=x, m=m, tau=tau)

        # FIXME there is some leakage of too small n or var
        
        # Calculate deviation
        dev = np.sqrt(var)

        # Calculate error
        err = dev / np.sqrt(n)

        devs[i], errs[i], ns[i] = dev, err, n

    # Cleanup datapoints calculated on too few samples
    taus, devs, errs, ns = utils.remove_small_ns(taus, devs, errs, ns)

    return DevResult(taus=taus, devs=devs, errs=errs, ns=ns)


def adev(data: Array, rate: float = 1., data_type: str = "phase",
         taus: Union[str, Array] = None, max_af: int = None) -> DevResult:
    """Allan deviation (ADEV):
    classic - use only if required - relatively poor confidence
    [[SP1065]_ (pg.14-15)].

    The Allan deviation - :math:`\\sigma_y(\\tau)` - is the square root of
    the Allan variance. The Allan variance is the same as  the  ordinary
    variance  for  white  FM  noise, but  has  the  advantage,  for  more
    divergent  noise  types  such  as flicker  noise,  of  converging  to  a
    value  that  is  independent  on  the  number  of  samples.

    In terms of `phase` data, the Allan variance may be calculated as:

    .. math::

        \\sigma^2_y(\\tau) = { 1 \\over 2 (N-2) \\tau^2 }
        \\sum_{i=1}^{N-2} \\left[ x_{i+2} - 2x_{i+1} + x_{i} \\right]^2

    where :math:`x_i` is the :math:`i`th of :math:`N` phase values spaced by
    an averaging time :math:`\\tau`.

    For a time-series of fractional frequency values, the Allan variance is
    defined as:

    .. math::

        \\sigma^{2}_y(\\tau) =  { 1 \\over 2 (M - 1) } \\sum_{i=1}^{M-1}
        \\left[ \\bar{y}_{i+1} - \\bar{y}_i \\right]^2

    where :math:`\\bar{y}_i` is the :math:`i`th of :math:`M` fractional
    frequency values averaged over the averaging time :math:`\\tau`.

    The  confidence  interval  of  an  Allan  deviation estimate is dependent
    on the noise type, but is often estimated as
    :math:`\\pm\\sigma^{2}_y(\\tau) / \\sqrt{N}`.

    Args:
        See documentation for allantoolkit.allantools.dev
    """

    return dev(dev_type='adev', data=data, rate=rate, data_type=data_type,
               taus=taus, max_af=max_af)


def oadev(data: Array, rate: float = 1., data_type: str = "phase",
          taus: Union[str, Array] = None, max_af: int = None) -> DevResult:
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


def tdev(data, rate=1.0, data_type="phase", taus=None):
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
    phase = utils.input_to_phase(data, rate, data_type)
    (taus, md, mde, ns) = mdev(phase, rate=rate, taus=taus)
    td = taus * md / np.sqrt(3.0)
    tde = td / np.sqrt(ns)
    return taus, td, tde, ns


def mdev(data, rate=1.0, data_type="phase", taus=None):
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
    phase = utils.input_to_phase(data, rate, data_type)
    (taus_used, ms) = utils.tau_generator(data=phase, rate=rate,
                                                 dev_type='mdev', taus=taus)
    data, taus = np.array(phase), np.array(taus)

    md = np.zeros_like(taus_used)
    mderr = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    # this is a 'loop-unrolled' algorithm following
    # http://www.leapsecond.com/tools/adev_lib.c
    for idx, m in enumerate(ms):
        m = int(m) # without this we get: VisibleDeprecationWarning:
                   # using a non-integer number instead of an integer
                   # will result in an error in the future
        tau = taus_used[idx]

        # First loop sum
        d0 = phase[0:m]
        d1 = phase[m:2*m]
        d2 = phase[2*m:3*m]
        e = min(len(d0), len(d1), len(d2))
        v = np.sum(d2[:e] - 2* d1[:e] + d0[:e])
        s = v * v

        # Second part of sum
        d3 = phase[3*m:]
        d2 = phase[2*m:]
        d1 = phase[1*m:]
        d0 = phase[0:]

        e = min(len(d0), len(d1), len(d2), len(d3))
        n = e + 1

        v_arr = v + np.cumsum(d3[:e] - 3 * d2[:e] + 3 * d1[:e] - d0[:e])

        s = s + np.sum(v_arr * v_arr)
        s /= 2.0 * m * m * tau * tau * n
        s = np.sqrt(s)

        md[idx] = s
        mderr[idx] = (s / np.sqrt(n))
        ns[idx] = n

    return utils.remove_small_ns(taus_used, md, mderr, ns)


def ohdev(data, rate=1.0, data_type="phase", taus=None):
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
    phase = utils.input_to_phase(data, rate, data_type)
    (taus_used, m) = utils.tau_generator(data=phase, rate=rate,
                                                dev_type='ohdev', taus=taus)
    hdevs = np.zeros_like(taus_used)
    hdeverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    for idx, mj in enumerate(m):
        (hdevs[idx],
         hdeverrs[idx],
         ns[idx]) = stats.calc_hdev(phase, rate, mj, 1)

    return utils.remove_small_ns(taus_used, hdevs, hdeverrs, ns)


def hdev(data, rate=1.0, data_type="phase", taus=None):
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
    phase = utils.input_to_phase(data, rate, data_type)
    (taus_used, m) = utils.tau_generator(data=phase, rate=rate,
                                                dev_type='hdev', taus=taus)
    hdevs = np.zeros_like(taus_used)
    hdeverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    for idx, mj in enumerate(m):
        (hdevs[idx],
         hdeverrs[idx],
         ns[idx]) = stats.calc_hdev(phase, rate, mj, mj)  # stride = mj

    return utils.remove_small_ns(taus_used, hdevs, hdeverrs, ns)


def totdev(data, rate=1.0, data_type="phase", taus=None):
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
    phase = utils.input_to_phase(data, rate, data_type)
    (taus_used, m) = utils.tau_generator(data=phase, rate=rate,
                                                dev_type='totdev', taus=taus)
    N = len(phase)

    # totdev requires a new dataset
    # Begin by adding reflected data before dataset
    x1 = 2.0 * phase[0] * np.ones((N - 2,))
    x1 = x1 - phase[1:-1]
    x1 = x1[::-1]

    # Reflected data at end of dataset
    x2 = 2.0 * phase[-1] * np.ones((N - 2,))
    x2 = x2 - phase[1:-1][::-1]

    # check length of new dataset
    assert len(x1)+len(phase)+len(x2) == 3*N - 4
    # Combine into a single array
    x = np.zeros((3*N - 4))
    x[0:N-2] = x1
    x[N-2:2*(N-2)+2] = phase # original data in the middle
    x[2*(N-2)+2:] = x2

    devs = np.zeros_like(taus_used)
    deverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    mid = len(x1)

    for idx, mj in enumerate(m):
        mj = int(mj)
        d0 = x[mid + 1:]
        d1 = x[mid  + mj + 1:]
        d1n = x[mid - mj + 1:]
        e = min(len(d0), len(d1), len(d1n))

        v_arr = d1n[:e] - 2.0 * d0[:e] + d1[:e]
        dev = np.sum(v_arr[:mid] * v_arr[:mid])

        dev /= float(2 * pow(mj / rate, 2) * (N - 2))
        dev = np.sqrt(dev)
        devs[idx] = dev
        deverrs[idx] = dev / np.sqrt(mid)
        ns[idx] = mid

    return utils.remove_small_ns(taus_used, devs, deverrs, ns)


def ttotdev(data, rate=1.0, data_type="phase", taus=None):
    """ Time Total Deviation

        Modified total variance scaled by tau^2 / 3

        NIST [SP1065]_ eqn (28) page 26.  Note that [SP1065]_ erroneously has tau-cubed here (!).
    """

    (taus, mtotdevs, mde, ns) = mtotdev(data, data_type=data_type,
                                        rate=rate, taus=taus)
    td = taus*mtotdevs / np.sqrt(3.0)
    tde = td / np.sqrt(ns)
    return taus, td, tde, ns


def mtotdev(data, rate=1.0, data_type="phase", taus=None):
    """ PRELIMINARY - REQUIRES FURTHER TESTING.
        Modified Total deviation.
        Better confidence at long averages for modified Allan

        FIXME: bias-correction http://www.wriley.com/CI2.pdf page 6

        The variance is scaled up (divided by this number) based on the
        noise-type identified.
        WPM 0.94
        FPM 0.83
        WFM 0.73
        FFM 0.70
        RWFM 0.69

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
    phase = utils.input_to_phase(data, rate, data_type)
    (taus_used, ms)= utils.tau_generator(data=phase, rate=rate,
                                                 dev_type='mtotdev', taus=taus,
                                                 maximum_m=float(len(phase))/3.0)
    devs = np.zeros_like(taus_used)
    deverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    for idx, mj in enumerate(ms):
        devs[idx], deverrs[idx], ns[idx] = stats.calc_mtotdev(phase, rate, mj)

    return utils.remove_small_ns(taus_used, devs, deverrs, ns)


def htotdev(data, rate=1.0, data_type="phase", taus=None):
    """ PRELIMINARY - REQUIRES FURTHER TESTING.
        Hadamard Total deviation.
        Better confidence at long averages for Hadamard deviation
        
        Computed for N fractional frequency points y_i with sampling
        period tau0, analyzed at tau = m*tau0
        1. remove linear trend by averaging first and last half and divide by interval
        2. extend sequence by uninverted even reflection
        3. compute Hadamard for extended, length 9m, sequence.

        FIXME: bias corrections from http://www.wriley.com/CI2.pdf
        W FM    0.995      alpha= 0
        F FM    0.851      alpha=-1
        RW FM   0.771      alpha=-2
        FW FM   0.717      alpha=-3
        RR FM   0.679      alpha=-4

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
    if data_type == "phase":
        phase = data
        freq = utils.phase2frequency(phase, rate)
    elif data_type == "freq":
        phase = utils.frequency2phase(data, rate)
        freq = data
    else:
        raise Exception("unknown data_type: " + data_type)

    rate = float(rate)
    (taus_used, ms) = utils.tau_generator(data=freq, rate=rate,
                                                dev_type='htotdev',
                                                taus=taus,
                                                maximum_m=float(len(freq))/3.0)
    phase = np.array(phase)
    freq = np.array(freq)
    devs = np.zeros_like(taus_used)
    deverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    # NOTE at mj==1 we use ohdev(), based on comment from here:
    # http://www.wriley.com/paper4ht.htm
    # "For best consistency, the overlapping Hadamard variance is used
    # instead of the Hadamard total variance at m=1"
    # FIXME: this uses both freq and phase datasets, which uses double the memory really needed...
    for idx, mj in enumerate(ms):
        if int(mj) == 1:
            (devs[idx],
             deverrs[idx],
             ns[idx]) = stats.calc_hdev(phase, rate, mj, 1)
        else:
            (devs[idx],
             deverrs[idx],
             ns[idx]) = stats.calc_htotdev(freq, mj)

    return utils.remove_small_ns(taus_used, devs, deverrs, ns)


def theo1(data, rate=1.0, data_type="phase", taus=None):
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
    phase = utils.input_to_phase(data, rate, data_type)

    tau0 = 1.0/rate
    (taus_used, ms) = utils.tau_generator(data=phase, rate=rate,
                                                 dev_type='theo1', taus=taus)

    devs = np.zeros_like(taus_used)
    deverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    N = len(phase)
    for idx, m in enumerate(ms):
        m = int(m) # to avoid: VisibleDeprecationWarning: using a
                   # non-integer number instead of an integer will
                   # result in an error in the future
        assert m % 2 == 0 # m must be even
        dev = 0
        n = 0
        for i in range(int(N-m)):
            s = 0
            for d in range(int(m/2)): # inner sum
                pre = 1.0 / (float(m)/2 - float(d))
                s += pre*pow(phase[i]-phase[i-d+int(m/2)] +
                             phase[i+m]-phase[i+d+int(m/2)], 2)
                n = n+1
            dev += s
        assert n == (N-m)*m/2 # N-m outer sums, m/2 inner sums
        dev = dev/(0.75*(N-m)*pow(m*tau0, 2))
        # factor 0.75 used here? http://tf.nist.gov/general/pdf/1990.pdf
        # but not here? http://tf.nist.gov/timefreq/general/pdf/2220.pdf page 29
        devs[idx] = np.sqrt(dev)
        deverrs[idx] = devs[idx] / np.sqrt(N-m)
        ns[idx] = n

    return utils.remove_small_ns(taus_used, devs, deverrs, ns)


def tierms(data, rate=1.0, data_type="phase", taus=None):
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
    phase = utils.input_to_phase(data, rate, data_type)
    (taus_used, m) = utils.tau_generator(data=phase, rate=rate,
                                               dev_type='tierms', taus=taus)

    count = len(phase)

    devs = np.zeros_like(taus_used)
    deverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    for idx, mj in enumerate(m):
        mj = int(mj)

        # This seems like an unusual way to
        phases = np.column_stack((phase[:-mj], phase[mj:]))
        p_max = np.max(phases, axis=1)
        p_min = np.min(phases, axis=1)
        phases = p_max - p_min
        tie = np.sqrt(np.mean(phases * phases))

        ncount = count - mj

        devs[idx] = tie
        deverrs[idx] = 0 / np.sqrt(ncount) # TODO! I THINK THIS IS WRONG!
        ns[idx] = ncount

    return utils.remove_small_ns(taus_used, devs, deverrs, ns)


def mtie_rolling_window(a, window):
    """
    Make an ndarray with a rolling window of the last dimension, from
    http://mail.scipy.org/pipermail/numpy-discussion/2011-January/054401.html

    Parameters
    ----------
    a : array_like
        Array to add rolling window to
    window : int
        Size of rolling window

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size window.
    
    Note
    ----
    This may consume large amounts of memory. See discussion:
    https://mail.python.org/pipermail/numpy-discussion/2011-January/054364.html
    https://mail.python.org/pipermail/numpy-discussion/2011-January/054370.html

    """
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def mtie(data, rate=1.0, data_type="phase", taus=None):
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
    phase = utils.input_to_phase(data, rate, data_type)
    (taus_used, m) = utils.tau_generator(data=phase, rate=rate,
                                                dev_type='mtie', taus=taus)
    devs = np.zeros_like(taus_used)
    deverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    for idx, mj in enumerate(m):
        try:
            # the older algorithm uses a lot of memory
            # but can be used for short datasets.
            rw = mtie_rolling_window(phase, int(mj + 1))
            win_max = np.max(rw, axis=1)
            win_min = np.min(rw, axis=1)
            tie = win_max - win_min
            dev = np.max(tie)
        except:
            if int(mj + 1) < 1:
                raise ValueError("`window` must be at least 1.")
            if int(mj + 1) > phase.shape[-1]:
                raise ValueError("`window` is too long.")

            mj = int(mj)
            currMax = np.max(phase[0:mj])
            currMin = np.min(phase[0:mj])
            dev = currMax - currMin
            for winStartIdx in range(1, int(phase.shape[0] - mj)):
                winEndIdx = mj + winStartIdx
                if currMax == phase[winStartIdx - 1]:
                    currMax = np.max(phase[winStartIdx:winEndIdx])
                elif currMax < phase[winEndIdx]:
                    currMax = phase[winEndIdx]

                if currMin == phase[winStartIdx - 1]:
                    currMin = np.min(phase[winStartIdx:winEndIdx])
                elif currMin > phase[winEndIdx]:
                    currMin = phase[winEndIdx]

                if dev < currMax - currMin:
                    dev = currMax - currMin

        ncount = phase.shape[0] - mj
        devs[idx] = dev
        deverrs[idx] = dev / np.sqrt(ncount)
        ns[idx] = ncount

    return utils.remove_small_ns(taus_used, devs, deverrs, ns)

#
# !!!!!!!
# FIXME: mtie_phase_fast() is incomplete.
# !!!!!!!
#
def mtie_phase_fast(phase, rate=1.0, data_type="phase", taus=None):
    """ fast binary decomposition algorithm for MTIE

        See: [Bregni2001]_ STEFANO BREGNI "Fast Algorithms for TVAR and MTIE Computation in
        Characterization of Network Synchronization Performance"
    """
    rate = float(rate)
    phase = np.asarray(phase)
    k_max = int(np.floor(np.log2(len(phase))))
    phase = phase[0:pow(2, k_max)] # truncate data to 2**k_max datapoints
    assert len(phase) == pow(2, k_max)
    #k = 1
    taus = [pow(2, k) for k in range(k_max)]
    #while k <= k_max:
    #    tau = pow(2, k)
    #    taus.append(tau)
        #print tau
    #    k += 1
    print("taus N=", len(taus), " ", taus)
    devs = np.zeros(len(taus))
    deverrs = np.zeros(len(taus))
    ns = np.zeros(len(taus))
    taus_used = np.array(taus) # [(1.0/rate)*t for t in taus]
    # matrices to store results
    mtie_max = np.zeros((len(phase)-1, k_max))
    mtie_min = np.zeros((len(phase)-1, k_max))
    for kidx in range(k_max):
        k = kidx+1
        imax = len(phase)-pow(2, k)+1
        #print k, imax
        tie = np.zeros(imax)
        ns[kidx] = imax
        #print np.max( tie )
        for i in range(imax):
            if k == 1:
                mtie_max[i, kidx] = max(phase[i], phase[i+1])
                mtie_min[i, kidx] = min(phase[i], phase[i+1])
            else:
                p = int(pow(2, k-1))
                mtie_max[i, kidx] = max(mtie_max[i, kidx-1],
                                        mtie_max[i+p, kidx-1])
                mtie_min[i, kidx] = min(mtie_min[i, kidx-1],
                                        mtie_min[i+p, kidx-1])

        #for i in range(imax):
            tie[i] = mtie_max[i, kidx] - mtie_min[i, kidx]
            #print tie[i]
        devs[kidx] = np.amax(tie) # maximum along axis
        #print "maximum %2.4f" % devs[kidx]
        #print np.amax( tie )
    #for tau in taus:
    #for
    devs = np.array(devs)
    print("devs N=", len(devs), " ", devs)
    print("taus N=", len(taus_used), " ", taus_used)
    return utils.remove_small_ns(taus_used, devs, deverrs, ns)


########################################################################
#
#  gap resistant Allan deviation
#

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
    return utils.remove_small_ns(taus_used, ad, np.array([ade_l, ade_h]), adn)



