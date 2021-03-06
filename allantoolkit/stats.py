import logging
import numpy as np
from . import ci
from . import utils
from typing import NamedTuple

# Spawn module-level logger
logger = logging.getLogger(__name__)

# shorten type hint to save some space
Array = np.ndarray


# define named tuples to hold var results
class VarResult(NamedTuple):
    """Container for variance calculated at single averaging factor. Defines
    the following fields:
    """

    var: float
    "variance at given averaging factor."
    n: int
    "number of samples used to estimate the variance."


class VarResults(NamedTuple):
    """Container for variance calculations over a range of averaging factors.
    Defines the following fields:
    """

    vars: Array
    "array of variances at each averaging factor."
    ns: Array
    "array of number of samples used to estimate the variance at each " \
        "averaging factor."


def calc_svar(x: Array, m: int, rate: float) -> VarResult:
    """Calculates standard variance (VAR) of phase data at given averaging
    factor.

    .. seealso::
        Function :func:`allantoolkit.stats.calc_svar_freq` for detailed
        implementation.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.
    """

    # Convert to frequency and then use frequency VAR definition
    y = utils.phase2frequency(x=x, rate=rate)

    return calc_svar_freq(y=y, m=m, rate=rate)


def calc_svar_freq(y: Array, m: int, rate: float = None) -> VarResult:
    """Calculates standard variance (VAR) of fractional frequency data at
    given averaging factor.

    .. hint::

        The standard variance is not convergent for some clock noises,
        and should not be used for the analysis of frequency stability.

    The classic `N`-sample or standard variance can be estimated from a
    set of :math:`M` fractional frequency measurements averaged with averaging
    time :math:`\\tau = m\\tau_0`, where :math:`m` is the averaging factor
    and :math:`\\tau_0` is the basic data sampling period, by the following
    expression:

    .. math::

        \\sigma_y^2 = { 1 \\over M-1} \\sum_{i=1}^{M} \\left( y_i -
        \\bar{y} \\right)^2

    where :math:`\\bar{y} = {1 \\over M} \\sum_{i=1}^M y_i` is the average
    frequency.

    Args:
        y:      input fractional frequency data.
        m:      averaging factor at which to calculate variance.
        rate:   sampling rate of the input data, in Hz. Not used here.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        [RileyStable32]_ (5.2.1. Standard Variance, pg.17-8)
    """

    # average at given averaging factor
    y = utils.decimate(data=y, m=m, data_type='freq')

    ybar = np.nanmean(y)
    summand = (y-ybar)**2

    n = summand[~np.isnan(summand)].size

    if n < 2:
        logger.warning("Not enough fractional frequency measurements to "
                       "compute variance at averaging factor %i: %s", m, y)
        var = np.NaN
        return VarResult(var=var, n=0)

    var = np.nansum(summand) / (n - 1)

    return VarResult(var=var, n=n)


def calc_o_avar(x: Array, m: int, rate: float, stride: int) -> VarResult:
    """Calculates Allan variance (AVAR) or overlapping Allan variance (
    OAVAR) of phase data at given averaging factor. The variance type is set
    by the ``stride`` parameter.

    .. seealso::
        Functions :func:`allantoolkit.stats.calc_avar`, and
        :func:`allantoolkit.stats.calc_oavar` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.
        stride: stride at which to parse input phase data.
                ``m`` for AVAR, ``1`` for OAVAR.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
       www.leapsecond.com/tools/adev_lib.c
    """

    N = x.size

    d = 2  # second difference algorithm

    if N < d*m + 1:
        logger.warning("Not enough phase measurements to compute "
                       "variance at averaging factor %i (N=%i)", m, N)
        return VarResult(var=np.NaN, n=0)

    # Calculate second differences
    summand = x[2*m::stride] - 2*x[m:-m:stride] + x[:-2*m:stride]
    n = summand[~np.isnan(summand)].size  # (N-2*m)/stride if no NaNs

    if n == 0:
        logger.warning("Not enough valid phase measurements to compute "
                       "variance at averaging factor %i: %s", m, x)
        return VarResult(var=np.NaN, n=0)

    # Calculate variance
    var = 1. / (2 * (m/rate)**2) * np.nanmean(summand**2)

    return VarResult(var=var, n=n)


def calc_avar(x: Array, m: int, rate: float) -> VarResult:
    """Calculates Allan variance (AVAR) of phase data at given averaging
    factor.

    The Allan variance is calculated as:

    .. math::

        \\sigma^2_y(\\tau) = { 1 \\over 2 (N-2) \\tau^2 }
        \\sum_{i=1}^{N-2} \\left[ x_{i+2} - 2x_{i+1} + x_{i} \\right]^2

    where :math:`x_i` is the :math:`i^{th}` of :math:`N` phase
    values spaced by an averaging time :math:`\\tau`.

    .. seealso::
        Function :func:`allantoolkit.devs.adev` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        TODO: add exact reference
    """

    return calc_o_avar(x=x, m=m, rate=rate, stride=m)


def calc_oavar(x: Array, m: int, rate: float) -> VarResult:
    """Calculates overlapping Allan variance (OAVAR) of phase data at given
    averaging factor.

    The overlapping Allan variance is estimated from a set of :math:`N`
    phase measurements for averaging time :math:`\\tau = m\\tau_0`, where
    :math:`m` is the averaging factor and :math:`\\tau_0` is the basic
    data sampling period, by the following expression:

    .. math::

        \\sigma^{2}_y(\\tau) = { 1 \\over 2 \\tau^2 (N-2m) }
        \\sum_{i=1}^{N-2m} \\left[ {x}_{i+2m} - 2x_{i+m} + x_{i} \\right]^2

    .. seealso::
        Function :func:`allantoolkit.devs.oadev` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        TODO: add exact reference
    """

    return calc_o_avar(x=x, m=m, rate=rate, stride=1)


def calc_avar_freq(y: Array, m: int, rate: float) -> VarResult:
    """Calculates Allan variance (AVAR) of fractional frequency data at given
    averaging factor.

    The Allan variance is calculated as:

    .. math::

        \\sigma^{2}_y(\\tau) =  { 1 \\over 2 (M - 1) } \\sum_{i=1}^{M-1}
        \\left[ \\bar{y}_{i+1} - \\bar{y}_i \\right]^2

    where :math:`\\bar{y}_i` is the :math:`i^{th}` of :math:`M`
    fractional frequency values averaged over the averaging time
    :math:`\\tau`.

    .. seealso::
        Function :func:`allantoolkit.devs.adev` for background details.

    Args:
        y:      input phase fractional frequency data.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz. Not used here.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        TODO: add exact reference
    """

    # average at given averaging factor
    y = utils.decimate(data=y, m=m, data_type='freq')

    M = y[~np.isnan(y)].size

    if M < 2:
        logger.warning("Not enough fractional frequency measurements to "
                       "compute variance at averaging factor %i: %s", m, y)
        return VarResult(var=np.NaN, n=0)

    # Calculate first difference
    summand = np.diff(y)
    m = summand[~np.isnan(summand)].size  # M - 1 if no NaNs

    if m == 0:
        logger.warning("Not enough valid phase measurements to compute "
                       "variance at averaging factor %i: %s", m, y)
        return VarResult(var=np.NaN, n=0)

    # Calculate variance
    var = (1. / 2) * np.nanmean(summand**2)

    return VarResult(var=var, n=m)


def calc_mvar(x: Array, m: int, rate: float) -> VarResult:
    """Calculates modified Allan variance (MVAR) of phase data at given
    averaging factor.

    The modified Allan variance is estimated from a set of
    :math:`N` phase measurements for averaging time :math:`\\tau =
    m\\tau_0`, where :math:`m` is the averaging factor and :math:`\\tau_0`
    is the basic data sampling period, by the following expression:

    .. math::

        \\sigma^{2}_y(\\tau) = { 1 \\over 2 m^2 \\tau^2 (N-3m+1) }
        \\sum_{j=1}^{N-3m+1} \\left\\{
        \\sum_{i=j}^{j+m-1} \\left[ {x}_{i+2m} - 2x_{i+m} + x_{i} \\right]
        \\right\\}^2

    .. note::
        The algorithm used in practice in this code, is a `loop-unrolled`
        algorithm - as per http://www.leapsecond.com/tools/adev_lib.c

    .. seealso::
        Function :func:`allantoolkit.devs.mdev` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        [Bregni2001]_
        S. Bregni and S. Maccabruni, "Fast Computation of Maximum Time
        Interval Error by Binary Decomposition", IEEE Trans. I&M, Vol. 49,
        No. 6, Dec. 2000, pp. 1240-1244.
    """

    # TODO: make gap resistant and return correct number of non-NaN samples

    # First loop sum: i=j -> + m-1
    d0 = x[0:m]
    d1 = x[m:2 * m]
    d2 = x[2 * m:3 * m]
    e = min(len(d0), len(d1), len(d2))

    v = np.nansum(d2[:e] - 2 * d1[:e] + d0[:e])
    s = v ** 2

    # Second part of sum
    d3 = x[3 * m:]
    d2 = x[2 * m:]
    d1 = x[1 * m:]
    d0 = x[0:]

    e = min(len(d0), len(d1), len(d2), len(d3))
    n = e + 1

    v_arr = v + np.nancumsum(d3[:e] - 3 * d2[:e] + 3 * d1[:e] - d0[:e])

    s = s + np.nansum(v_arr * v_arr)
    s /= 2 * m**2 * (m/rate)**2 * n

    return VarResult(var=s, n=n)


def calc_tvar(x: Array, m: int, rate: float) -> VarResult:
    """Calculates time Allan variance (TVAR) of phase data at given
    averaging factor.

    The Time Allan variance is calculated as:

    .. math::

        \\sigma^2_x( \\tau ) = { \\tau^2 \\over 3 } {\\textrm{MVAR}(\\tau)}

    where :math:`\\textrm{MVAR}(\\tau)` is the modified Allan variance of the
    data at averaging time :math:`\\tau`.

    Note that the Time Allan variance has units of seconds, and not fractional
    frequency.

    .. seealso::
        Function :func:`allantoolkit.devs.tdev` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        [Bregni2001]_
        S. Bregni and S. Maccabruni, "Fast Computation of Maximum Time
        Interval Error by Binary Decomposition", IEEE Trans. I&M, Vol. 49,
        No. 6, Dec. 2000, pp. 1240-1244.
    """

    mvar, n = calc_mvar(x=x, m=m, rate=rate)

    var = ((m/rate)**2 / 3) * mvar

    return VarResult(var=var, n=n)


def calc_o_hvar(x: Array, m: int, rate: float, stride: int) -> VarResult:
    """Calculates Hadamard variance (HVAR) or overlapping Hadamard variance (
    OHVAR) of phase data at given averaging factor. The variance type is set
    by the ``stride`` parameter.

    .. seealso::
        Functions :func:`allantoolkit.stats.calc_hvar`, and
        :func:`allantoolkit.stats.calc_ohvar` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.
        stride: stride at which to parse input phase data.
                ``m`` for HVAR, ``1`` for OHVAR.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        TODO: find exact references
        http://www.leapsecond.com/tools/adev_lib.c
    """

    N = x.size

    d = 3  # third difference algorithm

    if N < d*m + 1:
        logger.warning("Not enough phase measurements to compute "
                       "variance at averaging factor %i (N=%i)", m, N//stride)
        return VarResult(var=np.NaN, n=0)

    # Calculate third differences
    summand = x[3*m::stride] - 3*x[2*m:-m:stride] + 3*x[m:-2*m:stride] - x[:-3*m:stride]
    n = summand[~np.isnan(summand)].size  # (N-3*m)/stride if no NaNs

    if n == 0:
        logger.warning("Not enough valid phase measurements to compute "
                       "variance at averaging factor %i (N=%i)", m, N)
        return VarResult(var=np.NaN, n=0)

    # Calculate variance
    var = 1. / (6 * (m/rate)**2) * np.nanmean(summand**2)

    return VarResult(var=var, n=n)


def calc_hvar(x: Array, m: int, rate: float) -> VarResult:
    """Calculates the Hadamard variance (HVAR) of phase data at given
    averaging factor.

    The Hadamard variance is calculated as:

    .. math::

        \\sigma^2_y(\\tau) = { 1 \\over 6 \\tau^2 (N-3)}
        \\sum_{i=1}^{N-3} \\left[
        x_{i+3} - 3x_{i+2} + 3x_{i+1} - x_i
        \\right]^2

    where :math:`x_i` is the :math:`i^{th}` of :math:`N` phase
    values spaced by an averaging time :math:`\\tau`.

    .. seealso::
        Function :func:`allantoolkit.devs.hdev` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        TODO: find exact references
    """

    return calc_o_hvar(x=x, m=m, rate=rate, stride=m)


def calc_ohvar(x: Array, m: int, rate: float) -> VarResult:
    """Calculates the overlapping Hadamard variance (OHVAR) of phase data at
    given averaging factor.

    The overlapping Hadamard variance is calculated from a set of :math:`N`
    phase measurements for averaging time :math:`\\tau = m\\tau_0`, where
    :math:`m` is the averaging factor and :math:`\\tau_0` is the basic
    data sampling period, by the following expression:

    .. math::

        \\sigma^{2}_y(\\tau) = { 1 \\over 6 \\tau^2 (N-3m) }
        \\sum_{i=1}^{N-3m} \\left[
        x_{i+3m} - 3x_{i+2m} + 3x_{i+m} - x_{i}
        \\right]^2

    .. seealso::
        Function :func:`allantoolkit.devs.ohdev` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        TODO: find exact references
    """

    return calc_o_hvar(x=x, m=m, rate=rate, stride=1)


def calc_totvar(x: Array, m: int, rate: float) -> VarResult:
    """Calculates the total variance (TOTVAR) of phase data at
    given averaging factor.

    .. caution::
        This algorithm is slow. Try to limit the input data size.

    The total variance is calculated from a set of :math:`N` phase
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

    .. seealso::
        Function :func:`allantoolkit.devs.totdev` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        TODO: find exact references
    """

    N = x.size

    # Need more than 3 values to build an extended dataset with at least 3
    # samples on which to calculate algorithm
    if N < 3:
        logger.warning("Not enough phase measurements to compute "
                       "variance at averaging factor %i: %s", m, x)
        return VarResult(var=np.NaN, n=0)

    # Build extended virtual dataset as required by totvar, of size 3N - 4
    xxx = np.pad(x, N-1, 'symmetric', reflect_type='odd')
    xxx = np.delete(xxx, [N-2, -N+1])  # pop duplicate edge values

    # index at which the original dataset starts
    i0 = N - 2  # `i` = 1
    # index at which summation starts
    i = i0 + 1  # `i` = 2

    # Calculate differences
    summand = (xxx[i-m:][:N-2] - 2*xxx[i:][:N-2] + xxx[i+m:][:N-2])
    n = summand[~np.isnan(summand)].size  # (N-2) if no NaNs

    if n == 0:
        logger.warning("Not enough valid phase measurements to compute "
                       "variance at averaging factor %i: %s", m, x)
        return VarResult(var=np.NaN, n=0)

    # Calculate variance
    var = 1. / (2 * (m/rate)**2) * np.nanmean(summand**2)

    return VarResult(var=var, n=n)


def calc_mtotvar(x: Array, m: int, rate: float) -> VarResult:
    """Calculates the modified total variance (MTOTVAR) of phase data at
    given averaging factor.

    .. caution::
        This algorithm is slow. Try to limit the input data size.

    The modified total variance is calculated from a set of :math:`N` phase
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

    .. seealso::
        Function :func:`allantoolkit.devs.mtotdev` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        [Howe1999]_
        D.A. Howe and F. Vernotte, "Generalization of the Total Variance
        Approach to the Modified Allan Variance," Proc.
        31 st PTTI Meeting, pp. 267-276, Dec. 1999.

        TODO: find justification for last part of algorithm
    """

    # TODO: make gap resistant and return correct number of non-NaN samples

    # Start with N phase data points to be analysed at averaging time tau
    N = x.size

    # MTOT is computed from a set of N-3m+1 subsequences of 3*m points
    subseqs = [x[i:i+3*m] for i in range(N-3*m+1)]

    n = 0  # number of terms in the sum, for error estimation
    var = 0.0  # the variance we are computing
    for xi in subseqs:

        # First, a linear trend (frequency offset) is removed from the
        # subsequence by averaging the first and last halves of the subsequence
        # and dividing by half the interval

        # calc mean of halves
        w = xi.size // 2  # width of a `half`
        mu1, mu2 = np.nanmean(xi[:w]),  np.nanmean(xi[-w:])  # mean of `halves`

        # calc slope
        slope = (mu2 - mu1) / (w + (xi.size % 2))  # extra sep if odd n samples

        # remove the linear trend
        xi0 = xi - slope * np.arange(xi.size)

        # Then the offset-removed subsequence is extended at both ends by
        # uninverted, even reflection: 3m -> 9m points.
        xstar = np.pad(xi0, xi0.size, 'symmetric', reflect_type='even')
        assert xstar.size == 9*m

        # ---

        # Next the modified Allan variance is computed for these 9m points
        # now compute mdev on these 9m points
        # 6m unique groups of m-point averages,
        # use all possible overlapping second differences
        # one term in the 6m sum:  [ x_i - 2 x_i+m + x_i+2m ]^2
        squaresum = 0.0
        # print('m=%d 9m=%d maxj+3*m=%d' %( m, len(xstar), 6*int(m)+3*int(m)) )

        # below we want the following sums (averages, see squaresum where we divide by m)
        # xmean1=np.sum(xstar[j     :   j+m])
        # xmean2=np.sum(xstar[j+m   : j+2*m])
        # xmean3=np.sum(xstar[j+2*m : j+3*m])
        # for speed these are not computed with np.sum or np.mean in each loop
        # instead they are initialized at m=0, and then just updated
        for j in range(0, 6 * m):  # summation of the 6m terms.
            # faster inner sum, based on Stable32 MTC.c code
            if j == 0:
                # intialize the sum
                xmean1 = np.sum(xstar[0:m])
                xmean2 = np.sum(xstar[m:2 * m])
                xmean3 = np.sum(xstar[2 * m:3 * m])
            else:
                # j>=1, subtract old point, add new point
                xmean1 = xmean1 - xstar[j - 1] + xstar[j + m - 1]  #
                xmean2 = xmean2 - xstar[m + j - 1] + xstar[j + 2 * m - 1]  #
                xmean3 = xmean3 - xstar[2 * m + j - 1] + xstar[
                    j + 3 * m - 1]  #

            squaresum += pow((xmean1 - 2.0 * xmean2 + xmean3) / float(m), 2)

        var += squaresum
        n = n + 1

    var = 1./(6*m) * var

    # scaling in front of double sum
    assert n == N - 3 * m + 1  # sanity check on the number of terms n
    var = var * 1.0 / (2.0 * (m/rate)**2 * (N - 3 * m + 1))

    return VarResult(var=var, n=n)


def calc_ttotvar(x: Array, m: int, rate: float) -> VarResult:
    """Calculates the time total variance (TTOTVAR) of phase data at
    given averaging factor.

    .. caution::
        This algorithm is slow. Try to limit the input data size.

    The time total variance is calculated as:

    .. math::

        \\sigma^2_x( \\tau ) = { \\tau^2 \\over 3 } {\\textrm{MTOTVAR}(\\tau)}

    where :math:`\\textrm{MTOTVAR}(\\tau)` is the modified total variance of
    the data at averaging time :math:`\\tau`.

    Note that the time total variance has units of seconds, and not
    fractional frequency.

    .. seealso::
        Function :func:`allantoolkit.devs.ttotdev` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        TODO: find exact reference
    """

    mtotvar, n = calc_mtotvar(x=x, m=m, rate=rate)

    var = ((m/rate)**2 / 3) * mtotvar

    return VarResult(var=var, n=n)


def calc_htotvar(x: Array, m: int, rate: float) -> VarResult:
    """Calculates the Hadamard total variance (HTOTVAR) of phase data at
    given averaging factor. Phase data is pre-converted to frequency data
    before being processed by the algorithm.

    The Hadamard total variance is calculated from a set of :math:`M`
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

    TODO: switch to a phase only algorithm

    .. seealso::
        Function :func:`allantoolkit.devs.htotdev` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        TODO: find exact reference
        http://www.wriley.com/paper4ht.htm
    """

    # TODO: make gap resistant and return correct number of non-NaN samples

    # "For best consistency, the overlapping Hadamard variance is used
    # instead of the Hadamard total variance at m=1"
    # FIXME: this uses both freq and phase datasets, which uses double the
    #  memory really needed...
    if m == 1:

        out = calc_hvar(x=x, m=m, rate=rate)
        return out

    else:

        # This is the frequency version of the algorithm, so use frequency data
        y = utils.phase2frequency(x=x, rate=rate)

        N = y.size  # frequency data, N points

        n = 0  # number of terms in the sum, for error estimation
        var = 0.0  # the deviation we are computing
        for i in range(0, N - 3 * m + 1):

            # subsequence of length 3m, from the original phase data
            xs = y[i:i + 3 * m]

            assert len(xs) == 3 * m
            # remove linear trend. by averaging first/last half,
            # computing slope, and subtracting
            half1_idx = int(np.floor(3 * m / 2.0))
            half2_idx = int(np.ceil(3 * m / 2.0))
            # m
            # 1    0:1   2:2
            mean1 = np.mean(xs[:half1_idx])
            mean2 = np.mean(xs[half2_idx:])

            if int(3 * m) % 2 == 1:  # m is odd
                # 3m = 2k+1 is odd, with the averages at both ends over k points
                # the distance between the averages is then k+1 = (3m-1)/2 +1
                slope = (mean2 - mean1) / ((0.5 * (3 * m - 1) + 1))
            else:  # m is even
                # 3m = 2k is even, so distance between averages is k=3m/2
                slope = (mean2 - mean1) / (0.5 * 3 * m)

            # remove the linear trend
            x0 = [x - slope * (idx - np.floor(3 * m / 2)) for (idx, x) in
                  enumerate(xs)]
            x0_flip = x0[::-1]  # left-right flipped version of array

            # extended sequence, to length 9m, by uninverted even reflection
            xstar = np.concatenate((x0_flip, x0, x0_flip))
            assert len(xstar) == 9 * m

            # now compute totdev on these 9m points
            # 6m unique groups of m-point averages,
            # all possible overlapping second differences
            # one term in the 6m sum:  [ x_i - 2 x_i+m + x_i+2m ]^2
            squaresum = 0.0
            k = 0
            for j in range(0, 6 * int(m)):  # summation of the 6m terms.

                # old naive code
                # xmean1 = np.mean(xstar[j+0*m : j+1*m])
                # xmean2 = np.mean(xstar[j+1*m : j+2*m])
                # xmean3 = np.mean(xstar[j+2*m : j+3*m])
                # squaresum += pow(xmean1 - 2.0*xmean2 + xmean3, 2)

                # new faster way of doing the sums
                if j == 0:

                    # intialize the sum
                    xmean1 = np.sum(xstar[0:m])
                    xmean2 = np.sum(xstar[m:2 * m])
                    xmean3 = np.sum(xstar[2 * m:3 * m])
                else:
                    # j>=1, subtract old point, add new point
                    xmean1 = xmean1 - xstar[j - 1] + xstar[j + m - 1]  #
                    xmean2 = xmean2 - xstar[m + j - 1] + xstar[j + 2 * m - 1]  #
                    xmean3 = xmean3 - xstar[2 * m + j - 1] + xstar[
                        j + 3 * m - 1]  #

                squaresum += pow((xmean1 - 2.0 * xmean2 + xmean3) / float(m), 2)

                k = k + 1

            assert k == 6 * m  # check number of terms in the sum
            squaresum = (1.0 / (6.0 * k)) * squaresum
            var += squaresum
            n = n + 1

        # scaling in front of double-sum
        assert n == N - 3 * m + 1  # sanity check on the number of terms n
        var = var * 1.0 / (N - 3 * m + 1)

        return VarResult(var=var, n=n)


def calc_theo1_slow(x: Array, m: int, rate: float) -> VarResult:
    """Calculates the Thêo1 variance (THEO1VAR) of phase data at
    given averaging factor. The variance is calculated using a slow but
    straightforward algorithm.

    The Thêo1 variance is calculated from a set of :math:`N`
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

    .. seealso::
        Function :func:`allantoolkit.devs.theo1` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it. The averaging time for which Theo1 variances
        apply is :math:`\\tau^*=0.75m\\tau_0`.

    References:
        Theo1: characterization of very long-term frequency stability
        Howe,D.A. et al.
        18th European Frequency and Time Forum (EFTF 2004)
        2004
    """

    if m % 2 != 0 or m < 10:  # m must be even and >= 10
        logger.warning("Theo1 statistic is not compatible with an "
                       "averaging factor m=%i", m)
        return VarResult(var=np.NaN, n=0)

    else:

        N = x.size

        m2 = m // 2

        vals = np.zeros((N-m, m2))  # i x delta array of values

        for i in range(vals.shape[0]):
            for d in range(vals.shape[1]):

                vals[i, d] = 1. / (m2 - d) * (
                    x[i] - x[i - d + m2] + x[i + m] - x[i + d + m2]
                )**2

        # Total number of samples, (N-m)*m2 if no NaNs
        n = vals[~np.isnan(vals)].size

        inner_sum = np.nansum(vals, axis=1)
        outer_mean = np.nanmean(inner_sum)

        var = outer_mean / (0.75*(m/rate)**2)

        # Thus Theo1(m) will be giving a variance value @ tau = 0.75*m*tau_0
        return VarResult(var=var, n=n)


def calc_theo1_fast(x: Array, rate: float, explode: bool = True) -> \
        VarResults:
    """Calculates the Thêo1 variance (THEO1VAR) of phase data over a range
    of averaging factors. This batch variance calculation is implemented via a
    fast Lewis algorithm.

    .. seealso::
        Function :func:`allantoolkit.devs.theo1` for background details.

    Args:
        x:                  input phase data, in units of seconds.
        rate:               data sampling rate, in Hz.
        explode (optional): if ``True`` returns exploded arrays for intuitive
                            indexing; with Theo1(m) at index ``m`` of the
                            array. Defaults to ``False``.

    Returns:
        :class:`allantoolkit.stats.VarResults` NamedTuple of computed
        variances at all allowed (even) averaging factors and corresponding
        number of analysis points. If ``exploded``, the THEO1
        variance at averaging factor ``m`` will be at index ``m`` of the array.
        The averaging time for which Theo1 variances apply is
        :math:`\\tau^*=0.75m\\tau_0`.


    References:
        B. Lewis, "Fast Algorithm for Calculation of  Theo1" Submitted to
        IEEE Transactions on Ultrasonics Ferroelectrics,  and  Frequency
        Control, May 2020. This paper presents a fast, recursive algorithm
        for “all tau” Thêo1 calculations

       http://www.wriley.com/Fast%20Bias-Removed%20Theo1%20Calculation%20with%20R.pdf
    """

    N = x.size

    # Remove any linear offset in x
    midpoint = (N - 1) / 2
    sum1 = 0.
    sum2 = 0.
    for i in range(N):
        sum1 += x[i]
        sum2 += x[i] * (i-midpoint)

    a = sum1/N
    b = sum2/N*12/(N*N-1)
    for i in range(N):
        x[i] -= a + b*(i - midpoint)

    # Calculate C1
    c1 = np.zeros(N)
    s = 0.
    for i in range(N):
        s += x[i]**2
        c1[i] = s

    # Main loop
    k_max = int(np.floor((N-1)/2))  # max `theo1` averaging factors (m = 2k)
    ks = np.zeros(k_max).astype(int)
    theo1s, times = np.zeros(k_max), np.zeros(k_max)
    c3 = np.zeros(k_max + 1)
    c4 = np.zeros(2*k_max)

    c3[0] = c1[N-1]
    for k in range(1, k_max+1):  # calc Theo1 at each averaging factor

        # Calculate the required value of c2
        c2_2k = 0
        c2_2k_1 = 0
        for j in range(N-2*k):
            c2_2k += x[j]*x[j+2*k]
            c2_2k_1 += x[j]*x[j+2*k-1]
        c2_2k_1 += x[N-2*k]*x[N-1]

        # Update C3, C4 in place
        for v in range(k):
            c3[v] -= x[k-1-v]*x[k-1+v] + x[N-k+v]*x[N-k-v]
        for v in range(1, 2*k-1):
            c4[v-1] -= x[2*k-1-v]*x[2*k-1] + x[2*k-2-v]*x[2*k-2] + \
                       x[N-2*k]*x[N-2*k+v] + x[N-2*k+1]*x[N-2*k+1+v]
        c3[k] = c2_2k
        c4[2*k-2] = 2*c2_2k_1 - x[0]*x[2*k-1] - x[N-2*k]*x[N-1]
        c4[2*k-1] = 2*c2_2k

        # Calculate un-normalised T_k from C1-C4
        t_k = 0.
        a0 = c1[N-1] - c1[2*k-1] + c1[N-2*k-1] + 2*c2_2k
        for v in range(1, k+1):
            a1 = a0 - c1[v-1] + c1[N-1-v] - c1[2*k-v-1] + c1[N-1-2*k+v]
            a2 = c3[k-v] - c4[v-1] - c4[2*k-v-1]
            t_k += (a1+2*a2)/v
        assert t_k >= 0

        # Apply normalisation to get Theo1 (Lewis' definition)
        theo1s[k-1] = t_k / (3*(N-2*k)*(k/rate)**2)

        # Log corresponding tau
        times[k-1] = 1.5*k/rate

        # Log corresponding averaging factor
        ks[k-1] = k

        # END OF MAIN LOOP

    # This gives a value Theo1(k) @ tau = 1.5*k*tau_0;
    # as per the definition of Theo1 in Lewis' paper

    # The `Stable32` definition of Theo1 is instead as a function of m,
    # with Theo1(m) giving you Theo1 @ tau = 0.75*m*tau_0

    # so Stable32 Theo1(m, tau=0.75*m*tau0) = Lewis' Theo1(k) with k=m/2

    vars = theo1s
    ns = (N-2*ks)*ks  # TODO: calc this dynamically instead of hard-coding it

    # Remap to an extended array for easy indexing
    if explode:

        xvars = np.full(2*ks.size+2, np.NaN)
        xvars[2::2] = vars  # np.NaN, np.NaN, theo(k=1), np.NaN, theo1(k=2), ..
        vars = xvars
        # Now will have Theo1(m) by indexing vars[m]

        xns = np.full(2 * ks.size + 2, np.NaN)
        xns[2::2] = ns
        ns = xns

    return VarResults(vars=vars, ns=ns.astype(int))


def calc_theo1(x: Array, m: int, rate: float) -> VarResult:
    """Calculates the Thêo1 variance (THEO1VAR) of phase data at
    given averaging factor. The variance is calculated using a fast Lewis
    algorithm.

    .. caution::
        This is only a utility function to match the signatures of the other
        variance functions. Calculation of the variance is still not efficient
        as the `fast` algorithm requires calculating nonetheless an all-tau
        array of THEO1 variances every-time it is called.

    .. seealso::
        Function :func:`allantoolkit.stats.calc_theo1_fast` for the
        preferred implementation.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it. The averaging time for which Theo1 variances
        apply is :math:`\\tau^*=0.75m\\tau_0`.
    """

    vars, ns = calc_theo1_fast(x=x, rate=rate, explode=True)

    return VarResult(var=vars[m], n=ns[m])


def calc_mtie(x: Array, m: int, rate: float = None) -> VarResult:
    """Calculates the maximum time interval variance (MTIEVAR) of phase data at
    given averaging factor.

    MTIE is calculated by moving a `m`-point window (`m` being the
    averaging time of interest) through phase (time-error) data and finding
    the difference between the maximum and minimum values at each window
    position. MTIEVAR is the square of the overall maximum of this time
    interval error over the entire data set.

    .. seealso::
        Function :func:`allantoolkit.devs.mtie` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        TODO: find exact reference
    """

    # Move an n-point window through the phase data, and find the difference
    # between the max nad min values at each windows position
    n = m + 1  # n is defined as m+1

    if m < 1 or m >= x.size:
        logger.warning("Cannot calculate MTIE over this time interval: %f",
                       n/rate)
        return VarResult(var=np.NaN, n=0)

    grads = utils.rolling_grad(x, n)
    assert grads.size == x.size - (n-1)

    # MTIE is the overall maximum of this time interval error
    mtie = np.max(grads)

    # Need to return variance to match function signature
    var = mtie**2

    # The number of analysis points is the whole dataset
    return VarResult(var=var, n=x.size)


def calc_mtie_fast(x: Array, afs: Array, rate: float = None) -> VarResults:
    """Calculates the maximum time interval variance (MTIEVAR) of phase data
    at given averaging factor - using a fast binary decomposition algorithm.

    .. seealso::
        Function :func:`allantoolkit.devs.mtie` for background details.

    Args:
        x:      input phase data, in units of seconds.
        afs:    averaging factors at which to calculate variance. Should be
                such that m+1 = 2^k for all averaging factors in the sequence.
        rate:   sampling rate of the input data, in Hz. Not used here.

    Returns:
        :class:`allantoolkit.stats.VarResults` NamedTuple of
        computed variances at given averaging times, and number of samples
        used to estimate it.

    References:
        [Bregni2001]_
        S. Bregni and S. Maccabruni, "Fast Computation of Maximum Time
        Interval Error by Binary Decomposition", IEEE Trans. I&M, Vol. 49,
        No. 6, Dec. 2000, pp. 1240-1244.
    """

    if not np.all(utils.is_power(z=afs+1)):
        raise ValueError("Averaging factors provided do not support MTIE "
                         "fast binary decomposition algorithm. Averaging "
                         "factors should be such that m+1 = 2^k for all "
                         "averaging factors m.")

    N = x.size

    # Max MTIE windows will be made of of 2^k_max samples
    k_max = np.log2(N).astype(int)

    # k is an integer from 1 to log2(N).
    ks = np.arange(1, k_max+1)

    # matrices to store max and min MTIE results
    A_max = np.full(shape=(k_max, N-1), fill_value=np.NaN)
    A_min = A_max.copy()

    for k in ks:

        kk = k-1  # row index

        for i in range(N - 2**k + 1):

            if k == 1:  # Eq. (15)
                A_max[kk, i] = max(x[i], x[i+1])
                A_min[kk, i] = min(x[i], x[i+1])

            else:  # Eq. (16)
                p = 2**(k - 1)
                A_max[kk, i] = max(A_max[kk-1, i], A_max[kk-1, i+p])
                A_min[kk, i] = min(A_min[kk-1, i], A_min[kk-1, i+p])

    # Eq. (17)
    A = A_max - A_min
    mtie_k = np.nanmax(A, axis=1)

    # Calculate number of samples each mtie was calculated on
    ns = np.array([N - 2**k + 1 for k in ks])

    # Return mtie for requested averaging factor
    # The mtie_k array has mtie(k) at mtie[k-1]
    # --> so mtie(m) at mtie[log2(m+1)-1]
    mtie = mtie_k[np.log2(afs+1).astype(int) - 1]
    ns = ns[np.log2(afs+1).astype(int) - 1]

    return VarResults(vars=mtie**2, ns=ns)


def calc_tierms(x: Array, m: int, rate: float = None) -> VarResult:
    """Calculates rms time interval variance (TIE rms VAR) of phase
    data at given averaging factor.

    The rms time interval variance can be estimated from a set of
    :math:`N` phase measurements for averaging time :math:`\\tau =
    m\\tau_0`, where :math:`m` is the averaging factor and :math:`\\tau_0`
    is the basic data sampling period, by the following expression:

    .. math::

        \\sigma^2_y(\\tau) =
        {1 \\over N-m} \\sum_{i=1}^{N-m} \\left[ x_{i+m} - x_i \\right]^2

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    .. seealso::
        Function :func:`allantoolkit.devs.tierms` for background details.

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        rate:   sampling rate of the input data, in Hz.

    Returns:
        :class:`allantoolkit.stats.VarResult` NamedTuple of
        computed variance at given averaging time, and number of samples
        used to estimate it.

    References:
        TODO: find exact reference
    """

    summand = x[m:] - x[:-m]
    n = summand[~np.isnan(summand)].size  # x.size - m if no NaNs

    var = 1*np.nanmean(summand**2)

    return VarResult(var=var, n=n)