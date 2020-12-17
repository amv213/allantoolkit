import logging
import numpy as np
from . import ci
from . import utils
from typing import NamedTuple


# Spawn module-level logger
logger = logging.getLogger(__name__)

# shorten type hint to save some space
Array = np.ndarray

# define named tuple to hold var results
VarResult = NamedTuple('VarResult', [('var', float), ('n', int)])


def calc_o_avar(x: Array, m: int, tau: float, stride: int) -> VarResult:
    """Main algorithm for AVAR and OAVAR calculation.

    References:
        [RileyStable32]_ (5.2.2, pg.19)
        [Wikipedia]_
        http://www.leapsecond.com/tools/adev_lib.c

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time
        stride: stride at which to parse input phase data.
                `m` for AVAR, `1` for OAVAR.

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    N = x.size

    d = 2  # second difference algorithm

    if N < d*m + 1:
        logger.warning("Not enough phase measurements to compute "
                       "variance at averaging factor %i: %s", m, x)
        var = np.NaN
        return var, 0

    # Calculate second differences
    summand = x[2*m::stride] - 2*x[m:-m:stride] + x[:-2*m:stride]
    n = summand[~np.isnan(summand)].size  # (N-2*m)/stride if no NaNs

    if n == 0:
        logger.warning("Not enough valid phase measurements to compute "
                       "variance at averaging factor %i: %s", m, x)
        var = np.NaN
        return var, 0

    # Calculate variance
    var = 1. / (2 * tau**2) * np.nanmean(summand**2)

    return VarResult(var=var, n=n)


def calc_avar(x: Array, m: int, tau: float) -> VarResult:
    """Main algorithm for AVAR calculation.

    References:
        [RileyStable32]_ (5.2.2, pg.19)
        [Wikipedia]_
        http://www.leapsecond.com/tools/adev_lib.c

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    return calc_o_avar(x=x, m=m, tau=tau, stride=m)


def calc_oavar(x: Array, m: int, tau: float) -> VarResult:
    """Main algorithm for OAVAR calculation.

    References:
        [RileyStable32]_ (5.2.2, pg.19)
        [Wikipedia]_
        http://www.leapsecond.com/tools/adev_lib.c

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    return calc_o_avar(x=x, m=m, tau=tau, stride=1)


def calc_mvar(x: Array, m: int, tau: float) -> VarResult:
    """Main algorithm for MVAR calculation.

    This implementation is a `loop-unrolled` algorithm, as per
    http://www.leapsecond.com/tools/adev_lib.c


    References:
        [RileyStable32]_ (5.2.5, pg.22)
        TODO: add reference justifying loop unrolled algorithm

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
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
    s /= 2 * m**2 * tau**2 * n

    return VarResult(var=s, n=n)


def calc_tvar(x: Array, m: int, tau: float) -> VarResult:
    """Main algorithm for TVAR calculation.

    References:
        [RileyStable32]_ (5.2.6, pg.23)

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    mvar, n = calc_mvar(x=x, m=m, tau=tau)

    var = (tau**2 / 3) * mvar

    return VarResult(var=var, n=n)


def calc_o_hvar(x: Array, m: int, tau: float, stride: int) -> VarResult:
    """Main algorithm for HVAR and OHVAR calculation.

    References:
        [RileyStable32]_ (5.2.8-9, pg.25-27)
        http://www.leapsecond.com/tools/adev_lib.c

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time
        stride: stride at which to parse input phase data.
                `m` for HVAR, `1` for OHVAR.

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    N = x.size

    d = 3  # third difference algorithm

    if N < d*m + 1:
        logger.warning("Not enough phase measurements to compute "
                       "variance at averaging factor %i: %s", m, x)
        var = np.NaN
        return var, 0

    # Calculate third differences
    summand = x[3*m::stride] - 3*x[2*m:-m:stride] + 3*x[m:-2*m:stride] - x[:-3*m:stride]
    n = summand[~np.isnan(summand)].size  # (N-3*m)/stride if no NaNs

    if n == 0:
        logger.warning("Not enough valid phase measurements to compute "
                       "variance at averaging factor %i: %s", m, x)
        var = np.NaN
        return var, 0

    # Calculate variance
    var = 1. / (6 * tau**2) * np.nanmean(summand**2)

    return VarResult(var=var, n=n)


def calc_hvar(x: Array, m: int, tau: float) -> VarResult:
    """Main algorithm for HVAR calculation.

    References:
        [RileyStable32]_ (5.2.8, pg.25-26)

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    return calc_o_hvar(x=x, m=m, tau=tau, stride=m)


def calc_ohvar(x: Array, m: int, tau: float) -> VarResult:
    """Main algorithm for OHVAR calculation.

    References:
        [RileyStable32]_ (5.2.9, pg.26-27)

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    return calc_o_hvar(x=x, m=m, tau=tau, stride=1)


def calc_totvar(x: Array, m: int, tau: float) -> VarResult:
    """Main algorithm for TOTVAR calculation.

    References:
        [RileyStable32]_ (5.2.11, pg.29-31)

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    N = x.size

    # Need more than 3 values to build an extended dataset with at least 3
    # samples on which to calculate algorithm
    if N < 3:
        logger.warning("Not enough phase measurements to compute "
                       "variance at averaging factor %i: %s", m, x)
        var = np.NaN
        return var, 0

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
        var = np.NaN
        return var, 0

    # Calculate variance
    var = 1. / (2 * tau**2) * np.nanmean(summand**2)

    return VarResult(var=var, n=n)


def calc_mtotvar(x: Array, m: int, tau: float) -> VarResult:
    """Main algorithm for TOTVAR calculation.

    PRELIMINARY - REQUIRES FURTHER TESTING.

    References:
        [RileyStable32]_ (5.2.12, pg.31-32)
        [Howe1999]_
        D.A. Howe and F. Vernotte, "Generalization of the Total Variance
        Approach to the Modified Allan Variance," Proc.
        31 st PTTI Meeting, pp. 267-276, Dec. 1999.
        TODO: find justification for last part of algorithm

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
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
    var = var * 1.0 / (2.0 * pow(tau, 2) * (N - 3 * m + 1))

    return VarResult(var=var, n=n)


def calc_ttotvar(x: Array, m: int, tau: float) -> VarResult:
    """Main algorithm for TTOTVAR calculation.

    PRELIMINARY - REQUIRES FURTHER TESTING.

    References:
        [RileyStable32]_ (5.2.13, pg.31)

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    mtotvar, n = calc_mtotvar(x=x, m=m, tau=tau)

    var = (tau**2 / 3) * mtotvar

    return VarResult(var=var, n=n)


def calc_htotvar(x: Array, m: int, tau: float) -> VarResult:
    """Main algorithm for HTOTVAR calculation.

    PRELIMINARY - REQUIRES FURTHER TESTING.

    References:
        [RileyStable32]_ (5.2.14, pg.33-37)
        http://www.wriley.com/paper4ht.htm

    Args:
        x:      input FRACTIONAL FREQUENCY data.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    # TODO: make gap resistant and return correct number of non-NaN samples

    # NOTE: had to call parameter `x` to match the signature of all other
    # callables, but this functions operates on fractional frequency datasets!!
    # so this `x` should be an array of frequencies
    y = x

    # "For best consistency, the overlapping Hadamard variance is used
    # instead of the Hadamard total variance at m=1"
    # FIXME: this uses both freq and phase datasets, which uses double the
    #  memory really needed...
    if m == 1:

        x = utils.frequency2phase(y=y, rate=m/tau)
        var, n = calc_hvar(x=x, m=m, tau=tau)
        return var, n

    else:

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


def calc_theo1(x: Array, m: int, tau: float) -> VarResult:
    """Main algorithm for THEO1 calculation.

    PRELIMINARY - REQUIRES FURTHER TESTING.

    References:
        [RileyStable32]_ (5.2.15, pg.37-8)

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding theo1 `effective` averaging time. This should be
                0.75 of the normal m*tau_0

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    if m % 2 != 0 or m < 10:  # m must be even and >= 10
        logger.warning("Theo1 statistic is not compatible with an "
                       "averaging factor m=%i", m)
        return np.NaN, 0

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

        # tau passed here is 0.75*m*tau_0 -> so need to multiply by 0.75
        # nominator to get a single 0.75 multiplier on the denominator
        var = (0.75 / tau**2) * outer_mean

        return VarResult(var=var, n=n)


def calc_mtie(x: Array, m: int, tau: float = None) -> VarResult:
    """Main algorithm for MTIE var calculation.

    References:
        [RileyStable32]_ (5.2.17, pg.41-42)

    Args:
        x:      input phase data, in units of seconds.
        m:      averaging factor at which to calculate variance
        tau:    corresponding averaging time.

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    # Move an n-point window through the phase data, and find the difference
    # between the max nad min values at each windows position
    n = m + 1  # n is defined as m+1

    if m < 1 or m >= x.size:
        logger.warning("Cannot calculate MTIE over this time interval: %f",
                       n*tau/m)
        return np.NaN, 0

    grads = utils.rolling_grad(x, n)
    assert grads.size == x.size - (n-1)
    n = grads.size

    # MTIE is the overall maximum of this time interval error
    mtie = np.max(grads)

    # Need to return variance to match function signature
    var = mtie**2

    return var, n


# FIXME: mtie_phase_fast() is incomplete.
# TODO: Complete and swap in for when `fastu` selected
def calc_mtie_fast(phase, rate=1.0, data_type="phase", taus=None):
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


def calc_tierms(x: Array, m: int, tau: float = None) -> VarResult:
    """Main algorithm for TIErms calculation.

    References:
        [RileyStable32]_ (5.2.18, pg.42-43)

    Args:
        x:              input phase data, in units of seconds.
        m:              averaging factor at which to calculate variance
        tau (optional): corresponding averaging time. Not used here.

    Returns:
        (var, n) NamedTuple of computed variance at given averaging time, and
        number of samples used to estimate it.
    """

    summand = x[m:] - x[:-m]
    n = summand[~np.isnan(summand)].size  # x.size - m if no NaNs

    var = 1*np.nanmean(summand**2)

    return VarResult(var=var, n=n)


# FIXME: integrate this in normal adev i.e. make all stats gap resistant
def calc_gradev(data, rate, mj, stride, confidence, noisetype):
    """ see http://www.leapsecond.com/tools/adev_lib.c
        stride = mj for nonoverlapping allan deviation
        stride = 1 for overlapping allan deviation

        [Wikipedia]_
        see http://en.wikipedia.org/wiki/Allan_variance

    .. math::

        \\sigma^2_{y}(t) = { 1 \\over ( 2 \\tau^2 } sum [x(i+2) - 2x(i+1) + x(i) ]^2

    """

    d2 = data[2 * int(mj)::int(stride)]
    d1 = data[1 * int(mj)::int(stride)]
    d0 = data[::int(stride)]

    n = min(len(d0), len(d1), len(d2))

    v_arr = d2[:n] - 2 * d1[:n] + d0[:n]

    n = len(np.where(np.isnan(v_arr) == False)[0]) # only average for non-nans

    if n == 0:
        RuntimeWarning("Data array length is too small: %i" % len(data))
        n = 1

    N = len(np.where(np.isnan(data) == False)[0])

    s = np.nansum(v_arr * v_arr)   #  a summation robust to nans

    dev = np.sqrt(s / (2.0 * n)) / mj  * rate
    #deverr = dev / np.sqrt(n) # old simple errorbars
    if noisetype == 'wp':
        alpha = 2
    elif noisetype == 'wf':
        alpha = 0
    elif noisetype == 'fp':
        alpha = -2
    else:
        alpha = None

    if n > 1:
        edf = ci.edf_simple(N, mj, alpha)
        deverr = ci.confidence_interval(dev, confidence, edf)
    else:
        deverr = [0, 0]

    return dev, deverr, n