from . import ci
from . import utils
import numpy as np

Array = np.ndarray


def calc_avar(x, m, tau):
    """Main algorithm for adev() and oadev() calculations.

       References:
           [SP1065]_ eqn (7) and (11) page 16
           [Wikipedia]_
           http://www.leapsecond.com/tools/adev_lib.c

       Args:
           x:      input phase data, in units of seconds.
           rate:   sampling rate of the input data, in Hz.
           af:     averaging factor at which to calculate deviation
           stride: size of stride. 1 for overlapping, `af` for non-overlapping

       Returns:
           (dev, deverr, n) tuple of computed deviation, estimated error,
           and number of samples used to estimate it
       """

    # minimum number of values needed by algorithm:
    min_N = 3  # i --> i+2

    # Decimate input data, to get sample at this averaging factor
    x = x[::m]

    # Size of sample
    N = x.size

    if N < min_N:
        RuntimeWarning("Data array length is too small: %i" % len(x))
        N = min_N

    # Calculate variance
    var = 1. / (2 * (N-2) * tau**2) * np.sum((x[2:] - 2*x[1:-1] + x[:-2])**2)

    # num values looped through to gen variance
    n = N - 2  # capped by i+2

    return var, n


def calc_oavar(x, m, tau):

    # minimum number of values needed by algorithm:
    min_N = 2*m + 1  # i --> i+2m

    N = x.size

    if N < min_N:
        RuntimeWarning("Data array length is too small: %i" % len(x))
        N = min_N

    # Calculate variance
    var = 1. / (2 * (N-2*m) * tau**2) * np.sum(
        (x[2*m:] - 2*x[m:-m] + x[:-2*m])**2)

    # num values looped through to gen variance
    n = N - 2*m  # capped by i+2m

    return var, n


def calc_mvar(x, m, tau):

    # this is a 'loop-unrolled' algorithm following
    # http://www.leapsecond.com/tools/adev_lib.c

    # First loop sum
    d0 = x[0:m]
    d1 = x[m:2 * m]
    d2 = x[2 * m:3 * m]
    e = min(len(d0), len(d1), len(d2))

    v = np.sum(d2[:e] - 2 * d1[:e] + d0[:e])
    s = v * v

    # Second part of sum
    d3 = x[3 * m:]
    d2 = x[2 * m:]
    d1 = x[1 * m:]
    d0 = x[0:]

    e = min(len(d0), len(d1), len(d2), len(d3))
    n = e + 1

    v_arr = v + np.cumsum(d3[:e] - 3 * d2[:e] + 3 * d1[:e] - d0[:e])

    s = s + np.sum(v_arr * v_arr)
    s /= 2 * m**2 * tau**2 * n

    return s, n


def calc_tvar(x, m, tau):

    mvar, n = calc_mvar(x=x, m=m, tau=tau)

    tvar = (tau**2 / 3) * mvar

    return tvar, n


def calc_hvar(x, m, tau):
    """ main calculation fungtion for HDEV and OHDEV

    Parameters
    ----------
    phase: np.array
        Phase data in seconds.
    rate: float
        The sampling rate for phase or frequency, in Hz
    mj: int
        M index value for stride
    stride: int
        Size of stride

    Returns
    -------
    (dev, deverr, n): tuple
        Array of computed values.

    Notes
    -----
    http://www.leapsecond.com/tools/adev_lib.c

    .. math::

        \\sigma^2_{y}(t) = { 1 \\over 6\\tau^2 (N-3m) }
            \\sum_{i=1}^{N-3} [ x(i+3) - 3x(i+2) + 3x(i+1) - x(i) ]^2

        N=M+1 phase measurements
        m is averaging factor

    NIST [SP1065]_ eqn (18) and (20) pages 20 and 21
    """

    # Decimate values (non-overlapping)
    x = x[::m]

    d = 3  # 3rd difference algorithm

    # minimum number of values needed by algorithm:
    min_N = d + 1

    N = x.size

    # del with invalid number of samples
    if N < min_N:
        RuntimeWarning("Data array length is too small: %i" % len(x))
        N = min_N

    # Calculate variance
    var = 1. / (6. * tau**2 * (N-3)) * np.sum(
        (x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3])**2
    )

    # num values looped through to gen variance
    n = N-3  # sum i=1 --> N-3

    return var, n


def calc_ohvar(x, m, tau):

    d = 3  # 3rd difference algorithm

    # minimum number of values needed by algorithm:
    min_N = d*m + 1  # (overlapping)

    N = x.size

    # del with invalid number of samples
    if N < min_N:
        RuntimeWarning("Data array length is too small: %i" % len(x))
        N = min_N

    # Calculate variance
    var = 1. / (6. * tau ** 2 * (N - 3*m)) * np.sum(
        (x[3*m:] - 3 * x[2*m:-1*m] + 3 * x[1*m:-2*m] - x[:-3*m]) ** 2
    )

    # num values looped through to gen variance
    n = N - 3*m  # sum i=1 --> N-3

    return var, n


# TODO: Remove
def calc_hdev(phase, rate, mj, stride):

    tau0 = 1.0 / float(rate)
    mj = int(mj)
    stride = int(stride)
    d3 = phase[3 * mj::stride]
    d2 = phase[2 * mj::stride]
    d1 = phase[1 * mj::stride]
    d0 = phase[::stride]

    n = min(len(d0), len(d1), len(d2), len(d3))

    v_arr = d3[:n] - 3 * d2[:n] + 3 * d1[:n] - d0[:n]

    s = np.sum(v_arr * v_arr)

    if n == 0:
        n = 1

    h = np.sqrt(s / 6.0 / float(n)) / float(tau0 * mj)
    e = h / np.sqrt(n)
    return h, e, n


def calc_totvar(x, m, tau):

    N = x.size

    # extend dataset by reflection, as required by totdev
    xxx = np.pad(x, N-1, 'symmetric', reflect_type='odd')
    xxx = np.delete(xxx, [N-2, -N+1])  # pop duplicate edge values
    M = len(xxx)
    assert M == 3 * N - 4

    # index of start of original dataset
    i0 = N - 2  # `i` = 1

    d0 = xxx[i0 + 1:]
    d1 = xxx[i0 + 1 + m:]
    d1n = xxx[i0 + 1 -m:]
    e = min(len(d0), len(d1), len(d1n))

    # Calculate variance
    var = 1. / (2 * tau**2 * (N - 2)) * np.sum(
        (d1n[:e] - 2.0 * d0[:e] + d1[:e])[:i0]**2
    )

    # num values looped through to gen variance
    n = N - 2  # sum i=2 --> N-1

    return var, n


def calc_mtotvar(x, m, tau):
    """ PRELIMINARY - REQUIRES FURTHER TESTING.
    calculation of mtotdev for one averaging factor m; tau = m*tau0

    NIST [SP1065]_ Eqn (27), page 25.

    Computed from a set of N - 3m + 1 subsequences of 3m points.

    1. A linear trend (frequency offset) is removed from the subsequence by
    averaging the first and last halves of the subsequence and dividing by
    half the interval.
    2. The offset-removed subsequence is extended at both ends by
    uninverted, even reflection.

    [Howe1999]_
    D.A. Howe and F. Vernotte, "Generalization of the Total Variance
    Approach to the Modified Allan Variance," Proc.
    31 st PTTI Meeting, pp. 267-276, Dec. 1999.
    """

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

        squaresum = (1.0 / (6.0 * m)) * squaresum
        var += squaresum
        n = n + 1

    # scaling in front of double sum
    assert n == N - 3 * m + 1  # sanity check on the number of terms n
    var = var * 1.0 / (2.0 * pow(tau, 2) * (N - 3 * m + 1))

    return var, n


def calc_ttotvar(x, m, tau):

    mtotvar, n = calc_mtotvar(x=x, m=m, tau=tau)

    ttotvar = (tau**2 / 3) * mtotvar

    return ttotvar, n


def calc_htotvar(x, m, tau):
    """ PRELIMINARY - REQUIRES FURTHER TESTING.
        calculation of htotdev for one averaging factor m
        tau = m*tau0

        Parameters
        ----------
        frequency: np.array
            Fractional frequency data (nondimensional).
        m: int
            Averaging factor. tau = m*tau0, where tau0=1/rate.
    """

    # NOTE: had to call parameter `x` to match the signature of all other
    # callables, but this functions operates on fractional frequency datasets!!
    # so this `x` should be an array of frequencies
    freq = x

    # NOTE at mj==1 we use ohdev(), based on comment from here:
    # http://www.wriley.com/paper4ht.htm
    # "For best consistency, the overlapping Hadamard variance is used
    # instead of the Hadamard total variance at m=1"
    # FIXME: this uses both freq and phase datasets, which uses double the
    #  memory really needed...
    if m == 1:
        x = utils.frequency2phase(y=freq, rate=m/tau)
        var, n = calc_hvar(x=x, m=m, tau=tau)
        return var, n

    else:
        N = int(len(freq))  # frequency data, N points

        n = 0  # number of terms in the sum, for error estimation
        var = 0.0  # the deviation we are computing
        for i in range(0, N - 3 * m + 1):
            # subsequence of length 3m, from the original phase data
            xs = freq[i:i + 3 * m]
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

        return var, n


def calc_theo1(x, m, tau):

    assert m % 2 == 0  # m must be even

    N = x.size

    var = 0
    n = 0
    for i in range(int(N - m)):
        s = 0
        for d in range(int(m / 2)):  # inner sum
            pre = 1.0 / (float(m) / 2 - float(d))
            s += pre * pow(x[i] - x[i - d + int(m / 2)] +
                           x[i + m] - x[i + d + int(m / 2)], 2)
            n = n + 1
        var += s
    assert n == (N - m) * m / 2  # N-m outer sums, m/2 inner sums

    var = var / (0.75 * (N - m) * tau**2)
    # factor 0.75 used here? http://tf.nist.gov/general/pdf/1990.pdf
    # but not here? http://tf.nist.gov/timefreq/general/pdf/2220.pdf page 29

    # TODO: is the effective n to return this one or just N-m?
    # allantools had it at N-m...
    return var, n


def calc_mtie(x, m, tau=None):

    try:
        # the older algorithm uses a lot of memory
        # but can be used for short datasets.
        rw = utils.mtie_rolling_window(a=x, window=m+1)
        win_max = np.max(rw, axis=1)
        win_min = np.min(rw, axis=1)
        tie = win_max - win_min
        dev = np.max(tie)

    except:
        if int(m + 1) < 1:
            raise ValueError("`window` must be at least 1.")
        if int(m + 1) > x.shape[-1]:
            raise ValueError("`window` is too long.")

        currMax = np.max(x[0:m])
        currMin = np.min(x[0:m])
        dev = currMax - currMin
        for winStartIdx in range(1, int(x.shape[0] - m)):
            winEndIdx = m + winStartIdx
            if currMax == x[winStartIdx - 1]:
                currMax = np.max(x[winStartIdx:winEndIdx])
            elif currMax < x[winEndIdx]:
                currMax = x[winEndIdx]

            if currMin == x[winStartIdx - 1]:
                currMin = np.min(x[winStartIdx:winEndIdx])
            elif currMin > x[winEndIdx]:
                currMin = x[winEndIdx]

            if dev < currMax - currMin:
                dev = currMax - currMin

    var = dev**2
    ncount = x.shape[0] - m

    return var, ncount


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