from . import ci
import numpy as np


def oadev_core(x, af, rate, stride):
    """  Main algorithm for adev() (stride=mj) and oadev() (stride=1)

        see http://www.leapsecond.com/tools/adev_lib.c
        stride = mj for nonoverlapping allan deviation

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
    stride = mj for nonoverlapping Allan deviation
    stride = 1 for overlapping Allan deviation

    References
    ----------
    [Wikipedia]_
    * http://en.wikipedia.org/wiki/Allan_variance
    * http://www.leapsecond.com/tools/adev_lib.c

    NIST [SP1065]_ eqn (7) and (11) page 16
    """
    mj = int(af)
    stride = int(stride)
    d2 = x[2 * mj::stride]
    d1 = x[1 * mj::stride]
    d0 = x[::stride]

    n = min(len(d0), len(d1), len(d2))

    if n == 0:
        RuntimeWarning("Data array length is too small: %i" % len(x))
        n = 1

    v_arr = d2[:n] - 2 * d1[:n] + d0[:n]
    s = np.sum(v_arr * v_arr)

    dev = np.sqrt(s / (2.0 * n)) / mj  * rate
    deverr = dev / np.sqrt(n)

    return dev, deverr, n


def calc_adev(x, af, rate):
    return oadev_core(x=x, af=af, rate=rate, stride=af)


def calc_oadev(x, af, rate):
    return oadev_core(x=x, af=af, rate=rate, stride=1)


def calc_hdev(phase, rate, mj, stride):
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


def calc_mtotdev(phase, rate, m):
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
    tau0 = 1.0 / rate
    N = len(phase)  # phase data, N points
    m = int(m)

    n = 0  # number of terms in the sum, for error estimation
    dev = 0.0  # the deviation we are computing
    err = 0.0  # the error in the deviation
    # print('calc_mtotdev N=%d m=%d' % (N,m) )
    for i in range(0, N - 3 * m + 1):
        # subsequence of length 3m, from the original phase data
        xs = phase[i:i + 3 * m]
        assert len(xs) == 3 * m
        # Step 1.
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
            slope = (mean2 - mean1) / ((0.5 * (3 * m - 1) + 1) * tau0)
        else:  # m is even
            # 3m = 2k is even, so distance between averages is k=m/2
            slope = (mean2 - mean1) / (0.5 * 3 * m * tau0)

        # remove the linear trend
        x0 = [x - slope * idx * tau0 for (idx, x) in enumerate(xs)]
        x0_flip = x0[::-1]  # left-right flipped version of array

        # Step 2.
        # extend sequence, by uninverted even reflection
        # extended sequence xstar, of length 9m,
        xstar = np.concatenate((x0_flip, x0, x0_flip))
        assert len(xstar) == 9 * m

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
        dev += squaresum
        n = n + 1

    # scaling in front of double-sum
    assert n == N - 3 * m + 1  # sanity check on the number of terms n
    dev = dev * 1.0 / (2.0 * pow(m * tau0, 2) * (N - 3 * m + 1))
    dev = np.sqrt(dev)
    error = dev / np.sqrt(n)
    return (dev, error, n)


def calc_htotdev(freq, m):
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

    N = int(len(freq))  # frequency data, N points
    m = int(m)
    n = 0  # number of terms in the sum, for error estimation
    dev = 0.0  # the deviation we are computing
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
        dev += squaresum
        n = n + 1

    # scaling in front of double-sum
    assert n == N - 3 * m + 1  # sanity check on the number of terms n
    dev = dev * 1.0 / (N - 3 * m + 1)
    dev = np.sqrt(dev)
    error = dev / np.sqrt(n)
    return (dev, error, n)


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