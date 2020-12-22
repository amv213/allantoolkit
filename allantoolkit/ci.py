"""
this file is part of allantoolkit, https://github.com/aewallin/allantools

- functions for confidence intervals
- functions for computing equivalent degrees of freedom
"""

import logging
import numpy as np
import scipy.special
import scipy.stats  # used in confidence_intervals()
import scipy.signal  # decimation in lag-1 acf


# Spawn module-level logger
logger = logging.getLogger(__name__)

# Confidence Intervals
ONE_SIGMA_CI = scipy.special.erf(1/np.sqrt(2))
#    = 0.68268949213708585

# shorten type hint to save some space
Array = np.ndarray


def get_error_bars(dev, m, tau, n, alpha=0, d=2, overlapping=True,
                   modified=False):
    """Calculate non-naive Allan deviation errors. Equivalent to Stable32.

    References:
        [RileyStable32]_ (5.3, pg.45-46)
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050061319.pdf

    Args:
        dev:                    raw deviation for which to compute error bars
        m:                      averaging factor at which deviation was
                                computed
        tau:                    averaging time  for which deviation was
                                computed
        n:                      number of samples on which deviation was
                                computed
        alpha (optional):       +2,...,-4 dominant noise type , either
                                estimated or known
        d (optional):           statistic code: 1 first-difference variance,
                                2 allan variance, 3 hadamard variance
        overlapping (optional): True if overlapping statistic used. False if
                                standard statistic used
        modified (optional):    True if modified statistic used. False if
                                standard statistic used.

    Returns:
        err_lo:                 non-naive lower 1-sigma error bar
        err_high:               non-naive higher 1-sigma error_bar
    """

    # Greenhalls EDF (Equivalent Degrees of Freedom)
    edf = edf_greenhall(alpha=alpha, d=d, m=m, N=n, overlapping=overlapping,
                        modified=modified)


    # with the known EDF we get CIs
    (lo, hi) = confidence_interval(dev=dev, edf=edf)

    err_lo = dev - lo
    err_hi = hi - dev

    return err_lo, err_hi


def confidence_interval(dev, edf, ci=ONE_SIGMA_CI):
    """ returns confidence interval (dev_min, dev_max)
        for a given deviation dev, equivalent degrees of freedom edf,
        and degree of confidence ci.

    Parameters
    ----------
    dev: float
        Mean value (e.g. adev) around which we produce the confidence interval
    edf: float
        Equivalent degrees of freedon
    ci: float, defaults to scipy.special.erf(1/math.sqrt(2)) for 1-sigma
    standard error set ci = scipy.special.erf(1/math.sqrt(2)) =
    0.68268949213708585

    Returns
    -------
    (dev_min, dev_max): (float, float)
        Confidence interval
    """
    ci_l = min(np.abs(ci), np.abs((ci-1))) / 2
    ci_h = 1 - ci_l

    # function from scipy, works OK, but scipy is large and slow to build
    chi2_l = scipy.stats.chi2.ppf(ci_l, edf)
    chi2_h = scipy.stats.chi2.ppf(ci_h, edf)

    variance = dev*dev
    var_l = float(edf) * variance / chi2_h  # NIST SP1065 eqn (45)
    var_h = float(edf) * variance / chi2_l

    return np.sqrt(var_l), np.sqrt(var_h)


def confidence_interval_noiseID(x, dev, af, dev_type="adev", data_type="phase", ci=ONE_SIGMA_CI):
    """ returns confidence interval (dev_min, dev_max)
        for a given deviation dev = Xdev( x, tau = af*(1/rate) )

        steps:
        1) identify noise type
        2) compute EDF
        3) compute confidence interval

    Parameters
    ----------
    x: numpy.array
        time-series
    dev: float
        Mean value (e.g. adev) around which we produce the confidence interval
    af: int
        averaging factor
    dev_type: string
        adev, oadev, mdev, tdev, hdev, ohdev
    data_type:
        "phase" or "freq"
    ci: float, defaults to scipy.special.erf(1/math.sqrt(2)) for 1-sigma
    standard error set ci = scipy.special.erf(1/math.sqrt(2)) =
    0.68268949213708585

    Returns
    -------
    (dev_min, dev_max): (float, float)
        Confidence interval
    """

    # 1) noise ID
    dmax = 2
    if (dev_type == "hdev") or (dev_type == "ohdev"):
        dmax = 3

    alpha_int = autocorr_noise_id(x, int(af), data_type=data_type, dmin=0,
                                  dmax=dmax)[0]

    # 2) EDF
    if dev_type == "adev":
        edf = edf_greenhall(alpha=alpha_int, d=2, m=af, N=len(x),
                            overlapping=False, modified=False)
    elif dev_type == "oadev":
        edf = edf_greenhall(alpha=alpha_int, d=2, m=af, N=len(x),
                            overlapping=True, modified=False)
    elif (dev_type == "mdev") or (dev_type == "tdev"):
        edf = edf_greenhall(alpha=alpha_int, d=2, m=af, N=len(x),
                            overlapping=True, modified=True)
    elif dev_type == "hdev":
        edf = edf_greenhall(alpha=alpha_int, d=3, m=af, N=len(x),
                            overlapping=False, modified=False)
    elif dev_type == "ohdev":
        edf = edf_greenhall(alpha=alpha_int, d=3, m=af, N=len(x),
                            overlapping=True, modified=False)
    else:
        raise NotImplementedError

    # 3) confidence interval
    (low, high) = confidence_interval(dev, edf, ci)

    return low, high

# Equivalent Degrees of Freedom


def edf_greenhall_simple(alpha, d, m, S, F, N):
    """ Eqn (13) from [Greenhall2004]_ """
    L = m/F+m*d # length of filter applied to phase samples
    M = 1 + np.floor(S*(N-L) / m)
    J = min(M, (d+1)*S)
    inv_edf = (1.0/(pow(greenhall_sz(0, F, alpha, d), 2)*M)) * greenhall_BasicSum(J, M, S, F, alpha, d)
    return 1.0/inv_edf


def edf_greenhall(alpha, d, m, N, overlapping=False, modified=False, verbose=False):
    """ returns Equivalent degrees of freedom

        Parameters
        ----------
        alpha: int
            noise type, +2...-4
        d: int
            1 first-difference variance
            2 Allan variance
            3 Hadamard variance
            require alpha+2*d>1
        m: int
            averaging factor
            tau = m*tau0 = m*(1/rate)
        N: int
            number of phase observations (length of time-series)
        overlapping: bool
            True for oadev, ohdev
        modified: bool
            True for mdev, tdev

        Returns
        -------
        edf: float
            Equivalent degrees of freedom

        Greenhall, Riley, 2004
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050061319.pdf
        UNCERTAINTY OF STABILITY VARIANCES BASED ON FINITE DIFFERENCES

        Notes
        -----
        Used for the following deviations (see http://www.wriley.com/CI2.pdf page 8)
        adev()
        oadev()
        mdev()
        tdev()
        hdev()
        ohdev()
    """

    if modified:
        F = 1 # F filter factor, 1 modified variance, m unmodified variance
    else:
        F = int(m)
    if overlapping: # S stride factor, 1 nonoverlapped estimator,
        S = int(m)  # m overlapped estimator (estimator stride = tau/S )
    else:
        S = 1
    assert(alpha+2*d > 1.0)
    L = m/F+m*d # length of filter applied to phase samples
    M = 1 + np.floor(S*(N-L) / m)
    J = min(M, (d+1)*S)
    J_max = 100
    r = M/S
    if int(F) == 1 and modified: # case 1, modified variances, all alpha
        if J <= J_max:
            inv_edf = (1.0/(pow(greenhall_sz(0, 1, alpha, d), 2)*M))* \
                       greenhall_BasicSum(J, M, S, 1, alpha, d)
            if verbose:
                print("case 1.1 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
        elif r > d+1:
            (a0, a1) = greenhall_table1(alpha, d)
            inv_edf = (1.0/r)*(a0-a1/r)
            if verbose:
                print("case 1.2 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
        else:
            m_prime = J_max/r
            inv_edf = (1.0/(pow(greenhall_sz(0, F, alpha, d), 2)*J_max))* \
                       greenhall_BasicSum(J_max, J_max, m_prime, 1, alpha, d)
            if verbose:
                print("case 1.3 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
    elif int(F) == int(m) and int(alpha) <= 0 and not modified: 
        # case 2, unmodified variances, alpha <= 0
        if J <= J_max:
            if m*(d+1) <= J_max:
                m_prime = m
                variant = "a"
            else:
                m_prime = float('inf')
                variant = "b"

            inv_edf = (1.0/(pow(greenhall_sz(0, m_prime, alpha, d), 2)*M))* \
                       greenhall_BasicSum(J, M, S, m_prime, alpha, d)
            if verbose:
                print("case 2.1%s edf= %3f" % (variant, float(1.0/inv_edf)))
            return 1.0/inv_edf
        elif r > d+1:
            (a0, a1) = greenhall_table2(alpha, d)
            inv_edf = (1.0/r)*(a0-a1/r)
            if verbose:
                print("case 2.2 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
        else:
            m_prime = J_max/r
            inv_edf = (1.0/(pow(greenhall_sz(0, float('inf'), alpha, d), 2)*J_max))* \
                       greenhall_BasicSum(J_max, J_max, m_prime, float('inf'), alpha, d)
            if verbose:
                print("case 2.3 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
    elif int(F) == int(m) and int(alpha) == 1 and not modified:
        # case 3, unmodified variances, alpha=1
        if J <= J_max:
            inv_edf = (1.0/(pow(greenhall_sz(0, m, 1, d), 2)*M))* \
                       greenhall_BasicSum(J, M, S, m, 1, d) # note: m<1e6 to avoid roundoff
            if verbose:
                print("case 3.1 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
        elif r > d+1:
            (a0, a1) = greenhall_table2(alpha, d)
            (b0, b1) = greenhall_table3(alpha, d)
            inv_edf = (1.0/(pow(b0+b1*np.log(m), 2)*r))*(a0-a1/r)
            if verbose:
                print("case 3.2 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
        else:
            m_prime = J_max/r
            (b0, b1) = greenhall_table3(alpha, d)
            inv_edf = (1.0/(pow(b0+b1*np.log(m), 2)*J_max))* \
                       greenhall_BasicSum(J_max, J_max, m_prime, m_prime, 1, d)
            if verbose:
                print("case 3.3 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
    elif int(F) == int(m) and int(alpha) == 2 and not modified:
        # case 4, unmodified variances, alpha=2
        K = np.ceil(r)
        if K <= d:
            raise NotImplementedError  # FIXME: add formula from the paper here!
        else:
            a0 = scipy.special.binom(4*d, 2*d) / pow(scipy.special.binom(2*d, d), 2)
            a1 = d/2.0
            inv_edf = (1.0/M)*(a0-a1/r)
            if verbose:
                print("case 4.2 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf

    print("greenhall_edf() no matching case!")

    raise NotImplementedError


def greenhall_BasicSum(J, M, S, F, alpha, d):
    """ Eqn (10) from [Greenhall2004]_ """
    first = pow(greenhall_sz(0, F, alpha, d), 2)
    second = (1-float(J)/float(M))*pow(greenhall_sz(float(J)/float(S), F, alpha, d), 2)
    third = 0
    for j in range(1, int(J)):
        third += 2*(1.0-float(j)/float(M))*pow(greenhall_sz(float(j)/float(S), F, alpha, d), 2)
    return first+second+third


def greenhall_sz(t, F, alpha, d):
    """ Eqn (9) from [Greenhall2004]_ """
    if d == 1:
        a = 2*greenhall_sx(t, F, alpha)
        b = greenhall_sx(t-1.0, F, alpha)
        c = greenhall_sx(t+1.0, F, alpha)
        return a-b-c
    elif d == 2:
        a = 6*greenhall_sx(t, F, alpha)
        b = 4*greenhall_sx(t-1.0, F, alpha)
        c = 4*greenhall_sx(t+1.0, F, alpha)
        dd = greenhall_sx(t-2.0, F, alpha)
        e = greenhall_sx(t+2.0, F, alpha)
        return a-b-c+dd+e
    elif d == 3:
        a = 20.0*greenhall_sx(t, F, alpha)
        b = 15.0*greenhall_sx(t-1.0, F, alpha)
        c = 15.0*greenhall_sx(t+1.0, F, alpha)
        dd = 6.0*greenhall_sx(t-2.0, F, alpha)
        e = 6.0*greenhall_sx(t+2.0, F, alpha)
        f = greenhall_sx(t-3.0, F, alpha)
        g = greenhall_sx(t+3.0, F, alpha)
        return a-b-c+dd+e-f-g

    assert 0  # ERROR


def greenhall_sx(t, F, alpha):
    """ Eqn (8) from [Greenhall2004]_
    """
    if F == float('inf'):
        return greenhall_sw(t, alpha+2)
    a = 2*greenhall_sw(t, alpha)
    b = greenhall_sw(t-1.0/float(F), alpha)
    c = greenhall_sw(t+1.0/float(F), alpha)

    return pow(F, 2)*(a-b-c)


def greenhall_sw(t, alpha):
    """ Eqn (7) from [Greenhall2004]_
    """
    alpha = int(alpha)
    if alpha == 2:
        return -np.abs(t)
    elif alpha == 1:
        if t == 0:
            return 0
        else:
            return pow(t, 2)*np.log(np.abs(t))
    elif alpha == 0:
        return np.abs(pow(t, 3))
    elif alpha == -1:
        if t == 0:
            return 0
        else:
            return pow(t, 4)*np.log(np.abs(t))
    elif alpha == -2:
        return np.abs(pow(t, 5))
    elif alpha == -3:
        if t == 0:
            return 0
        else:
            return pow(t, 6)*np.log(np.abs(t))
    elif alpha == -4:
        return np.abs(pow(t, 7))

    assert 0  # ERROR


def greenhall_table3(alpha, d):
    """ Table 3 from Greenhall 2004 """
    assert(alpha == 1)
    idx = d-1
    table3 = [(6.0, 4.0), (15.23, 12.0), (47.8, 40.0)]
    return table3[idx]


def greenhall_table2(alpha, d):
    """ Table 2 from Greenhall 2004 """
    row_idx = int(-alpha+2) # map 2-> row0 and -4-> row6
    assert(row_idx in [0, 1, 2, 3, 4, 5])
    col_idx = int(d-1)
    table2 = [[(3.0/2.0, 1.0/2.0), (35.0/18.0, 1.0), (231.0/100.0, 3.0/2.0)],  # alpha=+2
              [(78.6, 25.2), (790.0, 410.0), (9950.0, 6520.0)],
              [(2.0/3.0, 1.0/6.0), (2.0/3.0, 1.0/3.0), (7.0/9.0, 1.0/2.0)],  # alpha=0
              [(-1, -1), (0.852, 0.375), (0.997, 0.617)],  # -1
              [(-1, -1), (1.079, 0.368), (1.033, 0.607)],  # -2
              [(-1, -1), (-1, -1), (1.053, 0.553)],  # -3
              [(-1, -1), (-1, -1), (1.302, 0.535)],  # alpha=-4
              ]

    # print("table2 = ", table2[row_idx][col_idx])
    return table2[row_idx][col_idx]


def greenhall_table1(alpha, d):
    """ Table 1 from Greenhall 2004 """
    row_idx = int(-alpha+2) # map 2-> row0 and -4-> row6
    col_idx = int(d-1)
    table1 = [[(2.0/3.0, 1.0/3.0), (7.0/9.0, 1.0/2.0), (22.0/25.0, 2.0/3.0)],  # alpha=+2
              [(0.840, 0.345), (0.997, 0.616), (1.141, 0.843)],
              [(1.079, 0.368), (1.033, 0.607), (1.184, 0.848)],
              [(-1, -1), (1.048, 0.534), (1.180, 0.816)],  # -1
              [(-1, -1), (1.302, 0.535), (1.175, 0.777)],  #-2
              [(-1, -1), (-1, -1), (1.194, 0.703)],  #-3
              [(-1, -1), (-1, -1), (1.489, 0.702)],  # alpha=-4
              ]

    # print("table1 = ", table1[row_idx][col_idx])
    return table1[row_idx][col_idx]


def edf_totdev(N, m, alpha):
    """ Equivalent degrees of freedom for Total Deviation
        FIXME: what is the right behavior for alpha outside 0,-1,-2?

        NIST SP1065 page 41, Table 7
    """
    alpha = int(alpha)
    if alpha in [0, -1, -2]:
        # alpha  0 WFM
        # alpha -1 FFM
        # alpha -2 RWFM
        NIST_SP1065_table7 = [(1.50, 0.0), (1.17, 0.22), (0.93, 0.36)]
        (b, c) = NIST_SP1065_table7[int(abs(alpha))]
        return b*(float(N)/float(m))-c
    # alpha outside 0, -1, -2:
    return edf_simple(N, m, alpha)


def edf_mtotdev(N, m, alpha):
    """ Equivalent degrees of freedom for Modified Total Deviation

        NIST SP1065 page 41, Table 8
    """
    assert(alpha in [2, 1, 0, -1, -2])
    NIST_SP1065_table8 = [(1.90, 2.1), (1.20, 1.40), (1.10, 1.2), (0.85, 0.50), (0.75, 0.31)]
    # (b, c) = NIST_SP1065_table8[ abs(alpha-2) ]
    (b, c) = NIST_SP1065_table8[abs(alpha-2)]
    edf = b*(float(N)/float(m))-c
    print("mtotdev b,c= ", (b, c), " edf=", edf)
    return edf


def edf_simple(N, m, alpha):
    """Equivalent degrees of freedom.
    Simple approximate formulae.

    Parameters
    ----------
    N : int
        the number of phase samples
    m : int
        averaging factor, tau = m * tau0
    alpha: int
        exponent of f for the frequency PSD:
        'wp' returns white phase noise.             alpha=+2
        'wf' returns white frequency noise.         alpha= 0
        'fp' returns flicker phase noise.           alpha=+1
        'ff' returns flicker frequency noise.       alpha=-1
        'rf' returns random walk frequency noise.   alpha=-2
        If the input is not recognized, it defaults to idealized, uncorrelated
        noise with (N-1) degrees of freedom.

    Notes
    -----
       S. Stein, Frequency and Time - Their Measurement and
       Characterization. Precision Frequency Control Vol 2, 1985, pp 191-416.
       http://tf.boulder.nist.gov/general/pdf/666.pdf

    Returns
    -------
    edf : float
        Equivalent degrees of freedom

    """

    edf = (N - 1)  # default value

    N = float(N)
    m = float(m)
    if alpha in [2, 1, 0, -1, -2]:
        # NIST SP 1065, Table 5
        if alpha == +2:
            edf = (N + 1) * (N - 2*m) / (2 * (N - m))

        if alpha == 0:
            edf = (((3 * (N - 1) / (2 * m)) - (2 * (N - 2) / N)) *
                   ((4*pow(m, 2)) / ((4*pow(m, 2)) + 5)))

        if alpha == 1:
            a = (N - 1)/(2 * m)
            b = (2 * m + 1) * (N - 1) / 4
            edf = np.exp(np.sqrt(np.log(a) * np.log(b)))

        if alpha == -1:
            if m == 1:
                edf = 2 * (N - 2) /(2.3 * N - 4.9)
            if m >= 2:
                edf = 5 * N**2 / (4 * m * (N + (3 * m)))

        if alpha == -2:
            a = (N - 2) / (m * (N - 3)**2)
            b = (N - 1)**2
            c = 3 * m * (N - 1)
            d = 4 * m **2
            edf = a * (b - c + d)

    else:
        print("Noise type not recognized. Defaulting to N - 1 degrees of freedom.")

    return edf

########################################################################
# end of ci.py
