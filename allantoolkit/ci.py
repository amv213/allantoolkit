"""
this file is part of allantoolkit, https://github.com/aewallin/allantools

- functions for confidence intervals
- functions for computing equivalent degrees of freedom
"""

import math
import logging
import numpy as np
import scipy.special
import scipy.stats  # used in confidence_intervals()
import scipy.signal  # decimation in lag-1 acf
from . import tables
from . import utils
from typing import Tuple


# Spawn module-level logger
logger = logging.getLogger(__name__)

# Confidence Intervals
ONE_SIGMA_CI = scipy.special.erf(1/np.sqrt(2))

# shorten type hint to save some space
Array = np.ndarray


def get_error_bars(x: Array, m: int, var: float, n: int, alpha: int,
                   dev_type: str, ci: float = ONE_SIGMA_CI):
    """Calculate non-naive variance errors. Equivalent to Stable32.

    References:
        [RileyStable32Manual]_ (Confidence Intervals, pg.89)
        http://www.wriley.com/CI2.pdf
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050061319.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        var:        variance value for which to compute error bars
        n:          number of analysis samples used to compute deviation
        alpha:      dominant power law frequency noise type
        dev_type:   type of deviation for which error bars are being
                    calculated, e.g. `adev`.
        ci:         confidence factor for which to set confidence limits.
                    Defaults to 1-sigma confidence intervals.

    Returns:
        lower and upper bounds for variance.
    """

    # Chi-squared statistics can be applied to calculate single and
    # double-sided confidence intervals at any desired confidence factor.
    # Those calculations, based on a determination of the number of degrees of
    # freedom for the estimated noise type, are the confidence intervals used
    # by the Stable32 variance functions.

    # Dispatch to appropriate edf calculator for given dev_type
    # should have function signature func(x, m, alpha) -> edf
    try:
        edf_func = globals()['calc_edf_' + dev_type]
    except KeyError:
        raise ValueError(f"Error bar calculation for {dev_type} has not "
                         f"been implemented.")

    edf = edf_func(x=x, m=m, alpha=alpha)

    # with the known EDF we get CIs
    (lo, hi) = confidence_interval(var=var, edf=edf, ci=ci)

    return lo, hi


def confidence_interval(var: float, edf: float, ci: float) -> \
        Tuple[float, float]:
    """Returns double-sided statistical limits on the true variance at
    requested confidence factor.

    Calculation based on Chi-square statistics of observed sample variance
    and corresponding equivalent degrees of freedom.

    References:
        http://www.wriley.com/CI2.pdf
        https://faculty.elgin.edu/dkernler/statistics/ch09/9-3.html


    Args:
        var:    sample variance from which to calculate confidence intervals
        edf:    equivalent degrees of freedom
        ci:     confidence factor for which to set confidence limits on the
                variance e.g. `0.6827` would set a 1-sigma confidence interval.

    Returns:
        lower and upper bounds for variance.
    """

    # This will not give the same bounds as Stable32, as Stable32 uses an
    # approximate implementation. If you want to replicate Stable32, swap with
    # chi2_interval_stable32(ci, edf, variant=True)
    chi2_r, chi2_l = scipy.stats.chi2.interval(ci, edf)

    var_lo = edf * var / chi2_l
    var_hi = edf * var / chi2_r

    return var_lo, var_hi


# Stable32 CI functions

def chi2_ppf_stable32(p: float, edf: float, variant: bool) -> float:
    """Calculates Stable32's implementation of the :math:`\\chi^2`
    inverse cumulative distribution function.

    .. warning::
        Stable32 implementation is an approximation of the true :math:`\\chi^2`
        inverse cumulative distribution function. Moreover it's practical
        implementation in software has a bug in the source code.

    .. seealso::
        http://www.anderswallin.net/2020/12/fun-with-chi-squared/ for
        discussion

    .. codeauthor:: Anders Wallin

    Args:
        p:          lower-tail probability for which to calculate inverse cdf
        edf:        number of degrees of freedom
        variant:    if ``False`` implements the Windows binary
                    implementation of the iterative algorithm for small
                    number of degrees of freedom

    Returns:
        approximate :math:`\\chi^2` inverse cumulative function at p.

    References:
        Stable32 source code: CICS3.c

        Abramowitz & Stegun, Handbook of Mathematical Functions, Sections
        26.2.22 & 26.4.17

    """

    # Number of max iterations used by Stable32 for iterative solution
    ITMAX = 100

    # Abramowitz & Stegun approximation for edf>100
    if edf > 100:

        # A&S 26.2.22:
        # approximate inverse Normal cumulative distribution for 0 < p < 0.5

        p1 = min(p, 1 - p)  # constrain to lower tail

        a0, a1, b1, b2 = tables.ABRAMOWITZ_COEFFICIENTS.values()
        a1 = 0.27601  # replace with the typo in the Stable32 source code!

        t = np.sqrt(-2 * np.log(p1))
        xp = t - (a0 + a1*t) / (1 + b1*t + b2*t**2)

        # Greenhall revision depending on which interval p really was on
        if p == 0.5:
            xp = 0
        elif (p - 0.5) < 0:
            xp *= -1

        # A&S 26.4.17:
        # approximate inverse chi-squared distribution for large edf

        chi2 = edf * (1 - (2/(9*edf)) + xp*np.sqrt(2/(9*edf)))**3

        return chi2

    # Iterative solution for edf <= 100
    # see e.g. https://daviddeley.com/random/code.htm
    # Press et al., Numerical Recipes
    else:
        p = 1 - p

        x = edf + (0.5 - p) * edf * 0.5  # start value for chi squared

        if not variant:  # use Windows binary implementation
            prob = CalcChiSqrProb(x, edf)

        else:  # use newer function from numerical recipes
            prob = gammp(edf / 2, float(x / 2), itmax=ITMAX)

        div = 0.1
        while abs((prob - p) / p) > 0.0001:  # accuracy criterion

            sign = 1 if (prob - p > 0) else -1  # save sign of error

            x += edf * (prob - p) / div  # iteration increment

            if x > 0:  # make sure argument is positive

                if not variant:  # use Windows binary implementation
                    prob = CalcChiSqrProb(x, edf)

                else:  # use newer function from numerical recipes
                    prob = gammp(edf / 2, (x / 2), itmax=ITMAX)

                prob = 1 - prob

            else:  # otherwise restore it & reduce increment
                x -= edf * (prob - p) / div
                div *= 2

            if ((prob - p) / sign) < 0:  # did sign of error reverse?
                div *= 2  # reduce increment if it did

        return x


def chi2_interval_stable32(ci: float, edf: float, variant: bool = False) -> \
        Tuple[float, float]:
    """Calculates double-sided confidence intervals with equal areas around
    the median for the Stable32's implementation of the :math:`\\chi^2`
    inverse cumulative distribution function.

    Args:
        ci:                 confidence factor for which to set confidence
                            limits e.g. `0.6827` would set a 1-sigma
                            confidence interval.
        edf:                degrees of freedom
        variant (optional): if ``False`` implements the Windows binary
                            implementation of the iterative algorithm for small
                            number of degrees of freedom

    Returns:
        Stable32 end-points of the range that contain ``100 * ci`` of its
        approximate :math:`\\chi^2` distribution possible values
    """

    alpha = (1 - ci) / 2

    chi2_r = chi2_ppf_stable32(alpha, edf, variant=variant)
    chi2_l = chi2_ppf_stable32(1-alpha, edf, variant=variant)

    return chi2_r, chi2_l


# Dispatchers


def calc_edf_adev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for ADEV

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

        http://www.wriley.com/CI2.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    # TODO: get d, overlapping, and modified from a table

    return cedf(alpha=alpha, d=2, m=m, N=int(x.size), overlapping=False,
                modified=False)


def calc_edf_oadev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for OADEV

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

        http://www.wriley.com/CI2.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    # TODO: get d, overlapping, and modified from a table

    return cedf(alpha=alpha, d=2, m=m, N=int(x.size), overlapping=True,
                modified=False)


def calc_edf_mdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for MDEV.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

        http://www.wriley.com/CI2.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    # TODO: get d, overlapping, and modified from a table

    return cedf(alpha=alpha, d=2, m=m, N=int(x.size), overlapping=True,
                modified=True)


def calc_edf_tdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for TDEV.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

        http://www.wriley.com/CI2.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    # TODO: get d, overlapping, and modified from a table

    return cedf(alpha=alpha, d=2, m=m, N=int(x.size), overlapping=True,
                modified=True)


def calc_edf_hdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for HDEV.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

        http://www.wriley.com/CI2.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    # TODO: get d, overlapping, and modified from a table

    return cedf(alpha=alpha, d=3, m=m, N=int(x.size), overlapping=False,
                modified=False)


def calc_edf_ohdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for OHDEV.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

        http://www.wriley.com/CI2.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    # TODO: get d, overlapping, and modified from a table

    return cedf(alpha=alpha, d=3, m=m, N=int(x.size), overlapping=True,
                modified=False)


def calc_edf_totdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for TOTDEV.

    References:
        http://www.wriley.com/CI2.pdf (TOTVAR and TTOT EDF)

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    try:
        b, c = tables.TOTVAR_EDF_COEFFICIENTS[alpha]

    except KeyError:
        logger.warning(f"Noise type alpha = {alpha} not supported for "
                       f"(T)TOTDEV edf calculation. Falling back to CEDF.")

        # TODO: check this fallback is correct
        return edf_simple(x=x, m=m, alpha=alpha)

    edf = b*(x.size/m) - c

    return edf


def calc_edf_ttotdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for TTOTDEV.

    References:
        http://www.wriley.com/CI2.pdf (TOTVAR and TTOT EDF)

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    return calc_edf_totdev(x=x, m=m, alpha=alpha)


def calc_edf_mtotdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for MTOTDEV.

    References:
        http://www.wriley.com/CI2.pdf (MTOT EDF)

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    try:

        b, c = tables.MTOTVAR_EDF_COEFFICIENTS[alpha]

    except KeyError:
        raise ValueError(f"Noise type alpha = {alpha} not supported for "
                         f"MTOT edf calculation.")

    edf = b*(x.size/m) - c

    return edf


def calc_edf_htotdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for HTOTDEV.

    References:
        D.A. Howe, et. Al., “A Total Estimator of the Hadamard Function Used
        for GPS Operations”, Proc. 32nd PTTI Meeting, pp. 255-268, November
        2000

        http://www.wriley.com/CI2.pdf (HTOT EDF)

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    if m < 16:  # see Stable32 Help Manual and paper

        # Use plain Hadamard estimator
        return calc_edf_hdev(x=x, m=m, alpha=alpha)

    try:
        b0, b1 = tables.HTOTVAR_EDF_COEFFICIENTS[alpha]

    except KeyError:
        raise ValueError(f"Noise type alpha = {alpha} not supported for "
                         f"HTOTDEV edf calculation.")

    edf = (x.size/m) / (b0 + b1*m/x.size)

    return edf


def calc_edf_theo1(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for THEO1.

    References:

        D.A. Howe and T.K. Peppler, “Estimation of Very Long-Term Frequency
        Stability Using a Special-Purpose Statistic”, ", Proc. 2003 Joint
        Meeting of the European Freq. and Time Forum and the IEEE
        International Freq. Contrl. Symp., May 2003


        http://www.wriley.com/CI2.pdf (THEO1 EDF)

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    N = x.size
    t = 0.75*m  # effective theo1 averaging factor

    if alpha == 2:

        edf = ((0.86 * (N+1) * (N - 4*t/3)) / (N - t)) * (t / (t + 1.14))

    elif alpha == 1:

        edf = ((4.798*N**2 - 6.374*N*t + 12.387*t) / (np.sqrt(t+36.6) * (
                N-t))) * (t / (t + 0.3))

    elif alpha == 0:

        edf = ((4.1*N+0.8)/t - (3.1*N+6.5)/N) * (t**1.5 / (t**1.5 + 5.2))

    elif alpha == -1:

        edf = ((2*N**2 - 1.3*N*t - 3.5*t) / (N*t)) * (t**3 / (t**3 + 2.3))

    elif alpha == -2:

        edf = ((4.4*N-2) / (2.9*t)) * (
                ((4.4*N-1)**2 - 8.6*t*(4.4*N-1) + 11.4*t**2) / (4.4*N-3)**2)

    else:
        raise ValueError(f"Noise type alpha = {alpha} not supported for "
                         f"Theo1 edf calculation.")

    return edf


# Combined Greenhall EDF algorithm

def cedf(alpha: int, d: int, m: int, N: int, overlapping: bool = False,
         modified: bool = False, version: str = 'full') -> float:
    """ Returns combined edf value, as per Greenhall & Riley PTTI '03.

    This method is based on modeling the phase as the first difference of a
    continuous-time pure power-law process.

    References:

        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

    Args:
        alpha:          frequency power law noise exponent.
        d:              order of phase difference of the variance for which
                        to calculate edf.
        m:              averaging factor at which to calculate edf.
        modified:       `True` if calculating edf for a `modified` variance.
                        `False` otherwise.
        overlapping:    `True` if calculating edf for an `overlapping`
                        variance. `False` otherwise.
        N:              number of phase data points, with sample period tau_0.
        version:        `simple` to only calculate one step of the algorithm.
                        Defaults to `full`.

    Returns:
        equivalent degrees of freedom of the variance estimator at given
        averaging factor.

    Notes:
        Used by Stable32 for AVAR, MVAR, TVAR, HVAR, and their overlapping
        versions. Output edf result matches Stable32 `Sigma` tool (need to
        tick the ACF noise id box)

    """

    # Consistency check on input params
    if alpha + 2*d < 1:
        raise ValueError(f"A d = {d} deviation cannot discriminate noise types"
                         f" alpha = {alpha}. Update input parameters.")

    # Set filter factor
    F = 1 if modified else m

    # Set stride factor (estimator stride = tau / S)
    S = m if overlapping else 1

    # Set an integer constant
    J_max = 100  # suggested value

    # 3.4. INITIAL STEPS

    # 1. compute M, the number of summands in the estimator

    L = m//F + m*d  # length of filter applied to the phase samples

    if N >= L:

        M = 1 + int(np.floor(S*(N-L)/m))

    else:
        raise ValueError(f"Not enough data to calculate edf. Should have N "
                         f">= {L}, but only have {N} samples.")

    # 2. Let
    J = int(min(M, (d+1)*S))

    # 3.5 Main procedure, simplified version
    if version == 'simple':

        inv_edf = cedf_basic_sum(J=J, M=M, S=S, F=F, alpha=alpha, d=d) / \
                      (M * cedf_sz(t=0, F=F, alpha=alpha, d=d)**2)

        return 1 / inv_edf

    # 3.6 Main Procedure, full version

    r = M//S

    # There are four cases. The calculations use coefficients from three
    # numerical tables:

    #  3.6.1 Case 1. Modified variances: F = 1, all alpha.
    #  This case also applies to unmodified variances when F = m = 1

    if F == 1:

        if J <= J_max:

            inv_edf = cedf_basic_sum(J=J, M=M, S=S, F=1, alpha=alpha, d=d) / \
                      (M * cedf_sz(t=0, F=1, alpha=alpha, d=d)**2)

        elif J > J_max and r >= d+1:

            a0, a1 = tables.GREENHALL_TABLE1[alpha][d]

            inv_edf = (a0 - a1/r) / r

        else:

            m_prime = J_max / r  # not necessarily an integer

            inv_edf = cedf_basic_sum(
                J=J_max, M=J_max, S=m_prime, F=1, alpha=alpha, d=d
            ) / (J_max * cedf_sz(t=0, F=1, alpha=alpha, d=d)**2)

    #  3.6.2 Case 2. Unmodified variances, WHFM to RRFM: F = m, alpha <= 0

    elif F == m and alpha <= 0:

        if J <= J_max:

            m_prime = m if m*(d+1) <= J_max else np.inf

            inv_edf = cedf_basic_sum(
                J=J, M=M, S=S, F=m_prime, alpha=alpha, d=d
            ) / (M * cedf_sz(t=0, F=m_prime, alpha=alpha, d=d)**2)

        elif J > J_max and r >= d+1:

            a0, a1 = tables.GREENHALL_TABLE2[alpha][d]

            inv_edf = (a0 - a1/r) / r

        else:

            m_prime = J_max / r  # not necessarily an integer

            inv_edf = cedf_basic_sum(
                J=J_max, M=J_max, S=m_prime, F=np.inf, alpha=alpha, d=d
            ) / (J_max * cedf_sz(t=0, F=np.inf, alpha=alpha, d=d)**2)

    #  3.6.3 Case 3. Unmodified variances, FLPM: F = m, alpha = 1

    elif F == m and alpha == 1:

        if J <= J_max:

            inv_edf = cedf_basic_sum(J=J, M=M, S=S, F=m, alpha=1, d=d) / \
                      (M * cedf_sz(t=0, F=m, alpha=1, d=d)**2)

            if m > 10**6:
                logger.warning("CEDF calculation at this averaging time "
                               "might have a roundoff error.")

        elif J > J_max and r >= d+1:

            a0, a1 = tables.GREENHALL_TABLE2[alpha][d]
            b0, b1 = tables.GREENHALL_TABLE3[alpha][d]

            inv_edf = (a0 - a1/r) / (r * (b0 + b1*np.log(m))**2)

        else:

            m_prime = J_max / r  # not necessarily an integer

            b0, b1 = tables.GREENHALL_TABLE3[alpha][d]

            inv_edf = cedf_basic_sum(
                J=J_max, M=J_max, S=m_prime, F=m_prime, alpha=1, d=d
            ) / (J_max * (b0 + b1*np.log(m))**2)

    #  3.6.4 Case 4. Unmodified variances, WHPM: F = m, alpha = 2

    elif F == m and alpha == 2:

        K = int(np.ceil(r))

        if K <= d:

            s = 0.
            for k in range(1, K):
                s += (1 - k/r) * utils.binom(n=2*d, k=d-k)**2

            s *= (2 / utils.binom(n=2*d, k=d)**2)

            s += 1

            inv_edf = s / M

        else:

            a0, a1 = tables.GREENHALL_TABLE2[alpha][d]

            inv_edf = (a0 - a1/r) / M

    else:

        raise ValueError("Could not calculate equivalent degrees of freedom "
                         "for given combination of input parameters... "
                         "something went wrong!")

    return 1 / inv_edf


def cedf_sw(t: float, alpha: int) -> float:
    """ Calculate the generalised autocovariance (GACV) of the pure
    continuous time power-law process w(t) with spectral density exponent alpha

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003 (Eq.7)

    Args:
        t:      time at which to compute the metric
        alpha:  frequency power law noise exponent.

    Returns:
        generalised autocovariance at given time for given noise type
    """

    t = np.abs(t)

    if alpha == 2:
        sw = -t

    elif alpha == 1:
        sw = np.log(t) * t**2 if t != 0. else 0

    elif alpha == 0:
        sw = t**3

    elif alpha == -1:
        sw = -np.log(t) * t**4 if t != 0. else 0

    elif alpha == -2:
        sw = -(t**5)

    elif alpha == -3:
        sw = np.log(t) * t**6

    elif alpha == -4:
        sw = t**7

    else:
        raise ValueError(f"Noise type alpha = {alpha} not recognised.")

    return sw


def cedf_sx(t: float, F: float, alpha: int) -> float:
    """ Calculate the phase noise at time t for given dominant noise type
    alpha.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003 (Eq.8)

    Args:
        t:      time at which to compute the metric
        F:      filter factor
        alpha:  frequency power law noise exponent.

    Returns:
        phase noise at given time for given noise type
    """

    if np.isinf(F) and (-4 <= alpha <= 0):

        sx = cedf_sw(t=t, alpha=alpha+2)

    else:

        a = 2*cedf_sw(t=t, alpha=alpha)
        b = cedf_sw(t=t-1/F, alpha=alpha)
        c = cedf_sw(t=t+1/F, alpha=alpha)

        sx = (a-b-c) * F**2

    return sx


def cedf_sz(t: float, F: float, alpha: int, d: int) -> float:
    """ Calculate the auto-covariance (ACV) sz at time t.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003 (Eq.9)

    Args:
        t:      time at which to compute the metric
        F:      filter factor
        alpha:  frequency power law noise exponent.
        d:      order of phase difference of the variance

    Returns:
        auto-covariance at given time for given noise type
    """

    if d == 1:

        a = cedf_sx(t=t, F=F, alpha=alpha)
        b = cedf_sx(t=t-1, F=F, alpha=alpha)
        c = cedf_sx(t=t+1, F=F, alpha=alpha)

        sz = 2*a - b - c

    elif d == 2:

        a = cedf_sx(t=t, F=F, alpha=alpha)
        b = cedf_sx(t=t-1, F=F, alpha=alpha)
        c = cedf_sx(t=t+1, F=F, alpha=alpha)
        d_ = cedf_sx(t=t-2, F=F, alpha=alpha)
        e = cedf_sx(t=t+2, F=F, alpha=alpha)

        sz = 6*a - 4*b - 4*c + d_ + e

    elif d == 3:

        a = cedf_sx(t=t, F=F, alpha=alpha)
        b = cedf_sx(t=t-1, F=F, alpha=alpha)
        c = cedf_sx(t=t+1, F=F, alpha=alpha)
        d_ = cedf_sx(t=t-2, F=F, alpha=alpha)
        e = cedf_sx(t=t+2, F=F, alpha=alpha)
        f = cedf_sx(t=t-3, F=F, alpha=alpha)
        g = cedf_sx(t=t+3, F=F, alpha=alpha)

        sz = 20*a - 15*b - 15*c + 6*d_ + 6*e - f - g

    else:

        raise ValueError(f"Phase differencing order d = {d} not recognised. "
                         f"Accepted values should be in [1, 2, 3].")

    return sz


def cedf_basic_sum(J: float, M: float, S: float, F: float, alpha: int, d: int):
    """ Calculate the BasicSum factor to obtain the edf for the
    calculated variance estimator under the given dominant noise type alpha.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003 (Eq.10)

    Args:
        J:      iteration number?
        M:      number of summands in the estimator
        S:      stride factor
        F:      filter factor
        alpha:  frequency power law noise exponent.
        d:      order of phase difference of the variance

    Returns:
        value of the BasicSum, for edf calculation
    """

    a = cedf_sz(t=0, F=F, alpha=alpha, d=d)**2

    b = (1 - J/M) * cedf_sz(t=J/S, F=F, alpha=alpha, d=d)**2

    c = 0.
    for j in range(1, int(J)):
        c += (1 - j/M) * cedf_sz(t=j/S, F=F, alpha=alpha, d=d)**2
    c *= 2

    basic_sum = a + b + c

    return basic_sum


# TODO: check and see if should use this when other edf functions don't
#  support the given noise type
def edf_simple(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    using a simple approximate formula (before CEDF algorithm).

    If the input the dominant noise type is not supported by the algorithm,
    it defaults to idealized, uncorrelated noise with (N-1) degrees of freedom.

    References:

        S. Stein, Frequency and Time - Their Measurement and
        Characterization. Precision Frequency Control Vol 2, 1985, pp 191-416.
        http://tf.boulder.nist.gov/general/pdf/666.pdf

        NIST SP 1065, Table 5

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    N = x.size

    if alpha == +2:

        edf = (N + 1) * (N - 2*m) / (2 * (N - m))

    elif alpha == 0:

        edf = (((3 * (N - 1) / (2 * m)) - (2 * (N - 2) / N)) *
               ((4*pow(m, 2)) / ((4*pow(m, 2)) + 5)))

    elif alpha == 1:

        a = (N - 1)/(2 * m)
        b = (2 * m + 1) * (N - 1) / 4
        edf = np.exp(np.sqrt(np.log(a) * np.log(b)))

    elif alpha == -1:

        if m == 1:

            edf = 2 * (N - 2) /(2.3 * N - 4.9)

        else:

            edf = 5 * N**2 / (4 * m * (N + (3 * m)))

    elif alpha == -2:

        a = (N - 2) / (m * (N - 3)**2)
        b = (N - 1)**2
        c = 3 * m * (N - 1)
        d = 4 * m**2
        edf = a * (b - c + d)

    else:
        logger.info("Noise type not recognized. Defaulting to N - 1 degrees "
                    "of freedom.")
        edf = (N - 1)

    return edf


########################################################################
# Numerical recipes in python
# from https://github.com/mauriceling/dose/blob/master/dose/copads/nrpy.py
#
# Copyright (c) Maurice H.T. Ling <mauriceling@acm.org>
# Date created: 19th March 2008
# License: Unless specified otherwise, all parts of this package, except
# those adapted, are covered under Python Software Foundation License
# version 2.

def gammp(a: float, x: float, itmax: int) -> float:
    """Calculates the lower incomplete Gamma function :math:`\\gamma (a, x)` .

    The incomplete gamma function is defined as:

    .. math::

        \\gamma(a,x) = {1 \\over \\Gamma(a) }
        \\int_0^x t^{a-1} e^{-t} \\mathrm{dt}

    where :math:`\\Gamma(a)` is the ordinary (complete) gamma function.

    Args:
        a:      gamma function parameter
        x:      gamma function integral upper limit
        itmax:  maximum number of iterations in algorithm approximation

    Returns:
        value of the lower incomplete gamma function :math:`\\gamma (a, x)`

    References:
        Press, William H., Flannery, Brian P., Teukolsky, Saul A., and
        Vetterling, William T. 1989. Numerical Recipes in Pascal. Cambridge
        University Press, Cambridge (ISBN 978-0521375160) (Chapter 6 section 2)

        Ling, MHT. 2009. Compendium of Distributions, I: Beta, Binomial, Chi-
        Square, F, Gamma, Geometric, Poisson, Student's t, and Uniform. The
        Python Papers Source Codes 1:4
    """

    if a <= 0 or x < 0:
        raise ValueError('Bad value for a or x: %s, %s' % (a, x))

    if x < a + 1:  # use series approximation
        out = gser(a, x, itmax=itmax)[0]

    else:  # use continued fraction approximation
        out = 1.0 - gcf(a, x, itmax=itmax)[0]

    return out


def gser(a: float, x: float, itmax: int = 700, eps: float = 3.e-7) -> Tuple[
    float, float]:
    """Calculates the series approximation to the incomplete gamma function
    :math:`\\gamma (a, x)` .

    Args:
        a:                  gamma function parameter
        x:                  gamma function integral upper limit
        itmax (optional):   maximum number of iterations in algorithm
                            approximation
        eps (optional):     tolerance for convergence

    Returns:
        tuple with value of the series approximation to the incomplete gamma
        function :math:`\\gamma (a, x)`, and value of the complete gamma
        function :math:`\\Gamma (a)`

    References:
        http://mail.python.org/pipermail/python-list/2000-June/671838.html

        Ling, MHT. 2009. Compendium of Distributions, I: Beta, Binomial, Chi-
        Square, F, Gamma, Geometric, Poisson, Student's t, and Uniform. The
        Python Papers Source Codes 1:4
    """

    gln = gammln(a)  # complete gamma function

    if x < 0:
        raise ValueError('Bad value for x: %s' % a)

    if x == 0:
        return 0, 0

    ap = a
    total = 1.0 / a
    delta = total
    n = 1

    while n <= itmax:
        ap = ap + 1.0
        delta = delta * x / ap
        total = total + delta

        if abs(delta) < abs(total) * eps:
            return total * math.exp(-x + a * math.log(x) - gln), gln

        n = n + 1

    raise RuntimeError('Maximum iterations reached: %s, %s' % (
        abs(delta), abs(total) * eps))


def gcf(a: float, x: float, itmax: int = 200, eps: float = 3.e-7) -> Tuple[
    float, float]:
    """Calculates the continued fraction approximation of the incomplete gamma
    function :math:`\\gamma (a, x)` .

    Args:
        a:                  gamma function parameter
        x:                  gamma function integral upper limit
        itmax (optional):   maximum number of iterations in algorithm
                            approximation
        eps (optional):     tolerance for convergence

    Returns:
        tuple with value of the  continued fraction approximation to the
        incomplete gamma function :math:`\\gamma (a, x)`, and value of the
        complete gamma function :math:`\\Gamma (a)`

    References:
        http://mail.python.org/pipermail/python-list/2000-June/671838.html

        Ling, MHT. 2009. Compendium of Distributions, I: Beta, Binomial, Chi-
        Square, F, Gamma, Geometric, Poisson, Student's t, and Uniform. The
        Python Papers Source Codes 1:4
    """

    gln = gammln(a)  # complete gamma function

    gold = 0.0
    a0 = 1.0
    a1 = x
    b0 = 0.0
    b1 = 1.0
    fac = 1.0
    n = 1
    while n <= itmax:
        an = n
        ana = an - a
        a0 = (a1 + a0 * ana) * fac
        b0 = (b1 + b0 * ana) * fac
        anf = an * fac
        a1 = x * a0 + anf * a1
        b1 = x * b0 + anf * b1

        if a1 != 0.0:
            fac = 1.0 / a1
            g = b1 * fac

            if abs((g - gold) / g) < eps:
                return g * math.exp(-x + a * math.log(x) - gln), gln

            gold = g

        n = n + 1

    raise RuntimeError('Maximum iterations reached: %s' % abs((g - gold) / g))


def gammln(a: float) -> float:
    """Calculates the complete Gamma function :math:`\\Gamma (a)` .

    Args:
        a:  gamma function parameter

    Returns:
        value of the complete gamma function :math:`\\Gamma (a)`

    References:
        Press, William H., Flannery, Brian P., Teukolsky, Saul A., and
        Vetterling, William T. 1989. Numerical Recipes in Pascal. Cambridge
        University Press, Cambridge (ISBN 978-0521375160) (Chapter 6 section 1)

        http://mail.python.org/pipermail/python-list/2000-June/671838.html

        Ling, MHT. 2009. Compendium of Distributions, I: Beta, Binomial, Chi-
        Square, F, Gamma, Geometric, Poisson, Student's t, and Uniform. The
        Python Papers Source Codes 1:4
    """

    gammln_cof = [76.18009173, -86.50532033, 24.01409822,
                  -1.231739516e0, 0.120858003e-2, -0.536382e-5]
    x = a - 1.0
    tmp = x + 5.5
    tmp = (x + 0.5) * math.log(tmp) - tmp
    ser = 1.0
    for j in range(6):
        x = x + 1.
        ser = ser + gammln_cof[j] / x

    return tmp + math.log(2.50662827465 * ser)


########################################################################
# Alternative implementation of Chi-Squared function in Stable32
# This is possibly used in the Window$ executable (?)
#
# Python code AW2020-12-28

def CalcNormalProb(x: float) -> float:
    """Calculates the cumulative normal distribution at x.

    Args:
        x:  input parameter

    Returns:
        cumulative normal probability at x

    References:
       Collected Algorithms from CACM, Vol. I, #209,
       D. Ibbetson and E. Brothers, 1963.

       Stable32 source code: CNP.C
    """

    if x == 0:
        z = 0

    else:

        y = abs(x) / 2

        if y >= 3:
            z = 1

        else:

            if y < 1:

                w = y * y
                z = ((((((((0.000124818987 * w
                            - .001075204047) * w + .005198775019) * w
                          - .019198292004) * w + .059054035642) * w
                        - .151968751364) * w + .319152932694) * w
                      - .531923007300) * w + .797884560593) * y * 2.0

            else:
                y = y - 2
                z = (((((((((((((-.000045255659 * y
                                 + .000152529290) * y - .000019538132) * y
                               - .000676904986) * y + .001390604284) * y
                             - .000794620820) * y - .002034254874) * y
                           + .006549791214) * y - .010557625006) * y
                         + .011630447319) * y - .009279453341) * y
                       + .005353579108) * y - .002141268741) * y
                     + .000535310849) * y + .999936657524

    if x > 0:
        return (z + 1) / 2

    else:
        return (1 - z) / 2


def CalcChiSqrProb(x: float, edf: float) -> float:
    """Calculates alternative implementation of the :math:`\\chi^2` function
    used in Stable32. This is possibly used in the Window$ executable.

    Args:
        x:      value at which to calculate chi-squared probability
        edf:    number of degrees of freedom

    Returns:
        Stable32 window$ binary approximate :math:`\\chi^2` probability at x.

    References:
        Collected Algorithms from CACM, Vol. I, #299,
        I.D. Hill and M.C. Pike, 1965.

        Stable32 source code: CNP.C
    """

    # Check validity of input parameters
    if (x < 0) or (edf < 1):
        raise ValueError("Input parameter values out of bounds for Stable32 "
                         "windows binary implementation of chi-squared "
                         "probability function.")

    # avoid O/F of exp(-0.5*x) < DBL_MIN / 2.225e-308
    bigx = True if x > 1416 else False

    f = int(edf)
    even = True if f % 2 == 0 else False

    a = .5 * x

    if even or f > 2 and (not bigx):
        y = np.exp(-a)

    if even:
        s = y
    else:  # cumulative normal distribution
        s = 2.0 * CalcNormalProb(-np.sqrt(x))

    if f > 2:

        x = (0.5 * (edf - 1.0))

        z = 1 if even else 0.5

        if bigx:

            e = 0 if even else .572364942925

            c = np.log(a)

            while z <= x:
                e = np.log(z) + e
                s = np.exp(c * z - a - e) + s
                z += 1

            chiprob = s

        else:

            e = 1 if even else .564189583548 / np.sqrt(a)
            c = 0

            while z <= x:
                e = e * a / z
                c += e
                z += 1

        chiprob = (c * y + s)

    else:
        chiprob = s

    return 1.0 - chiprob
