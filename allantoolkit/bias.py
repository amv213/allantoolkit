import numpy as np
from . import tables
from . import stats

# shorten type hint to save some space
Array = np.ndarray


def calc_bias_avar(data: Array, m: int, alpha: int) -> float:
    """Calculates bias by which to correct AVAR results for given noise type.

    References:
        [RileyStable32Manual]_ (Table pg. 97)

    Args:
        data:   array for which variance was computed
        m:      averaging factor for which variance was computed
        alpha:  dominant fractional frequency noise type `alpha` at given
                averaging factor

    Returns:
         bias correction by which to multiply computed variance

    """

    return 1


def calc_bias_oavar(data: Array, m: int, alpha: int) -> float:
    """Calculates bias by which to correct OAVAR results for given noise type.

    References:
        [RileyStable32Manual]_ (Table pg. 97)

    Args:
        data:   array for which variance was computed
        m:      averaging factor for which variance was computed
        alpha:  dominant fractional frequency noise type `alpha` at given
                averaging factor

    Returns:
         bias correction by which to multiply computed variance

    """

    return 1


def calc_bias_mvar(data: Array, m: int, alpha: int) -> float:
    """Calculates bias by which to correct MVAR results for given noise type.

    References:
        [RileyStable32Manual]_ (Table pg. 97)

    Args:
        data:   array for which variance was computed
        m:      averaging factor for which variance was computed
        alpha:  dominant fractional frequency noise type `alpha` at given
                averaging factor

    Returns:
         bias correction by which to multiply computed variance

    """

    return 1


def calc_bias_tvar(data: Array, m: int, alpha: int) -> float:
    """Calculates bias by which to correct TVAR results for given noise type.

    References:
        [RileyStable32Manual]_ (Table pg. 97)

    Args:
        data:   array for which variance was computed
        m:      averaging factor for which variance was computed
        alpha:  dominant fractional frequency noise type `alpha` at given
                averaging factor

    Returns:
         bias correction by which to multiply computed variance

    """

    return 1


def calc_bias_hvar(data: Array, m: int, alpha: int) -> float:
    """Calculates bias by which to correct HVAR results for given noise type.

    References:
        [RileyStable32Manual]_ (Table pg. 97)

    Args:
        data:   array for which variance was computed
        m:      averaging factor for which variance was computed
        alpha:  dominant fractional frequency noise type `alpha` at given
                averaging factor

    Returns:
         bias correction by which to multiply computed variance

    """

    return 1


def calc_bias_ohvar(data: Array, m: int, alpha: int) -> float:
    """Calculates bias by which to correct OHVAR results for given noise type.

    References:
        [RileyStable32Manual]_ (Table pg. 97)

    Args:
        data:   array for which variance was computed
        m:      averaging factor for which variance was computed
        alpha:  dominant fractional frequency noise type `alpha` at given
                averaging factor

    Returns:
         bias correction by which to multiply computed variance

    """

    return 1


def calc_bias_totvar(data: Array, m: int, alpha: int) -> float:
    """Calculates bias by which to correct TOTVAR results for given noise type.


    References:

        D.A. Howe, "Total Variance Explained", Proc. 1999 Joint Meeting of the
        European Freq. and Time Forum and the IEEE Freq. Contrl. Symp.,
        pp. 1093-1099, April 1999.
        https://www.nist.gov/publications/total-variance-explained (Table 1)

    Args:
        data:   array for which variance was computed
        m:      averaging factor for which variance was computed
        alpha:  dominant fractional frequency noise type `alpha` at given
                averaging factor

    Returns:
         bias correction by which to multiply computed variance
    """

    a = tables.BIAS_TOTVAR.get(alpha, None)

    if a is None:

        # no bias correction for this noise type
        return 1

    if m/data.size > 0.5:
        print(f"No bias correction at m = {m}")
        # no bias corrections beyond half of dataset size
        return 1

    b = 1 - (a*m/data.size)

    return 1/b


def calc_bias_mtotvar(data: Array, m: int, alpha: int) -> float:
    """Calculates bias by which to correct MTOTVAR results for given noise
    type.

    The TOTMVAR bias factor (the ratio of the expected value of TOTMVAR to
    MVAR) depends on the noise type but is essentially independent of the
    averaging factor and # of data points.

    FIXME: Bias is now disabled because Stable32 doesn't seem to be applying
    it (at least not on averaging factors seen so far). Should we implement it
    and diverge from Stable32?

    References:
        http://www.wriley.com/CI2.pdf
        MTOT and TTOT Bias Function, Table Pg.6)

        D.A. Howe and F. Vernotte, "Generalization of the Total Variance
        Approach to the ModifiedAllan Variance",Proc. 31st PTTI Meeting,
        pp. 267-276, Dec. 1999

    Args:
        data:   array for which variance was computed
        m:      averaging factor for which variance was computed
        alpha:  dominant fractional frequency noise type `alpha` at given
                averaging factor

    Returns:
         bias correction by which to multiply computed variance
    """

    b = tables.BIAS_MTOTVAR.get(alpha, None)

    if b is None:

        # no bias correction for this noise type
        return 1

    return 1  # should return b, but Stable32 doesn't seem to be doing it


def calc_bias_ttotvar(data: Array, m: int, alpha: int) -> float:
    """Calculates bias by which to correct TTOTVAR results for given noise
    type.

    Same as MTOTVAR.

    References:
        http://www.wriley.com/CI2.pdf
        MTOT and TTOT Bias Function, Table Pg.6)

    Args:
        data:   array for which variance was computed
        m:      averaging factor for which variance was computed
        alpha:  dominant fractional frequency noise type `alpha` at given
                averaging factor

    Returns:
         bias correction by which to multiply computed variance
    """

    return calc_bias_mtotvar(data=data, m=m, alpha=alpha)


def calc_bias_htotvar(data: Array, m: int, alpha: int) -> float:
    """Calculates bias by which to correct HTOTVAR results for given noise
    type.

    The TOTMVAR bias factor (the ratio of the expected value of TOTMVAR to
    MVAR) depends on the noise type but is essentially independent of the
    averaging factor and # of data points.

    References:
        D.A. Howe, et. Al., “A Total Estimator of the Hadamard Function Used
        for GPS Operations”, Proc. 32nd PTTI Meeting, pp. 255-268, November
        2000

        http://www.wriley.com/CI2.pdf
        (HTOT Bias Function, Table Pg.6)


    Args:
        data:   array for which variance was computed
        m:      averaging factor for which variance was computed
        alpha:  dominant fractional frequency noise type `alpha` at given
                averaging factor

    Returns:
         bias correction by which to multiply computed variance
    """

    a = tables.BIAS_HTOTVAR.get(alpha, None)

    if a is None:

        # no bias correction for this noise type
        return 1

    # HTOTVAR uses HVAR at m=1, so dispatch to correct bias calculator
    b = 1 + a if m > 1 else calc_bias_hvar(data=data, m=m, alpha=alpha)

    return 1/b


def calc_bias_theo1(data: Array, m: int, alpha: int) -> float:
    """Calculates bias by which to correct THEO1 results for given noise
    type.

    The Thêo1 statistic is an unbiased estimator of the Allan variance for
    white FM noise. For other power law noise types, the following bias
    corrections should be applied to the Thêo1 deviation.

    References:
        Theo1: characterization of very long-term frequency stability
        Howe,D.A. et al.
        18th European Frequency and Time Forum (EFTF 2004)
        2004 (Table 1)

        [RileyStable32Manual]_ (Theo1 Bias, pg.80)


    Args:
        data:   array for which variance was computed
        m:      averaging factor for which variance was computed
        alpha:  dominant fractional frequency noise type `alpha` at given
                averaging factor

    Returns:
         bias correction by which to multiply computed variance
    """

    params = tables.BIAS_THEO1.get(alpha, None)

    if params is None:

        # no bias correction for this noise type
        return 1

    a, b, c = params

    # The `theo1` tau is 0.75*m
    b = a + b/(0.75*m)**c

    # this bias is b = AVAR/THEO1 so THEO1/AVAR = 1/b

    return b


def calc_bias_theobr(x: Array, rate: float) -> float:
    """Calculate dynamic correction factor by which to debias THEO1 variances,
    to get correspoinding THEOBR value. Bias removal via THEOBR does not
    require the prior estimation of a dominant noise type at any single
    averaging time.

    There is one bias factor value for the whole run, expressed as a sum of
    ratios of AVAR / THEO1 over a number of relevant averaging factors.

    References:
        Theo1: characterization of very long-term frequency stability
        Howe,D.A. et al.
        18th European Frequency and Time Forum (EFTF 2004)
        2004

        J.A. Taylor and D.A. Howe, “Fast ThêoBR: A Method for Long Data Set
        Stability  Analysis” (Used by Stable32)

        http://www.wriley.com/Fast%20Bias-Removed%20Theo1%20Calculation%20with%20R.pdf

        [RileyStable32Manual]_ (TheoBR and TheoH, pg.80)


    Args:
        x:      array of phase data for which variance was computed
        rate:   sampling rate of the input data, in Hz.

    Returns:
         bias correction by which to scale computed variance
    """

    # For TheoBR algorithm to work, need to precalculate a support vector of
    # THEO1 values at all taus
    theo1s, _ = stats.calc_theo1_fast(x=x, rate=rate, explode=True)

    # phase array size
    N = x.size

    # Number of AVAR/Theo1 variance ratio averages to use:
    # - Use 6 on the denominator to end up with the bias correction factor used
    #   by Stable32 in the `Run` & `Plot` calculations
    # - Use 30 on the denominator to end up with the bias correction factor
    #   used by Stable32 in the `Sigma` calculations
    n = int(np.floor(N / 6 - 3))

    # Correction factor summation loop
    kf = 0.
    for i in range(n + 1):

        # Calculate ratio of AVAR to THEO1 at each equivalent taus, starting
        # from m=12 for THEO1 == m=9 for AVAR

        # TODO: for some reason replacing this loop with the official stats
        #  avar function doesn't give the same results ... investigate...

        m = 9 + 3 * i
        avar = 0.
        for j in range(N - 2 * m):
            avar += (x[j + 2 * m] - 2 * x[j + m] + x[j]) * (
                        x[j + 2 * m] - 2 * x[j + m] + x[j])
        avar /= (2 * (N - 2 * m) * (m / rate) ** 2)

        kf += (avar / theo1s[12 + 4 * i])

    # Divide kf sum by # ratios
    kf /= (n + 1)

    return kf

