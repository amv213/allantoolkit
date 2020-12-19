import numpy as np
from . import tables

# shorten type hint to save some space
Array = np.ndarray


def calc_bias_totvar(data: Array, m: int, alpha: int) -> float:
    """Corrected reported TOTVAR results accounting for bias.

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
         bias correction by which to scale computed variance

    """

    a = tables.BIAS_TOTVAR.get(alpha, None)

    if a is None:

        # no bias correction
        return 1.

    return 1 - (a*m/data.size)


