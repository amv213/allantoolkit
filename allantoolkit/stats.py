import numpy as np


def calc_adev(phase, rate, mj, stride):
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
    mj = int(mj)
    stride = int(stride)
    d2 = phase[2 * mj::stride]
    d1 = phase[1 * mj::stride]
    d0 = phase[::stride]

    n = min(len(d0), len(d1), len(d2))

    if n == 0:
        RuntimeWarning("Data array length is too small: %i" % len(phase))
        n = 1

    v_arr = d2[:n] - 2 * d1[:n] + d0[:n]
    s = np.sum(v_arr * v_arr)

    dev = np.sqrt(s / (2.0 * n)) / mj  * rate
    deverr = dev / np.sqrt(n)

    return dev, deverr, n