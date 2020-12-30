"""
 Useful collection of functions for the allantoolkit test-suite

"""

import time
import numpy
import logging
import decimal
import numpy as np
from pathlib import Path
from typing import Union, Callable
from . import utils

# Spawn module-level logger
logger = logging.getLogger(__name__)

# shorten type hint to save some space
Array = np.ndarray

# Set intuitive rounding behaviour for Decimal
decimal.getcontext().rounding = decimal.ROUND_HALF_UP


# read a result-file, produced by copy/paste from Stable32
# note: header-lines need to be manually commented-out with "#"
def read_resultfile(filename):
    rows = []
    with open(filename) as f:
        for line in f:
            if not line.startswith("#"): # skip comments
                row = []
                l2 = line.split(" ")
                l2 = [_f for _f in l2 if _f]
                for n in range(len(l2)):
                    row.append(float(l2[n]))
                rows.append(row)
    return rows


# parse numbers from a Stable32 result-file
# the columns are:
# AF          Tau        #     Alpha  Min Sigma     Mod Totdev      Max Sigma
# AF = m, averaging factor i.e. tau=m*tau0
# # = n, number of pairs in the dev calculation
# alpha = noise PSD coefficient
def read_stable32(fn: Union[str, Path], file_type: str = None) -> Array:

    # Number of lines to skip in the file
    skip_header = 10 if file_type is None else 9  # for TIE-like files

    data = numpy.genfromtxt(fn, skip_header=skip_header, comments='#')

    logger.info("Read %i entries from %s", data.size, fn)

    return data

# test one tau-value (i.e. one row in the result file) at a time
# test a deviation function by:
# - running the function on the datafile
# - reading the correct answers from the resultfile
# - checking that tau, n, and dev are correct
def test_row_by_row(function: Callable,
                    datafile: Union[str, Path],
                    datarate: float,
                    resultfile: Union[str, Path],
                    verbose: bool = False,
                    tolerance: float = 1e-4,
                    frequency: bool = False,
                    normalize: bool = False):

    # if Stable32 results were given with more digits we could decrease tolerance

    input = utils.read_datafile(datafile)

    logger.info("Read %i entries from %s", len(input), datafile)

    expected_output = read_stable32(resultfile)

    print("test of function ", function)
    if verbose:
        print("Tau N  \t DEV(Stable32) \t DEV(allantoolkit) \t rel.error\t bias")

    # Unpack resultfile columns
    if expected_output.shape[1] == 7:
        # Typical Stable32 Result structure
        ms, taus, ns, alphas, dev_mins, devs, dev_maxs = expected_output.T
    elif expected_output.shape[1] == 4:
        # the MTIE/TIErms results are formatted slightly differently
        ms, taus, ns, devs = expected_output.T

    else:
        raise ValueError("Stable32 Result File format not recognised")

    n_errors = 0
    # run allantoolkit algorithm, row by row
    for i, _ in enumerate(expected_output):

        # If checking a theo1, the file will have an effective 75% of the
        # original one
        tau_ori = taus[i] if function.__name__ != 'theo1' else taus[i] / 0.75

        if frequency:
            (taus2, devs2, errs_lo2, errs_hi2, ns2) = function(
                input, rate=datarate, data_type="freq", taus=tau_ori)
        else:
            (taus2, devs2, errs_lo2, errs_hi2, ns2) = function(
                input, rate=datarate, taus=tau_ori)

        n_errors += check_equal(ns[i], ns2[0])

        n_errors += check_equal(taus[i], taus2[0])

        n_errors += check_approx_equal(devs[i], devs2[0],
                                       tolerance=tolerance, verbose=verbose)
        if verbose:
            rel_error = (devs2[0] - devs[i]) / devs[i]
            bias = pow(devs[i]/devs2[0], 2)
            print("%.1f %d %0.6g \t %0.6g \t %0.6f \t %0.4f OK!" % (
                taus[i], ns[i], devs[i], devs2[0], rel_error, bias))


def check_equal(a,b):
    try:
        assert ( a == b )
        return 0
    except:
        print("ERROR a=", a, " b=", b)
        assert(0)
        return 1

def check_approx_equal(a1,a2, tolerance=1e-4, verbose=False):
    # check the DEV result, with a given relative error tolerance
    rel_error = (a2 - a1) / a1
    bias = pow(a2/a1,2)
    # tolerance = 1e-4 # if Stable32 results were given with more digits we could decrease tol
    try:
        assert ( abs(rel_error) < tolerance )

        return 0
    except:
        print("ERROR %0.6g \t %0.6g \t rel_err = %0.6f \t %0.4f" % ( a1, a2, rel_error, bias))
        assert(0)
        return 1


def print_elapsed(label, start, start0=None):

    end = time.clock()
    start0 = start0 if start0 is not None else start

    logger.info("%s test done in %.2f s, elapsed= %.2f min",
                label, end-start, (end-start0)/60)

    return time.clock()


def S32format(data: Array) -> Array:
    """Format an array of numbers in the same way as Stable32: numbers are
    rounded - half up - to the 4th digit of their scientific notation.

    Args:
        data:  input array to format

    Returns:
        input array rounded to the correct number of digits.

    References:
        https://realpython.com/python-rounding/
    """

    exponent = np.floor(np.log10(data))
    scaled = data / (10 ** exponent)

    # Round number in the intuitive way avoiding machine precision issues
    rounded = np.array([
        float(decimal.Decimal(str(n)).quantize(decimal.Decimal("0.0001")))
        for n in scaled])

    # Convert back to original decimal places
    ori = rounded * 10 ** exponent

    # Avoid machine precision errors by reformatting one last time
    return np.array([float(f"{n:.4e}") for n in ori])


def test_Stable32_run(data: Array, func: Callable, rate: float, data_type: str,
                      taus: Union[str, Array], fn: Union[str, Path],
                      test_alpha: bool, test_ci: bool):
    """Tests that allantoolkit dev results match exactly refrence `Run`
    results from Stable 32.

    Args:
        data:       input array of phase or fractional frequency data on which
                    to calculate statistics
        func:       allantoolkit function implementing the statistic to compute
        rate:       sampling rate of the input data, in Hz.
        data_type:  input data type. Either `phase` or `freq`.
        taus:       array of averaging times for which to compute deviation.
                    Can also be one of the keywords: `all`, `octave`, `decade`.
        fn:         path to the Stable32 file holding reference expected
                    results
        test_alpha: if `True` also checks validity of estimated noise type
        test_ci:    if `True` also checks validity of  calculated confidence
                    intervals
    """

    # Check if testing against deviations not available to frequency data
    if func.__name__ in ['theo1', 'tierms', 'mtie'] and data_type == 'freq':
        return 1

    # Check if we are reading a file for a TIE-like deviation from Stable32 (
    # they have a different number of columns from standard)
    file_type = 'tie' if 'tie' in func.__name__ else None

    # Read in Stable32 Run results
    expected_results = read_stable32(fn, file_type=file_type)

    # Calculate equivalent results with allantoolkit
    actual_results = func(data=data, rate=rate, data_type=data_type, taus=taus)

    # Unpack and format integers
    if file_type != 'tie':
        afs, taus, ns, alphas, devs_lo, devs, devs_hi = expected_results.T
        alphas = alphas.astype(int)
    else:
        afs, taus, ns, devs = expected_results.T
        alphas = np.full(afs.size, np.NaN)
        devs_lo, devs_hi = np.full(afs.size, np.NaN),  np.full(afs.size, np.NaN)
    afs, ns = afs.astype(int), ns.astype(int)

    # Unpack and round/format to same number of significant digits as Stable32
    afs2, taus2, ns2, alphas2, devs_lo2, devs2, devs_hi2 = actual_results
    taus2, devs_lo2, devs2, devs_hi2 = S32format(taus2), S32format(devs_lo2), \
        S32format(devs2), S32format(devs_hi2)

    # for theo1, ns are also formatted in rounded scientific notation
    if func.__name__ == 'theo1':
        ns2 = S32format(ns2)

    logger.debug(" AF TAU   #   ALPHA   DEV_LO       DEV      DEV_HI")

    # Test results match row-by-row
    for i, row in enumerate(expected_results):

        # Extract individual fields from Stable32 results
        af, tau, n, alpha, minus, dev, plus = \
            afs[i], taus[i], ns[i], alphas[i], devs_lo[i], devs[i], devs_hi[i]

        # Extract individual fields from Allantoolkit results
        af2, tau2, n2, alpha2, minus2, dev2, plus2 = \
            afs2[i], taus2[i], ns2[i], alphas2[i], devs_lo2[i], devs2[i], \
            devs_hi2[i]

        logger.debug("%s <-REF", [af, tau, n, alpha, minus, dev, plus])
        logger.debug("%s <-ACTUAL", [af2, tau2, n2, alpha2, minus2, dev2,
                                    plus2])

        assert af == af2, f'S32:{af} vs. AT {af2}'
        assert tau == tau2, f'S32:{tau} vs. AT {tau2}'
        assert n == n2, f'S32:{n} vs. AT {n2}'
        assert dev == dev2, f'S32:{dev} vs. AT: {dev2}'

        if file_type != 'tie':

            # Test also estimated noise type
            if test_alpha:
                assert alpha == alpha2, f'S32:{alpha} vs. AT {alpha2}'

            # Test also confidence intervals
            if test_ci:
                assert np.isclose(minus, minus2, rtol=1e-3, atol=0), \
                    f'S32:\n{minus}\nvs.\nAT:\n{minus2}'
                assert np.isclose(plus, plus2, rtol=1e-2, atol=0), \
                    f'S32:\n{plus}\nvs.\nAT:\n{plus2}'