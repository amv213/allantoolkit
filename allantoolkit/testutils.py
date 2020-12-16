"""
 Useful collection of functions for the allantoolkit test-suite

"""

import os
import time
import gzip
import numpy
import logging
from pathlib import Path
from contextlib import contextmanager

# Spawn module-level logger
logger = logging.getLogger(__name__)


# read a simple data-file with phase or frequency numbers on each line
def read_datafile(filename: Path):

    p = []
    if filename.suffix == '.gz':
        with gzip.open(filename, mode='rt') as f:
            for line in f:
                if not line.startswith("#"):  # skip comments
                    p.append(float(line))
    else:
        with open(filename) as f:
            for line in f:
                if not line.startswith("#"):  # skip comments
                    p.append(float(line))

    return numpy.array(p)


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
def read_stable32(resultfile, datarate):
    devresults = read_resultfile(resultfile)
    print("Read ", len(devresults), " rows from ", resultfile)
    rows=[] 
    # parse textfile produced by Stable32
    for row in devresults:
        if len(row) == 7:  # typical ADEV result file has 7 columns of data
            d={}
            d['m']= row[0]
            d['tau']= row[1]
            d['n']=row[2]
            d['alpha']=row[3]
            d['dev_min']=row[4]
            d['dev']=row[5]
            d['dev_max']=row[6]

            rows.append(d)
        elif len(row) == 4:  # the MTIE/TIErms results are formatted slightly differently
            d={}
            d['m']= row[0]
            d['tau']= row[1]
            d['n']=row[2]
            d['dev']=row[3]
            rows.append(d)
    return rows


def to_fractional(data):
    mu = numpy.mean(data)
    return [(x-mu)/mu for x in data]


# test one tau-value (i.e. one row in the result file) at a time
# test a deviation function by:
# - running the function on the datafile
# - reading the correct answers from the resultfile
# - checking that tau, n, and dev are correct
def test_row_by_row(function, datafile, datarate, resultfile, verbose=False, tolerance=1e-4, frequency=False, normalize=False):
    # if Stable32 results were given with more digits we could decrease tolerance

    data = read_datafile(datafile)

    if normalize:  # convert frequencies in Hz to fractional frequencies
        data = to_fractional(data)

    print("Read ", len(data), " values from ", datafile)

    s32rows = read_stable32(resultfile, datarate)
    print("test of function ", function )
    if verbose:
        print("Tau N  \t DEV(Stable32) \t DEV(allantoolkit) \t rel.error\t bias")
    
    n_errors=0
    # run allantoolkit algorithm, row by row
    for s32data in s32rows:

        # If checking a theo1, the file will have an effective 75% of the
        # original one
        tau_ori = s32data['tau'] if function.__name__ != 'theo1' else \
            s32data['tau'] / 0.75

        if frequency:
            (taus2, devs2, errs2, ns2) = function(data, rate=datarate,
                                                  data_type="freq",
                                                  taus=[tau_ori])
        else:
            (taus2, devs2, errs2, ns2) = function(data, rate=datarate,
                                                  taus=[tau_ori])

        n_errors += check_equal(s32data['n'], ns2[0] )

        n_errors += check_equal( s32data['tau'], taus2[0] )

        n_errors += check_approx_equal( s32data['dev'], devs2[0], tolerance=tolerance, verbose=verbose )

        if verbose:
            rel_error = (devs2[0] - s32data['dev']) / s32data['dev']
            bias = pow(s32data['dev']/devs2[0],2)
            print("%.1f %d %0.6g \t %0.6g \t %0.6f \t %0.4f OK!" % ( s32data['tau'], s32data['n'], s32data['dev'], devs2[0], rel_error,bias))

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
                (label, end-start, (end-start0)/60))

    return time.clock()


def change_to_test_dir():
    # hack to run script from its own directory
    abspath = os.path.abspath(__file__)
    print(abspath)
    dname = os.path.dirname(abspath)
    os.chdir(dname)


@contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@contextmanager
def this_working_directory():
    """Changes working directory to directory calling function and returns to
    previous on exit."""

    this_dir = Path(__file__).parent

    prev_cwd = Path.cwd()
    os.chdir(this_dir)
    try:
        yield
    finally:
        os.chdir(prev_cwd)