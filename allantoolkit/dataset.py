"""
Allantools dataset object

**Authors:** Frederic Meynadier (frederic.meynadier "at" gmail.com),
    Mike DePalatis (http://mike.depalatis.net)
"""

import logging
import numpy as np
from . import utils
from . import allantools

# shorten type hint to save some space
Array = allantools.Array
Taus = allantools.Taus

# Spawn module-level logger
logger = logging.getLogger(__name__)


class Dataset:
    """ Dataset class for Allantools

    :Example:
        ::

            import numpy as np
            # Load random data
            a = allantoolkit.Dataset(data=np.random.rand(1000))
            # compute mdev
            a.compute("mdev")
            print(a.out["stat"])

    compute() returns the result of the computation and also stores it in the
    object's ``out`` member.

    """

    def __init__(self, data: Array, rate: float, data_type: str = "phase",
                 taus: Taus = None) -> None:
        """ Initialize object with input data

        Parameters
        ----------
        data: np.array
            Input data. Provide either phase or frequency (fractional,
            adimensional)
        rate: float
            The sampling rate for data, in Hz. Defaults to 1.0
        data_type: {'phase', 'freq'}
            Data type, i.e. phase or frequency. Defaults to "phase".
        taus: np.array
            Array of tau values, in seconds, for which to compute statistic.
            Optionally set taus=["all"|"octave"|"decade"] for automatic
            calculation of taus list

        Returns
        -------
        Dataset()
            A Dataset() instance

        """

        if data_type not in ['phase', 'freq']:
            raise ValueError(f"Invalid data_type value: {data_type}. "
                             f"Should be `phase` or `freq`.")

        self.rate = rate
        self.tau_0 = 1 / rate
        self.data_type = data_type
        self.taus = taus

        self.x = data if data_type == 'phase' else None
        self.y = data if data_type == 'freq' else None

        # Data scaling factors
        self.x_scale = 1.
        self.y_scale = 1.

        # Initialise attributes for stat results
        self.dev_type = None
        self.afs = None
        self.ns = None
        self.alphas = None
        self.devs_lo = None
        self.devs = None
        self.devs_hi = None

    def convert(self, to: str, adjust_zero_freq: bool = True) -> None:

        if to == 'phase':

            self.x = utils.frequency2phase(y=self.y, rate=self.rate,
                                           normalize=adjust_zero_freq)

        elif to == 'freq':

            self.y = utils.phase2frequency(x=self.x, rate=self.rate)

        else:
            raise ValueError(f"Conversion should be to `phase` or `freq`. "
                             f"Not to {to}.")

    def scale(self, data_type: str, factor: float, fractional: bool = False) \
            -> None:

        if data_type == 'phase' and self.x is not None:

            self.x *= factor
            self.x_scale = factor

        elif data_type == 'freq' and self.y is not None:

            self.y = utils.frequency2fractional(f=self.y, v0=factor) if \
                fractional else factor*self.y
            self.y_scale = factor

        else:
            raise ValueError(f"{data_type} data not available. Try converting "
                             f"your data to the desired type first.")

    def normalize(self, data_type: str) -> None:

        if data_type == 'phase' and self.x is not None:

            self.x -= np.nanmean(self.x)

        elif data_type == 'freq' and self.y is not None:

            self.y -= np.nanmean(self.y)

        else:
            raise ValueError(f"{data_type} data not available. Try converting "
                             f"your data to the desired type first.")

    def calc(self, dev_type: str, data_type: str = 'phase') -> None:
        """Evaluate the passed function with the supplied data.

        Stores result in self.out.

        Parameters
        ----------
        dev_type: str
            Name of the :mod:`allantoolkit` function to evaluate

        Returns
        -------
        result: dict
            The results of the calculation.

        """

        if data_type not in ['phase', 'freq']:
            raise ValueError(f"Invalid data_type value: {data_type}. "
                             f"Should be `phase` or `freq`.")

        # Dispatch to correct deviation calculator
        try:
            func = getattr(allantools, dev_type)

        except AttributeError:
            raise ValueError(f"{dev_type} is not implemented in Allantoolkit.")

        data = self.x if data_type == 'phase' else self.y

        if data is None:
            raise ValueError(f"{data_type} data not available. Try "
                             f"converting your data to the desired type "
                             f"first.")

        out = func(data=data, rate=self.rate, data_type=data_type,
                   taus=self.taus)

        self.dev_type = dev_type
        self.afs = out.afs
        self.ns = out.ns
        self.alphas = out.alphas
        self.devs_lo = out.devs_lo
        self.devs = out.devs
        self.devs_hi = out.devs_hi

        print(out)

    def write_results(self, filename, digits=5, header_params={}):
        """ Output result to text

        Save calculation results to disk. Will overwrite any existing file.

        Parameters
        ----------
        filename: str
            Path to the output file
        digits: int
            Number of significant digits in output
        header_params: dict
            Arbitrary dict of params to be included in header

        Returns
        -------
        None
        """

        with open(filename, 'w') as fp:
            fp.write("# Generated by Allantoolkit\n")
            fp.write("# Input data type: {}\n".format(self.inp["data_type"]))
            fp.write("# Input data rate: {}\n".format(self.inp["rate"]))
            for key, val in header_params.items():
                fp.write("# {}: {}\n".format(key, val))
            # Fields
            fp.write(("{af:>5s} {tau:>{width}s} {n:>10s} {alpha:>5s} "
                      "{minsigma:>{width}} "
                      "{sigma:>{width}} "
                      "{maxsigma:>{width}} "
                      "\n").format(
                          af="AF",
                          tau="Tau",
                          n="N",
                          alpha="alpha",
                          minsigma="min_" + self.out["stat_id"],
                          sigma=self.out["stat_id"],
                          maxsigma="max_" + self.out["stat_id"],
                          width=digits + 5
                      )
                     )
            out_fmt = ("{af:5d} {tau:.{prec}e} {n:10d} {alpha:5s} "
                       "{minsigma:.{prec}e} "
                       "{sigma:.{prec}e} "
                       "{maxsigma:.{prec}e} "
                       "\n")
            for i in range(len(self.out["taus"])):
                fp.write(out_fmt.format(
                    af=int(self.out["taus"][i] / self.out["taus"][0]),
                    tau=self.out["taus"][i],
                    n=int(self.out["stat_n"][i]),
                    alpha="NaN",  # Not implemented yet
                    minsigma=self.out["stat"][i] - self.out["stat_err"][i]/2,
                    sigma=self.out["stat"][i],
                    maxsigma=(self.out["stat"][i] +
                              self.out["stat_err"][i]/2),
                    prec=digits-1,
                ))
