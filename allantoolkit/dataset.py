"""
Allantools dataset object

**Authors:** Frederic Meynadier (frederic.meynadier "at" gmail.com),
    Mike DePalatis (http://mike.depalatis.net)
"""

import logging
import numpy as np
from . import utils
from . import allantools
from pathlib import Path
from typing import Union

# shorten type hint to save some space
Array = allantools.Array
Taus = allantools.Taus

# Spawn module-level logger
logger = logging.getLogger(__name__)


class Dataset:
    """ Dataset class for `allantoolkit`"""

    def __init__(self, data: Union[Array, str, Path], rate: float,
                 data_type: str = "phase") -> None:
        """ Initialize object with input data

        Args:
            data:       path to / array of phase data (in units of seconds) or
                        fractional frequency data.
            rate:       sampling rate of the input data, in Hz.
            data_type:  input data type. Either `phase` or `freq`.
        """

        if data_type not in ['phase', 'freq']:
            raise ValueError(f"Invalid data_type value: {data_type}. "
                             f"Should be `phase` or `freq`.")

        self.rate = rate
        self.tau_0 = 1 / rate
        self.data_type = data_type

        # Read data from file if a filename is provided
        if isinstance(data, str) or isinstance(data, Path):
            data = utils.read_datafile(fn=data)

        self.x = data if data_type == 'phase' else None
        self.y = data if data_type == 'freq' else None

        # Data scaling factors
        self.x_scale = 1.
        self.y_scale = 1.

        # Initialise attributes for stat results
        self.dev_type = None
        self.dev_data_type = None
        self.dataset = None
        self.N = None

        self.afs = None
        self.taus = None
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

    def calc(self, dev_type: str, data_type: str = 'phase',
             taus: Taus = 'octave') -> None:
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
                   taus=taus)

        # Metadata
        self.dev_type = dev_type
        self.dev_data_type = data_type
        self.N = data.size
        self.dataset = data

        # Dev Results
        self.afs = out.afs
        self.taus = out.taus
        self.ns = out.ns
        self.alphas = out.alphas
        self.devs_lo = out.devs_lo
        self.devs = out.devs
        self.devs_hi = out.devs_hi

        print(out)

    def save(self, filename: Union[str, Path] = None):
        """ Saves results to text

        Save calculation results to disk. Will overwrite any existing file.

        Parameters
        ----------
        filename: str
            Path to the output file

        Returns
        -------
        None
        """

        filename = Path.cwd() / (self.dev_type + '_results.txt') \
            if filename is None else filename

        table = np.column_stack((self.afs, self.taus, self.ns, self.alphas,
                                 self.devs_lo, self.devs, self.devs_hi))

        header = f"" \
                 f"ALLANTOOLKIT STABILITY ANALYSIS RESULTS:\n" \
                 f"{self.dev_data_type.title()} Data Points 1 through " \
                 f"{self.N} of {self.N}\n" \
                 f"Maximum =           \t{np.nanmax(self.dataset)}\n" \
                 f"Minimum =           \t{np.nanmin(self.dataset)}\n" \
                 f"Average =           \t{np.nanmean(self.dataset)}\n" \
                 f"Sigma Type =        \t{self.dev_type}\n" \
                 f"Confidence Factor = \t0.683\n" \
                 f"Deadtime T/tau =    \t1.000000\n\n"

        column_labels = "{:6s}\t{:12s}\t{:6s}\t{:6s}\t{:12s}\t{:12s}\t{:12s}" \
                        "".format('AF', 'TAU', '#', 'ALPHA', 'MIN SIGMA',
                                  'SIGMA', 'MAX SIGMA')

        header += column_labels

        np.savetxt(filename, table, header=header, delimiter='\t',
                   fmt=['%6i', '%12.4e', '%6i', '%6i', '%12.4e', '%12.4e',
                        '%12.4e'])

        logger.info("Stability analysis results saved in %s", filename)
