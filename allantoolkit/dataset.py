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
        self.filename = None

        # Read data from file if a filename is provided
        if isinstance(data, str) or isinstance(data, Path):
            self.filename = data
            data = utils.read_datafile(fn=data)

        # 'live' phase or frequency dataset to manipulate
        self.data = data

        # This will not get updated if operations are done on x/y datasets
        self.data_ori = data

        # Initialise attributes for stat results
        self.dev_type = None
        self.N = None

        self.afs = None
        self.taus = None
        self.ns = None
        self.alphas = None
        self.devs_lo = None
        self.devs = None
        self.devs_hi = None

    def convert(self, to: str, normalize: bool = True) -> 'Dataset':
        """Generates a new Dataset with data converted between phase and
        frequency data.

        Gaps in frequency data are filled in the converted phase data with
        values based on the interpolated frequency.

        References:
            [RileyStable32Manual]_ (Convert Function, pg.173-174)

        Args:
            to:         the data_type to which to convert. Should be
                        'phase' or 'freq'.
            normalize:  if `True`, removes average frequency before f -> p
                        conversion
        """

        if to == 'phase':

            new_data = utils.frequency2phase(y=self.y, rate=self.rate,
                                             normalize=normalize)

        elif to == 'freq':

            new_data = utils.phase2frequency(x=self.x, rate=self.rate)

        else:
            raise ValueError(f"Conversion should be to `phase` or `freq`. "
                             f"Not to {to}.")

        new_dataset = Dataset(data=new_data, rate=self.rate, data_type=to)
        new_dataset.filename = self.filename  # tag along orig filename if any

        return new_dataset

    def normalize(self) -> None:
        """Removes the average value from phase or frequency data. This
        normalizes the data to have a mean value of zero.

        References:
            [RileyStable32Manual]_ (Normalize Function, pg.175)
        """

        self.data -= np.nanmean(self.data)

    def average(self, m: int) -> None:
        """Combine groups of phase or frequency data into values
        corresponding to longer averaging time, tau = m*tau_0.

        Phase data averaging (decimation) is done by simply eliminating the
        intermediate data points to form data at longer tau. Frequency
        averaging is done by finding the mean value of the data points being
        averaged.

        Gaps (NaN) are ignored, and the average will be a gap if all points
        in the group being averaged are gaps.

        References:
            [RileyStable32Manual]_ (Average Function, pg.177)

        Args:
            m:          averaging factor at which to average
        """

        self.data = utils.decimate(data=self.data, m=m,
                                   data_type=self.data_type)

        # Rescale for consistency
        self.tau_0 *= m
        self.rate = 1./self.tau_0

    def fill(self):
        """Fills gaps (NaNs) in phase or frequency data with interpolated
        values. Leading and trailing gaps are removed.

        Caution:
            The data inserted are simply interpolated values using the
            closest non-gap data points at either side of each gap. No
            attempt is made to simulate noise, and the resulting statistics
            are not necessarily valid. It is generally better practice to
            leave gaps unfilled.

        References:
            [RileyStable32Manual]_ (Fill Function, pg.179)
        """

        n_gaps = self.data[~np.isnan(self.data)].size
        self.data = utils.fill_gaps(data=self.data)

        logger.info("Filled # %i %s gaps in the dataset.", n_gaps,
                    self.data_type)

    def scale(self, addend: float = 0., multiplier: float = 1.,
              slope: float = 0., reverse: bool = False) -> None:
        """Modifies the selected phase or frequency data by an additive or
        multiplicative factor, by adding a linear slope, or by reversing the
        data.

        Data reversal is particularly useful in conjunction with frequency jump
        detection and analysis, where it can provide a better average location
        estimate for a frequency jump. Repeating the reversal will restore the
        original data.

        References:
            [RileyStable32Manual]_ (Scale Function, pg.181-2)

        Args:
            addend:     additive factor to be added to the data.
            multiplier: multiplicative factor by which to scale the data.
            slope:      linear slope by which to scale the data.
            reverse:    if `True` reverses the data, after scaling it.
        """

        self.data = utils.scale(data=self.data, addend=addend,
                                multiplier=multiplier, slope=slope,
                                reverse=reverse)

    def part(self, start: int = 0, end: int = None) -> None:
        """Clears all except the selected portion of phase or
        frequency data. The part function changes the data in memory.

        References:
            [RileyStable32Manual]_ (Part Function, pg.183)

        Args:
            start:      index of first point to include. Defaults to start
                        of dataset
            end:        index of last point to include, Defaults to end of
                        dataset
        """

        if end is None or end + 1 >= self.data.size:
            self.data = self.data[start:]

        else:
            self.data = self.data[start:end]

        logger.info("Dataset parted to size %i", self.data.size)

    # TODO: Implement
    def filter(self) -> None:
        """Filters the current phase or frequency data.

        References:
            [RileyStable32Manual]_ (Filter Function, pg.185-6)

        Args:

        """
        raise NotImplementedError("Filter Function yet to be implemented!")

    # TODO: Implement
    def show_stats(self) -> None:
        """Displays basic statistics for, and a simple plot of, phase or
        frequency data.

        References:
            [RileyStable32Manual]_ (Statistics Function, pg.187-8)

        Args:

        """
        raise NotImplementedError("Statistics Function yet to be implemented!")

    # TODO: Implement
    def check(self) -> None:
        """Check for and remove outliers from frequency data.

        References:
            [RileyStable32Manual]_ (Check Function, pg.189-90)

        Args:

        """
        raise NotImplementedError("Statistics Function yet to be implemented!")

    # TODO: Implement
    def drift(self, type: str, remove: bool = False) -> None:
        """Analyze phase or frequency data for frequency drift, or find
        frequency offset in phase data.

        It is common to remove the deterministic frequency drift from phase
        or frequency data before analyzing the noise with Allan variance
        statistics. It is sometimes useful to remove only the frequency
        offset from phase data.

        References:
            [RileyStable32Manual]_ (Drift Function, pg.191-4)

            http://www.wriley.com/Frequency%20Drift%20Characterization%20in%20Stable32.pdf

        Args:
            type:   drift analysis type
            remove: if `True` remove the drift from the data
        """

        if self.data_type == 'phase':
            # Four frequency drift methods are available for phase data

            if type == 'quadratic':

                t = range(1, self.data.size+1)
                coeffs = np.polyfit(t, self.data, deg=2)
                a, b, c = coeffs[::-1]
                slope = 2*c / self.tau_0

                logger.warning("\nQuadratic")
                logger.warning("a=%.7g\nb=%.7g\nc=%.7g", a, b, c)
                logger.warning("%.7g", slope)

            elif type == 'avg 2diff':

                slope = self.data[2:] - 2*self.data[1:-1] + self.data[:-2]

                slope = np.mean(slope)

                logger.warning("\nAvg of 2nd Diff")
                logger.warning("%.7g", slope)

            elif type == '3-point':

                M = self.data.size
                slope = 4*(self.data[-1] - 2*self.data[M//2] + self.data[0])
                slope /= (M-1) ** 2

                logger.warning("\n3-point")
                logger.warning("%.7g", slope)

            elif type == 'greenhall':
                # C. A. Greenhall, "A frequency-drift estimator and its removal
                # from modified Allan variance," Proceedings of International
                # Frequency Control Symposium, Orlando, FL, USA, 1997,
                # pp. 428-432, doi: 10.1109/FREQ.1997.638639.

                def w(n, w0=0):

                    return w0 + np.sum(self.data[:n])

                N = self.data.size
                w_N = np.sum(self.data)
                n1 = int(N / 10)
                r1 = n1 / N

                slope = 6. / (N**3 * self.tau_0**2 * r1 * (1 - r1)) * (
                    w_N - (w(N-n1) - w(n1)) / (1 - 2*r1))

                logger.warning("\nGreenhall")
                logger.warning("%.7g", slope)

            elif type == 'linear':

                t = range(self.data.size)
                coeffs = np.polyfit(t, self.data, deg=1)
                a, b = coeffs[::-1]
                slope = b

                logger.warning("\nLinear")
                logger.warning("a=%.7g\nb=%.7g", a, b)
                logger.warning("slope=%.7g", slope)
                logger.warning("f_offset=%.7g", slope*self.rate)

            elif type == 'avg 1diff':

                slope = np.diff(self.data)

                #FIXME: floating point precision error when accumulating
                slope = np.mean(slope)

                logger.warning("\nAvg of 1st Diff")
                logger.warning("slope=%.7g", slope)
                logger.warning("f_offset=%.7g", slope*self.rate)

            elif type == 'endpoints':

                slope = (self.data[-1] - self.data[0]) / (self.data.size - 1)

                logger.warning("\nEndpoints")
                logger.warning("slope=%.7g", slope)
                logger.warning("f_offset=%.7g", slope*self.rate)

            else:

                raise ValueError(f"`{type}` drift analysis method is not "
                                 f"available for phase data")

        else:
            raise NotImplementedError(
                "Statistics Function yet to be implemented!")

    def calc(self, dev_type: str, taus: Taus = 'octave') -> None:
        """Calculate the selected frequency stability statistic on phase or
        fractional frequency data over a range of averaging times. The
        averaging times may be selected in octave or sub-decade increments,
        or at every (or almost) possible tau out to a reasonable fraction of
        the record length.

        References:
            [RileyStable32Manual]_ (Run Function, pg.237-242)

        Args:
            dev_type:   name of the :mod:`allantoolkit` function to evaluate
                        e.g. 'oadev'
            taus:       array of averaging times for which to compute
                        deviation. Can also be one of the keywords: `all`,
                        `many`, `octave`, `decade`.
        """

        # Dispatch to correct deviation calculator
        try:
            func = getattr(allantools, dev_type)

        except AttributeError:
            raise ValueError(f"{dev_type} is not implemented in Allantoolkit.")

        # Calculate statistics
        out = func(data=self.data, rate=self.rate, data_type=self.data_type,
                   taus=taus)

        # Metadata
        self.dev_type = dev_type
        self.N = self.data.size

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
                 f"{self.data_type.title()} Data Points 1 through " \
                 f"{self.N} of {self.N}\n" \
                 f"Maximum =           \t{np.nanmax(self.data)}\n" \
                 f"Minimum =           \t{np.nanmin(self.data)}\n" \
                 f"Average =           \t{np.nanmean(self.data)}\n" \
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
