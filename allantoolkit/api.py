"""
Allantools dataset object

**Authors:** Frederic Meynadier (frederic.meynadier "at" gmail.com),
    Mike DePalatis (http://mike.depalatis.net)
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from . import utils
from . import devs
from . import noise_id
from . import tables
from pathlib import Path
from typing import Union
from scipy.optimize import curve_fit
from scipy import signal

# shorten type hint to save some space
Array = devs.Array
Taus = devs.Taus

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

    def filter(self, type: str, f_low: float = None,
               f_high: float = None) -> None:
        """Filters the current phase or frequency data.

        Low pass filtration can be useful for removing high frequency noise
        that may otherwise obscure underlying variations in the data.
        Its effect is similar to data averaging, but does not lengthen the
        sampling interval or reduce the number of data points. High pass
        filtration can be useful for removing large amplitude low frequency
        fluctuation in the data due to divergent noise, drift or wandering in
        order to better see and analyze the high frequency noise. This is
        particularly effective when the drift or wandering does not fit a
        function to allow its removal. Band pass filtration can be useful for
        analyzing the amplitude variations of a discrete interfering component.
        Its function resembles that of a classic wave analyzer. Band stop
        filtration can be useful for removing a discrete interfering
        component. By repeating this operation, multiple components may be
        removed without significantly affecting the underlying behavior.

        References:
            [RileyStable32Manual]_ (Filter Function, pg.185-6)

        Args:
            type:   the type of filter. Can be any of {'lowpass', 'highpass',
                    'bandpass', 'bandstop'}
            f_low:  for a 'highpass' or 'band*' filter, the lower cutoff
                    frequency.
            f_high: for a 'lowpass' or 'band*' filter, the higher cutoff
                    frequency.
        """

        self.data = utils.filter(data=self.data, rate=self.rate, type=type,
                                 f_low=f_low, f_high=f_high)

    def show(self) -> plt.Axes:
        """Displays basic statistics for, and a simple plot of, phase or
        frequency data.

        # TODO: find out which custom algorithm Stable32 uses to calculate
        median, sigma, and std

        References:
            [RileyStable32Manual]_ (Statistics Function, pg.187-8)
            [RileyStable32]_ (10.12, pg.109)

        Returns:
            plot axes
        """

        # Log Statistics:

        N = self.data.size
        n_gaps = np.count_nonzero(np.isnan(self.data))
        maximum = np.max(self.data)
        minimum = np.min(self.data)
        average = np.nanmean(self.data)
        median = np.nanmedian(self.data)
        std = np.nanstd(self.data)
        var = np.nanvar(self.data)
        p = noise_id.acf_noise_id_core(z=self.data, dmax=2)
        alpha = p + 2 if self.data_type == 'phase' else p
        noise = tables.ALPHA_TO_NAMES.get(alpha)

        logger.info("\n\nSTATISTICS:\n"
                    "\tFile:      \t%s\n"
                    "\tData Type: \t%s\n"
                    "\t# Points:  \t%i\n"
                    "\t# Gaps:    \t%i\n"
                    "\tMaximum:   \t%.7g\n"
                    "\tMinimum:   \t%.7g\n"
                    "\tAverage:   \t%.7g\n"
                    "\tMedian:    \t%.7g\n"
                    "\tStd Dev:   \t%.7g\n"
                    "\tSigma:     \t%.7g\n"
                    "\tNoise:     \t%s (%i)\n", self.filename, self.data_type,
                    N, n_gaps, maximum, minimum, average, median, std, var,
                    noise, alpha)

        # Plot:

        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()

        # plot frequency as steps to highlight averaging time associated
        # with measurement
        drawstyle = 'steps-post' if self.data_type == 'freq' else 'default'

        ax.plot(self.data, drawstyle=drawstyle)
        ax.axhline(average, ls='--', c='DarkGray')

        ax.set_ylim(average - 6*std, average + 6*std)
        ax.set_xlim(0, N)

        ax.set_ylabel("Measured Value")
        ax.set_xlabel("Point #")

        return ax

    def check(self, sigmas: float = 3.5, replace: str = None) -> None:
        """Check for and remove outliers from frequency data. Outliers are
        replaced by gaps (NaNs).

        Outliers are detected using the median absolute deviation (MAD). The
        MAD is a robust statistic based on the median of the data. It is the
        median of the scaled absolute deviation of the data points from their
        median value.

        References:

            [RileyStable32Manual]_ (Check Function, pg.189-90)

            [RileyStable32]_ (10.11, pg.108-9)

            https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list

            https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

        Args:
            sigmas:     desired number of deviations for which a point is to be
                        classified as outlier. Defaults to 3.5
            replace:    whether to replace the detected outliers, and how.
                        If set to `all`, all outliers are replaced. If set to
                        `largest`, only the largest outlier is removed. If
                        not set, outliers are not removed but only logged.
        """

        self.data = utils.replace_outliers(data=self.data, sigmas=sigmas,
                                           replace=replace)

    def drift(self, type: str = None, m: int = 1, remove: bool = False) -> \
            None:
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
            m:      averaging factor for the log, diffusion or autoregression
                    noise identification models
            remove: if `True` remove the drift from the data
        """

        self.data = utils.drift(data=self.data, rate=self.rate,
                                data_type=self.data_type, type=type, m=m,
                                remove=remove)

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
            func = getattr(devs, dev_type)

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

        logger.info("\n%s", out)

    def save(self, filename: Union[str, Path] = None) -> None:
        """ Saves statistical analysis results to a .TXT file.

        Will overwrite any existing file.

        Args:
        filename:   path to output .TXT file
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

    def plot_hist(self) -> plt.Axes:
        """Plots a histogram of phase or frequency data.

        Returns:
            plot axes
        """

        N = self.data.size
        sigma = np.nanstd(self.data)
        mu = np.nanmean(self.data)

        # Spawn figure
        fig = plt.figure(figsize=(7, 7))
        ax = plt.gca()

        # Histogram
        n, bins, patches = ax.hist(self.data, bins=int(np.sqrt(N)),
                                   density=True, rwidth=0.8)

        # Gaussian kernel
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            -0.5 * (1 / sigma * (bins - mu)) ** 2))

        ax.plot(bins, y)

        # Vertical Mean Line
        ax.axvline(mu, ls='--', c='DarkGray')

        ax.set_xlim(mu - 6*sigma, mu + 6*sigma)

        ax.set_xlabel("Data Value")
        ax.set_ylabel("# Points Per Bin")

        return ax

    def plot(self) -> plt.Axes:
        """Plot frequency stability results.

        Make sure to call `.calc()` first to generate the results you wish
        to be plotted.

        Returns:
            plot axes
        """

        if self.devs is None:
            raise ValueError("No stability analysis results to plot. Make "
                             "sure you have called .calc() to populate "
                             "results first.")

        # Spawn figure
        fig = plt.figure(figsize=(13, 8))
        ax = plt.gca()

        # Confidence Interval
        ax.fill_between(self.taus, y1=self.devs_hi, y2=self.devs_lo,
                        color='LightGray', alpha=0.8, linewidth=0)

        # Deviations
        ax.plot(self.taus, self.devs, marker='o', ls='-')

        # Text Box
        tau_box = f"Tau\n\n"
        for t in self.taus:
            tau_box += f"{int(round(t)):<6n}\n"
        sigma_box = f"Sigma\n\n"
        for dev in self.devs:
            sigma_box += f"{dev:<.2e}\n"

        ax.text(0.8, 0.9, tau_box, horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes)
        ax.text(0.85, 0.9, sigma_box, horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes)


        # Axes Parameters
        ax.set_ylim(10**np.floor(np.log10(min(self.devs))),
                    10**np.ceil(np.log10(max(self.devs))))
        ax.set_xlim(1, 10**np.ceil(np.log10(self.taus[-1])))

        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.grid(which='major', linestyle='--')
        ax.grid(which='minor', linestyle=':')

        ax.set_title("FREQUENCY STABILITY")
        ax.set_ylabel(f"{self.dev_type.upper()}")
        ax.set_xlabel("Averaging Time (s)")

        return ax
