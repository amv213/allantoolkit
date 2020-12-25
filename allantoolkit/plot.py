"""
Allantools plotting utilities

**Authors:** Frederic Meynadier (frederic.meynadier "at" gmail.com),
    Mike DePalatis (http://mike.depalatis.net)
"""

import logging

# Spawn module-level logger
logger = logging.getLogger(__name__)


class Plot:
    """ A class for plotting data once computed by Allantools

    :Example:
        ::

            import allantoolkit
            import numpy as np
            a = allantoolkit.Dataset(data=np.random.rand(1000))
            a.compute("mdev")
            b = allantoolkit.Plot()
            b.plot(a)
            b.show()

    Uses matplotlib. self.fig and self.ax stores the return values of
    matplotlib.pyplot.subplots(). plot() sets various defaults, but you
    can change them by using standard matplotlib method on self.fig and self.ax
    """
    def __init__(self, no_display=False):
        """ set ``no_display`` to ``True`` when we don't have an X-window
        (e.g. for tests)
        """
        try:
            import matplotlib
            if no_display:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            self.plt = plt
        except ImportError:
            raise RuntimeError("Matplotlib is required for plotting")
        self.fig, self.ax = plt.subplots()
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")

    def plot(self, atDataset,
             errorbars=False,
             grid=False,
             **kwargs
             ):
        """ Use matplotlib methods for plotting

        Additional keywords arguments are passed to
        :py:func:`matplotlib.pyplot.plot`.

        Parameters
        ----------
        atDataset : allantoolkit.Dataset()
            a dataset with computed data
        errorbars : boolean
            Plot errorbars. Defaults to False
        grid : boolean
            Plot grid. Defaults to False

        """

        err_m = atDataset.devs - atDataset.devs_lo
        err_p = atDataset.devs_hi - atDataset.devs

        if errorbars:
            self.ax.errorbar(atDataset.taus,
                             atDataset.devs,
                             yerr=[err_m, err_p],
                             **kwargs
                             )
        else:
            self.ax.plot(atDataset.taus,
                         atDataset.devs,
                         **kwargs
                         )
        self.ax.set_xlabel("Tau")
        self.ax.set_ylabel(atDataset.dev_type)
        self.ax.grid(grid, which="minor", ls="-", color='0.65')
        self.ax.grid(grid, which="major", ls="-", color='0.25')

    def show(self):
        """Calls matplotlib.pyplot.show()

        Keeping this separated from ``plot()`` allows to tweak display before
        rendering
        """
        self.plt.show()

    def save(self, f):
        self.plt.savefig(f)
