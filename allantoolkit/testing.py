import allantoolkit
import pathlib
import numpy as np

# top level directory with original (frequency) data for these tests
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'tests/assets/Sr87'

X0 = allantoolkit.testutils.read_datafile(ASSETS_DIR /
                                          'phase0/phase0_data.txt')

# Data sampling rate
RATE = 0.4  # Hz, tau_0 = 2.5s


if __name__ == "__main__":

    ks, taus, vars = allantoolkit.stats.calc_theo1_fast(x=X0, rate=RATE)

    for k, tau, dev in zip(ks, taus, np.sqrt(vars)):
        print(f"K: {k}, AF:{2*k} - TAU:{tau} | DEV = {dev*np.sqrt(1.349)}")