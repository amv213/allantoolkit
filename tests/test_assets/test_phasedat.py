"""
  PHASE.DAT test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  PHASE.DAT comes with Stable32 (version 1.53 was used in this case)

"""

import pathlib
import pytest
import allantoolkit
import allantoolkit.testutils as testutils

# top level directory with asset files
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets'

# input data files, and associated verbosity, tolerance, and acquisition rate
assets = [
    ('phasedat/PHASE.DAT', 1, 1e-4, 1.),
]


# input result files and function which should replicate them
results = [
    ('phase_dat_adev.txt', allantoolkit.allantools.adev),
    ('phase_dat_oadev.txt', allantoolkit.allantools.oadev),
    ('phase_dat_mdev.txt', allantoolkit.allantools.mdev),
    ('phase_dat_tdev.txt', allantoolkit.allantools.tdev),
    ('phase_dat_hdev.txt', allantoolkit.allantools.hdev),
    ('phase_dat_ohdev.txt', allantoolkit.allantools.ohdev),
    ('phase_dat_totdev.txt', allantoolkit.allantools.totdev),
    pytest.param('phase_dat_htotdev_octave_nobias.txt',
                 allantoolkit.allantools.htotdev, marks=pytest.mark.slow),
    pytest.param('phase_dat_mtotdev_octave.txt',
                 allantoolkit.allantools.mtotdev, marks=pytest.mark.slow),
    pytest.param('phase_dat_ttotdev_octave.txt',
                 allantoolkit.allantools.ttotdev, marks=pytest.mark.slow),
    ('phase_dat_theo1_alpha0_decade.txt', allantoolkit.allantools.theo1),
    ('phase_dat_mtie.txt', allantoolkit.allantools.mtie),
    ('phase_dat_tierms.txt', allantoolkit.allantools.tierms),
]


@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
@pytest.mark.parametrize('result, fct', results)
def test_generic(datafile, result, fct, verbose, tolerance, rate):

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / result

    testutils.test_row_by_row(fct, datafile, rate, result,
                              verbose=verbose, tolerance=tolerance)
