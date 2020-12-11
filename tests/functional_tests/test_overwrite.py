import allantoolkit as at
import numpy


def test_overwrite():
    """ Tests if data is overwritten as pointed out in issue
    https://github.com/aewallin/allantools/issues/76
    """

    x1 = at.noise.white(num_points=1024)
    x2 = at.noise.pink(num_points=1024)
    ds1 = at.dataset.Dataset(x1)
    ds2 = at.dataset.Dataset(x2)
    r1 = ds1.compute('oadev')
    r2 = ds2.compute('oadev')

    # If both are identical, something is wrong...
    assert(not numpy.allclose(r1['stat'], r2['stat']))