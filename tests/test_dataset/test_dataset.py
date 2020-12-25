import allantoolkit
import tempfile
import pytest

fcts = [
    allantoolkit.allantools.adev,
    allantoolkit.allantools.oadev,
    allantoolkit.allantools.mdev,
    allantoolkit.allantools.tdev,
    allantoolkit.allantools.hdev,
    allantoolkit.allantools.ohdev,
    allantoolkit.allantools.totdev,
    allantoolkit.allantools.mtotdev,
    allantoolkit.allantools.ttotdev,
    allantoolkit.allantools.htotdev,
    allantoolkit.allantools.theo1,
    allantoolkit.allantools.tierms,
    allantoolkit.allantools.mtie,
]


def test_no_function_in_allantools(dataset):
    with pytest.raises(AttributeError):
        dataset.compute("nosuchfunction")


# Now functions that are in .allantools are already all in whitelist. So
# there is no need to check anymore for a blacklist
@pytest.mark.skip
def test_blacklisted_function(dataset):
    with pytest.raises(RuntimeError):
        dataset.compute("calc_mtotdev")


@pytest.mark.parametrize("func", fcts)
def test_compute_functions(dataset, func):

    dev_type = func.__name__

    # calculate deviation X on freq data
    dataset.calc(dev_type=dev_type, data_type='freq')

    assert dataset.devs is not None

