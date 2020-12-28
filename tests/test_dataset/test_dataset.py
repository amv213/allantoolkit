import allantoolkit
import tempfile
import pytest

fcts = [
    allantoolkit.devs.adev,
    allantoolkit.devs.oadev,
    allantoolkit.devs.mdev,
    allantoolkit.devs.tdev,
    allantoolkit.devs.hdev,
    allantoolkit.devs.ohdev,
    allantoolkit.devs.totdev,
    allantoolkit.devs.mtotdev,
    allantoolkit.devs.ttotdev,
    allantoolkit.devs.htotdev,
    allantoolkit.devs.theo1,
    allantoolkit.devs.tierms,
    allantoolkit.devs.mtie,
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

    # calculate deviation X on dataset fixture
    dataset.calc(dev_type=dev_type)

    assert dataset.devs is not None

