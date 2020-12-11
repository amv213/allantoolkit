import allantoolkit
import tempfile
import pytest


def test_no_function_in_allantools(dataset):
    with pytest.raises(AttributeError):
        dataset.compute("nosuchfunction")


def test_blacklisted_function(dataset):
    with pytest.raises(RuntimeError):
        dataset.compute("calc_mtotdev_phase")


def test_compute_functions(dataset):
    types = ["adev", "oadev", "mdev", "hdev", "ohdev", "tdev", "totdev",
             "mtotdev", "ttotdev", "htotdev", "theo1", "mtie", "tierms"]
    for calc in types:
        result = dataset.compute(calc)
        assert isinstance(result, dict)
        # Also test output for all types
        tmpfile = tempfile.NamedTemporaryFile(mode="w")
        dataset.write_results(tmpfile.name)
        dataset.write_results(tmpfile.name,
                              digits=10,
                              header_params={"test": 1, "test2": "foo"}
                              )


def test_dataset_parameters():
    ds = allantoolkit.dataset.Dataset()
    ds.set_input(allantoolkit.noise.white(10),
                 rate=1.234,
                 data_type="frequency",
                 taus=[1, 3, 4])
