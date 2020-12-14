import pytest
import allantoolkit
import numpy as np

# Randomise fixed parameters
RATE = np.random.random()*100

# List all available deviation types
dev_types = [dev for dev in allantoolkit.tables.STOP_RATIOS.keys()]
tau_types = ['octave', 'decade', 'all', np.random.random(128)]
tau_types_stop_ratio = tau_types[:2]
tau_types_immune = tau_types[2:]


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types)
def test_return_type(data_with_gaps, dev_type, taus):
    """Test that function returns a `Taus` NamedTuple"""

    output = allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                              dev_type=dev_type, taus=taus)

    assert isinstance(output, allantoolkit.utils.Taus)


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types)
def test_return_namedtuple(data_with_gaps, dev_type, taus):
    """Test that fields of the output NamedTuple match what expected"""

    output = allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                              dev_type=dev_type, taus=taus)

    assert output._fields == ('taus', 'afs')


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types)
def test_outputs_sorted(data_with_gaps, dev_type, taus):
    """Test that function returns a sorted arrays"""

    output = allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                              dev_type=dev_type, taus=taus)

    assert (np.all(np.diff(output.taus) >= 0) and np.all(np.diff(output.afs)
                                                         >= 0))


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types)
def test_outputs_types(data_with_gaps, dev_type, taus):
    """Test types of arrays in named tuple"""

    output = allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                              dev_type=dev_type, taus=taus)

    assert output.taus.dtype.name == 'float64' and \
           output.afs.dtype.name == 'int64'


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types)
def test_outputs_same_length(data_with_gaps, dev_type, taus):
    """Test output arrays have same size"""

    output = allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                              dev_type=dev_type, taus=taus)

    assert output.taus.size == output.afs.size


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types)
def test_outputs_related(data_with_gaps, dev_type, taus):
    """Test output taus are correctly calculated from averaging factors"""

    output = allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                              dev_type=dev_type, taus=taus)

    assert np.allclose(output.taus, output.afs / RATE, rtol=1e-12, atol=1e-12,
                       equal_nan=True)


@pytest.mark.parametrize('dev_type', dev_types)
def test_misspelled_taus(data_with_gaps, dev_type):
    """Test that ValueError is raised if ask for wrong taus mode"""

    with pytest.raises(ValueError):
        allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                         dev_type=dev_type, taus='alll')


@pytest.mark.parametrize('taus', tau_types_stop_ratio)
def test_misspelled_devtype(data_with_gaps, taus):
    """Test that KeyError is raised if provide wrong dev_type"""

    with pytest.raises(KeyError):
        allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                         dev_type='andev', taus=taus)


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types)
def test_infinite_tau0(data_with_gaps, dev_type, taus):
    """Test that ZeroDivisionError is raised if rate = 0"""

    with pytest.raises(ZeroDivisionError):
        allantoolkit.utils.tau_generator(data_with_gaps, rate=0.,
                                         dev_type=dev_type, taus=taus)


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types_stop_ratio)
def test_Stable32_stopratio(data_with_gaps, dev_type, taus):
    """Test that averaging factors are capped by Stable32 stop ratio when
    applicable"""

    output = allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                              dev_type=dev_type, taus=taus)

    assert max(output.afs) <= data_with_gaps.size // \
           allantoolkit.tables.STOP_RATIOS[dev_type]


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types_immune)
def test_max_m(data_with_gaps, dev_type, taus):
    """Test that averaging factors are capped by max_n parameter if Stable32
    stop ratio not kicking in"""

    output = allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                              dev_type=dev_type, taus=taus,
                                              maximum_m=2)

    if output.afs.size:
        assert max(output.afs) <= 2


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types)
def test_m_within_dataset_bound(data_with_gaps, dev_type, taus):
    """Test that averaging factors are always smaller than size of dataset"""

    output = allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                              dev_type=dev_type, taus=taus)

    assert min(output.afs) < data_with_gaps.size


@pytest.mark.parametrize('taus', tau_types)
def test_even(data_with_gaps, taus):
    """Test that averaging factors are even if dev_type is theo1"""

    output = allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                              dev_type='theo1', taus=taus)

    assert np.all(output.afs % 2 == 0)


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types)
def test_min_m(data_with_gaps, dev_type, taus):
    """Test that minimum averaging factor is greater than 0 [if any
    averaging factors calculated]"""

    output = allantoolkit.utils.tau_generator(data_with_gaps, rate=RATE,
                                              dev_type=dev_type, taus=taus)

    if output.afs.size:
        assert min(output.afs) > 0


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types)
def test_empty_data(dev_type, taus):
    """Test that ValueError raised if calculating taus for empty
    data"""

    with pytest.raises(ValueError):
        allantoolkit.utils.tau_generator(data=np.array([]), rate=RATE,
                                         dev_type=dev_type, taus=taus)


@pytest.mark.parametrize('dev_type', dev_types)
@pytest.mark.parametrize('taus', tau_types)
def test_negative_max_m_data(data_with_gaps, dev_type, taus):
    """Test that get empty results if cap the maximum averaging factor to
    negative numbers"""

    output = allantoolkit.utils.tau_generator(data=data_with_gaps, rate=RATE,
                                              dev_type=dev_type, taus=taus,
                                              maximum_m=-1)

    assert output.taus.size == 0 and output.afs.size == 0


@pytest.mark.parametrize('dev_type', dev_types)
def test_empty_taus(data_with_gaps, dev_type):
    """Test that get empty results if provide empty averaging times"""

    output = allantoolkit.utils.tau_generator(data=data_with_gaps, rate=RATE,
                                              dev_type=dev_type,
                                              taus=np.array([]))

    assert output.taus.size == 0 and output.afs.size == 0


@pytest.mark.parametrize('dev_type', dev_types)
def test_NaN_taus(data_with_gaps, dev_type):
    """Test that get empty results if provide NaN averaging times"""

    taus = np.array([np.NaN, np.NaN, np.NaN, np.NaN])

    output = allantoolkit.utils.tau_generator(data=data_with_gaps, rate=RATE,
                                              dev_type=dev_type,
                                              taus=taus)

    assert output.taus.size == 0 and output.afs.size == 0


# TODO:
#  test decades, octaves and all building correctly
#  test outputs unique
