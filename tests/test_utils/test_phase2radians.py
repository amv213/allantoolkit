import allantoolkit
import numpy as np

V0 = np.random.random()*100


def test_return_type(data_with_gaps):
    """Test that function returns a numpy.array"""

    output = allantoolkit.utils.phase2radians(data_with_gaps, v0=V0)

    assert isinstance(output, np.ndarray)


def test_works_with_no_data():
    """Test that function is invariant if input is an empty list"""

    empty = np.array([])

    output = allantoolkit.utils.phase2radians(empty, v0=V0)

    assert np.array_equal(output, empty)


def test_works_all_gaps(data_with_only_gaps):
    """Test that function is invariant if input is an empty list"""

    output = allantoolkit.utils.phase2radians(data_with_only_gaps, v0=V0)

    np.testing.assert_array_equal(output, data_with_only_gaps)


def test_conversion():
    """Test that function performs expected conversion"""

    data = np.array([2.1, 4., 5.6, np.NaN, 4.2])
    v0 = 3.

    output = allantoolkit.utils.phase2radians(data, v0=v0)

    expected_output = np.array([39.58406744,  75.39822369, 105.55751316,
                                np.NaN, 79.16813487])

    np.testing.assert_allclose(output, expected_output, rtol=1e-9,
                               atol=1e-9, equal_nan=True)

