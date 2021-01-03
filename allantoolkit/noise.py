"""
Allantools Noise generator

**Authors:** Julia Leute (julia.leute "at" gmail.com)
    Anders Wallin (anders.e.e.wallin "at" gmail.com)
    Alvise Vianello
"""

import logging
import numpy as np
from . import utils, tables

# Spawn module-level logger
logger = logging.getLogger(__name__)

# shorten type hint to save some space
Array = np.ndarray


class Noise:
    """Container for simulated power-law noise

    Input noise should have phase autospectral density:

    .. math::

        S_x(f) = {h_{\\alpha} \\over \\left( 2\\pi \\right)^2 } f^{\\beta}

    where :math:`\\alpha = \\beta + 2` and :math:`h_{\\alpha}` is the
    intensity coefficient.

    Args:
        data:           input noise data
        rate:           sampling rate of the input noise, in Hz.
        data_type:      input data type. Either ``phase`` or ``freq``.
        beta:           input data noise type, as its phase noise power
                        law exponent ``beta``
        qd:             discrete variance of input noise data
    """

    def __init__(self, data: Array, rate: float, data_type: str, beta: float,
                 qd: float):
        """Initialize object with input noise data
        """

        self.data = data
        self.data_type = data_type
        self.rate = rate
        self.tau_0 = 1 / rate
        self.n = data.size
        self.qd = qd
        self.b = beta
        self.alpha = beta + 2

        # Return PSD coefficients for given noise type
        self.g = self.calc_g_beta()
        self.h = self.calc_h_alpha()

    def calc_g_beta(self):
        """Returns phase power spectral density coefficient :math:`g_{\\beta}`
        for given noise data.

        The coefficient :math:`g_{\\beta}` is defined for noise with
        discrete variance :math:`Q^d` and sampling period :math:`\\tau_0` as:

        .. math::

            g_{\\beta} \\equiv 2 \\left( 2\\pi \\right)^{\\beta} Q^d
            \\tau_0^{\\beta+1}

        such that the noise shows phase power spectral density of:

        .. math::

            S_x(f) = g_{\\beta} f^{\\beta}

        References:
            [Kasdin1992]_
            Kasdin, N.J., Walter, T., "Discrete simulation of power law noise,
            " Frequency Control Symposium, 1992. 46th., Proceedings of the
            1992 IEEE, pp.274,283, 27-29 May 1992
            http://dx.doi.org/10.1109/FREQ.1992.270003 (Eq. 39)
        """

        return self.qd*2.0*pow(2.0*np.pi, self.b)*pow(self.tau_0, self.b+1.0)

    def calc_h_alpha(self):
        """Returns frequency power spectral density coefficient
        :math:`h_{\\alpha}` for given noise data.

        The coefficient :math:`h_{\\alpha}` is related to the corresponding
        phase power spectral density coefficient :math:`g_{\\beta}`
        ( :math:`\\beta = \\alpha -2` ), as:

        .. math::

            h_{\\alpha} = \\left( 2\\pi \\right)^2  g_{\\beta}

        such that the noise shows frequency power spectral density of:

        .. math::

            S_y(f) = h_{\\alpha} f^{\\alpha}

        .. seealso::

            Function :func:`allantoolkit.noise.Noise.calc_g_beta`
        
        References:
            [Kasdin1992]_
            Kasdin, N.J., Walter, T., "Discrete simulation of power law noise,
            " Frequency Control Symposium, 1992. 46th., Proceedings of the
            1992 IEEE, pp.274,283, 27-29 May 1992
            http://dx.doi.org/10.1109/FREQ.1992.270003 (Eq. 39)
        """

        return self.g * (2*np.pi)**2

    def adev(self, m: int) -> float:
        """Return predicted ADEV at given averaging factor for characteristic
        noise type.

        The expected Allan variance for a characteristic frequency power
        spectral density noise exponent ``alpha``, has the form:

        .. math::

            \\sigma_y^2(\\tau) = k h_{\\alpha} \\tau^\\mu

        where :math:`h_{\\alpha}` is the frequency power spectral
        density intensity coefficient for the given noise type, and the
        appropriate coefficients :math:`k` and :math:`\\mu` can be found in
        the reference.

        Args:
            m:  averaging factor at which to calculate predicted ADEV

        Returns:
            predicted ADEV at given averaging factor

        References:
            S. T. Dawkins, J. J. McFerran and A. N. Luiten, "Considerations on
            the measurement of the stability of oscillators with frequency
            counters," in IEEE Transactions on Ultrasonics, Ferroelectrics, and
            Frequency Control, vol. 54, no. 5, pp. 918-925, May 2007.
            doi: 10.1109/TUFFC.2007.337
        """

        # Check noise is characteristic noise type for which there is reference
        if self.alpha not in tables.ALPHA_TO_NAMES.keys() or \
                self.alpha in [-3, -4]:
            raise ValueError(f"No theoretical reference for ADEV for given "
                             f"noise type alpha = {self.alpha}")

        self.alpha = int(self.alpha)

        f_h = self.rate / 2  # Nyquist cutoff frequency

        # Prefactors for different noise types (Table I)
        ks = {2: (3*f_h) / (4*np.pi**2),
              1: (1.038 + 3*np.log(2*np.pi*f_h*m*self.tau_0)) / (4*np.pi**2),
              0: 1/2,
              -1: 2*np.log(2),
              -2: (2*np.pi**2) / 3}
        k = ks[self.alpha]

        # AVAR tau exponent
        mu = tables.ALPHA_TO_MU[self.alpha]

        avar = k * self.h * (m*self.tau_0)**mu

        return np.sqrt(avar)

    def mdev(self, m: int) -> float:
        """Return predicted MDEV at given averaging factor for characteristic
        noise type.

        The expected modified Allan variance for a characteristic frequency
        power spectral density noise exponent ``alpha``, has the form:

        .. math::

            \\sigma_y^2(\\tau) = k^\\prime h_{\\alpha} \\tau^{\\mu^\\prime}

        where :math:`h_{\\alpha}` is the frequency power spectral
        density intensity coefficient for the given noise type, and the
        appropriate coefficients :math:`k^\\prime` and :math:`\\mu^\\prime`
        can be found in the reference.

        Args:
            m:  averaging factor at which to calculate predicted MDEV

        Returns:
            predicted MDEV at given averaging factor

        References:
            S. T. Dawkins, J. J. McFerran and A. N. Luiten, "Considerations on
            the measurement of the stability of oscillators with frequency
            counters," in IEEE Transactions on Ultrasonics, Ferroelectrics, and
            Frequency Control, vol. 54, no. 5, pp. 918-925, May 2007.
            doi: 10.1109/TUFFC.2007.337
        """

        # Check noise is characteristic noise type for which there is reference
        if self.alpha not in tables.ALPHA_TO_NAMES.keys() or \
                self.alpha in [-3, -4]:
            raise ValueError(f"No theoretical reference for MDEV for given "
                             f"noise type alpha = {self.alpha}")

        self.alpha = int(self.alpha)

        # Prefactors for different noise types (Table I)
        k_primes = {2: 3 / (8 * np.pi ** 2),
                    1: (3 * np.log(256/27)) / (8 * np.pi**2),
                    0: 1 / 4,
                    -1: 2 * np.log((33**(11/16)) / 4),
                    -2: (11 * np.pi ** 2) / 20}
        k_prime = k_primes[self.alpha]

        # MVAR tau exponent
        mu_prime = tables.ALPHA_TO_MU_PRIME[self.alpha]

        avar = k_prime * self.h * (m * self.tau_0) ** mu_prime

        return np.sqrt(avar)


def kasdin_generator(nr: int, alpha: float, qd: float) -> Array:
    """Kasdin & Walter algorithm for discrete simulation of power law
    phase noise.

    Args:
        nr:     number of samples to generate. Should be a power of two.
        alpha:  frequency power law exponent ``alpha`` noise type to simulate.
        qd:     discrete variance of generated timeseries.

    Returns:
        timeseries of simulated noise

    References:
        [Kasdin1992]_
        Kasdin, N.J., Walter, T., "Discrete simulation of power law noise [for
        oscillator stability evaluation]," Frequency Control Symposium, 1992.
        46th., Proceedings of the 1992 IEEE, pp.274,283, 27-29 May 1992
        http://dx.doi.org/10.1109/FREQ.1992.270003
    """

    if not utils.is_power(np.array([nr])):
        raise ValueError(f"Cannot simulate noise timeseries of given length "
                         f"n = {nr}. Length should be a power of two.")

    # Phase noise power law exponent beta = alpha - 2
    b = alpha - 2

    # Fill wfb array with white noise based on given discrete variance
    wfb = np.zeros(nr * 2)
    wfb[:nr] = np.random.normal(0, np.sqrt(qd), nr)

    # Generate the hfb coefficients based on the noise type
    mhb = -b / 2.0
    hfb = np.zeros(nr * 2)
    hfb[0] = 1.0
    indices = np.arange(nr - 1)
    hfb[1:nr] = (mhb + indices) / (indices + 1.0)
    hfb[:nr] = np.multiply.accumulate(hfb[:nr])

    # Perform discrete Fourier transform of wfb and hfb time series
    wfb_fft = np.fft.rfft(wfb)
    hfb_fft = np.fft.rfft(hfb)

    # Perform inverse Fourier transform of the product of wfb and hfb FFTs
    time_series = np.fft.irfft(wfb_fft * hfb_fft)[:nr]

    return time_series


def generate_noise(n: int, alpha: float, qd: float, rate: float,
                   data_type: str) -> Noise:
    """Generate timeseries of simulated discrete power law noise of given type.

    Args:
        n:          number of samples to generate
        alpha:      frequency noise power law exponent ``alpha``
        qd:         discrete variance of generated timeseries
        rate:       sampling rate of the output frequency noise, in Hz.
        data_type:  desired output data type. Either ``phase`` or ``freq``.
                    Defaults to ``phase``.

    Returns:
        timeseries of simulated noise

    References:
        [Kasdin1992]_
        Kasdin, N.J., Walter, T., "Discrete simulation of power law noise [for
        oscillator stability evaluation]," Frequency Control Symposium, 1992.
        46th., Proceedings of the 1992 IEEE, pp.274,283, 27-29 May 1992
        http://dx.doi.org/10.1109/FREQ.1992.270003
    """

    if data_type not in ['phase', 'freq']:
        raise ValueError(f"Invalid data_type value: {data_type}. Should be "
                         f"`phase` or `freq`.")

    # Algorithm can only generate phase datasets of length 2^k
    # If want frequency data we will lose one point in conversion so add one
    k = np.ceil(np.log2(n)) if data_type == 'phase' else np.ceil(np.log2(n+1))
    n2 = int(2**k)

    # Phase noise
    z = kasdin_generator(nr=n2, alpha=alpha, qd=qd)

    if data_type == 'freq':
        z = utils.phase2frequency(x=z, rate=rate)

    # Trim to original desired length
    z = z[:n]
    assert z.size == n

    return Noise(data=z, rate=rate, data_type=data_type, beta=alpha-2, qd=qd)

# Wrappers


def violet(n: int, qd: float = 1, rate: float = 1,
           data_type: str = 'phase') -> Noise:
    """Kasdin & Walter discrete simulation of violet phase noise
    ( :math:`\\propto f^2` ).

    Violet noise has phase autospectral density:

    .. math::

        S_x(f) = {h_{\\alpha} \\over \\left( 2\\pi \\right)^2 } f^{\\alpha - 2}

    where :math:`\\alpha = 4` and :math:`h_{\\alpha}` is the intensity
    coefficient.

    It can be obtained by differentiation of white noise.

    .. seealso::

        Function :func:`allantoolkit.noise.generate_noise` for detailed usage
    """

    return generate_noise(n=n, rate=rate, data_type=data_type, alpha=4, qd=qd)


def blue(n: int, qd: float = 1, rate: float = 1,
         data_type: str = 'phase') -> Noise:
    """Kasdin & Walter discrete simulation of blue phase noise
    ( :math:`\\propto f` ).

    Blue noise has phase autospectral density:

    .. math::

        S_x(f) = {h_{\\alpha} \\over \\left( 2\\pi \\right)^2 } f^{\\alpha - 2}

    where :math:`\\alpha = 3` and :math:`h_{\\alpha}` is the intensity
    coefficient.

    .. seealso::

        Function :func:`allantoolkit.noise.generate_noise` for detailed usage
    """

    return generate_noise(n=n, rate=rate, data_type=data_type, alpha=3, qd=qd)


def white(n: int, qd: float = 1, rate: float = 1,
          data_type: str = 'phase') -> Noise:
    """Kasdin & Walter discrete simulation of white phase noise
    ( :math:`\\propto f^0` ).

    White noise has phase autospectral density:

    .. math::

        S_x(f) = {h_{\\alpha} \\over \\left( 2\\pi \\right)^2 } f^{\\alpha - 2}

    where :math:`\\alpha = 2` and :math:`h_{\\alpha}` is the intensity
    coefficient.

    .. seealso::

        Function :func:`allantoolkit.noise.generate_noise` for detailed usage
    """

    return generate_noise(n=n, rate=rate, data_type=data_type, alpha=2, qd=qd)


def pink(n: int, qd: float = 1, rate: float = 1,
         data_type: str = 'phase') -> Noise:
    """Kasdin & Walter discrete simulation of pink phase noise
    ( :math:`\\propto f^{-1}` ).

    Pink noise has phase autospectral density:

    .. math::

        S_x(f) = {h_{\\alpha} \\over \\left( 2\\pi \\right)^2 } f^{\\alpha - 2}

    where :math:`\\alpha = 1` and :math:`h_{\\alpha}` is the intensity
    coefficient.

    .. seealso::

        Function :func:`allantoolkit.noise.generate_noise` for detailed usage
    """

    return generate_noise(n=n, rate=rate, data_type=data_type, alpha=1, qd=qd)


def brown(n: int, qd: float = 1, rate: float = 1,
          data_type: str = 'phase') -> Noise:
    """Kasdin & Walter discrete simulation of brown phase noise
    ( :math:`\\propto f^{-2}` ).

    Brown noise has phase autospectral density:

    .. math::

        S_x(f) = {h_{\\alpha} \\over \\left( 2\\pi \\right)^2 } f^{\\alpha - 2}

    where :math:`\\alpha = 0` and :math:`h_{\\alpha}` is the intensity
    coefficient.

    More than a color, brown noise is Brownian or random-walk noise. It can
    be obtained by integration of white noise.

    .. seealso::

        Function :func:`allantoolkit.noise.generate_noise` for detailed usage
    """

    return generate_noise(n=n, rate=rate, data_type=data_type, alpha=0, qd=qd)


def flfm(n: int, qd: float = 1, rate: float = 1,
         data_type: str = 'phase') -> Noise:
    """Kasdin & Walter discrete simulation of Flicker FM phase noise
    ( :math:`\\propto f^{-3}` ).

    FLFM noise has phase autospectral density:

    .. math::

        S_x(f) = {h_{\\alpha} \\over \\left( 2\\pi \\right)^2 } f^{\\alpha - 2}

    where :math:`\\alpha = -1` and :math:`h_{\\alpha}` is the intensity
    coefficient.

    .. seealso::

        Function :func:`allantoolkit.noise.generate_noise` for detailed usage
    """

    return generate_noise(n=n, rate=rate, data_type=data_type, alpha=-1, qd=qd)


def rwfm(n: int, qd: float = 1, rate: float = 1,
         data_type: str = 'phase') -> Noise:
    """Kasdin & Walter discrete simulation of Random Walk FM phase noise
    ( :math:`\\propto f^{-4}` ).

    RWFM noise has phase autospectral density:

    .. math::

        S_x(f) = {h_{\\alpha} \\over \\left( 2\\pi \\right)^2 } f^{\\alpha - 2}

    where :math:`\\alpha = -2` and :math:`h_{\\alpha}` is the intensity
    coefficient.

    .. seealso::

        Function :func:`allantoolkit.noise.generate_noise` for detailed usage
    """
    return generate_noise(n=n, rate=rate, data_type=data_type, alpha=-2, qd=qd)


def custom(beta: int, n: int, qd: float = 1, rate: float = 1,
           data_type: str = 'phase') -> Noise:
    """Kasdin & Walter discrete simulation of arbitrary phase noise
    ( :math:`\\propto f^{\\beta}` ).

    This will generate noise with phase autospectral density:

    .. math::

        S_x(f) = {h_{\\alpha} \\over \\left( 2\\pi \\right)^2 } f^{\\beta}

    where :math:`\\alpha = \\beta + 2` and :math:`h_{\\alpha}` is the intensity
    coefficient.

    .. seealso::

        Function :func:`allantoolkit.noise.generate_noise` for detailed usage
    """
    return generate_noise(n=n, rate=rate, data_type=data_type, alpha=beta+2,
                          qd=qd)