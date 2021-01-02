"""
Allantools Noise generator

**Authors:** Julia Leute (julia.leute "at" gmail.com)
    Anders Wallin (anders.e.e.wallin "at" gmail.com)
    Alvise Vianello
"""

import logging
import numpy as np
from allantoolkit import utils

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

    def phase_psd_from_qd(self):
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

    def frequency_psd_from_qd(self):
        """Returns frequency power spectral density coefficient
        :math:`h_{\\alpha}` for given noise data.

        The coefficient :math:`h_{\\alpha}` is related to the equivalent
        phase power spectral density coefficient :math:`g_{\\beta}`, as:

        .. math::

            h_{\\alpha} = \\left( 2\\pi \\right)^2  g_{\\beta}

        such that the noise shows frequency power spectral density of:

        .. math::

            S_y(f) = h_{\\alpha} f^{\\alpha}

        References:
            [Kasdin1992]_
            Kasdin, N.J., Walter, T., "Discrete simulation of power law noise,
            " Frequency Control Symposium, 1992. 46th., Proceedings of the
            1992 IEEE, pp.274,283, 27-29 May 1992
            http://dx.doi.org/10.1109/FREQ.1992.270003 (Eq. 39)
        """

        return self.phase_psd_from_qd() * (2*np.pi)**2

    def adev(self, tau):
        """ return predicted ADEV of noise-type at given tau
        """
        prefactor = self.adev_from_qd(tau=tau)
        c = self.c_avar()
        avar = pow(prefactor, 2)*pow(tau, c)
        return np.sqrt(avar)

    def mdev(self, tau):
        """ return predicted MDEV of noise-type at given tau

        """
        prefactor = self.mdev_from_qd(tau=tau)
        c = self.c_mvar()
        mvar = pow(prefactor, 2)*pow(tau, c)
        return np.sqrt(mvar)

    def c_avar(self):
        """ return tau exponent "c" for noise type.
            AVAR = prefactor * h_a * tau^c
        """
        if self.b == -4:
            return 1.0
        elif self.b == -3:
            return 0.0
        elif self.b == -2:
            return -1.0
        elif self.b == -1:
            return -2.0
        elif self.b == 0:
            return -2.0

    def c_mvar(self):
        """ return tau exponent "c" for noise type.
            MVAR = prefactor * h_a * tau^c
        """
        if self.b == -4:
            return 1.0
        elif self.b == -3:
            return 0.0
        elif self.b == -2:
            return -1.0
        elif self.b == -1:
            return -2.0
        elif self.b == 0:
            return -3.0

    def adev_from_qd(self, tau=1.0):
        """ prefactor for Allan deviation for noise type defined by (qd, b,
        tau0)

        Colored noise generated with (qd, b, tau0) parameters will
        show an Allan variance of:

        AVAR = prefactor * h_a * tau^c

        where a = b + 2 is the slope of the frequency PSD.
        and h_a is the frequency PSD prefactor S_y(f) = h_a * f^a

        The relation between a, b, c is:

        +---------+---------+---------+---------+
        |    a    |    b    | c(AVAR) | c(MVAR) |
        +=========+=========+=========+=========+
        |   -2    |   -4    |    1    |    1    |
        +---------+---------+---------+---------+
        |   -1    |   -3    |    0    |    0    |
        +---------+---------+---------+---------+
        |    0    |   -2    |   -1    |   -1    |
        +---------+---------+---------+---------+
        |   +1    |   -1    |   -2    |   -2    |
        +---------+---------+---------+---------+
        |   +2    |    0    |   -2    |   -3    |
        +---------+---------+---------+---------+

        Coefficients from:
        S. T. Dawkins, J. J. McFerran and A. N. Luiten, "Considerations on
        the measurement of the stability of oscillators with frequency
        counters," in IEEE Transactions on Ultrasonics, Ferroelectrics, and
        Frequency Control, vol. 54, no. 5, pp. 918-925, May 2007.
        doi: 10.1109/TUFFC.2007.337

        """
        g_b = self.phase_psd_from_qd()
        f_h = 0.5/self.tau_0

        if self.b == 0:
            coeff = 3.0*f_h / (4.0*pow(np.pi, 2)) # E, White PM, tau^-1
        elif self.b == -1:
            coeff = (1.038+3*np.log(2.0*np.pi*f_h*tau))/(4.0*pow(np.pi, 2))# D, Flicker PM, tau^-1
        elif self.b == -2:
            coeff = 0.5 # C, white FM,  1/sqrt(tau)
        elif self.b == -3:
            coeff = 2*np.log(2) # B, flicker FM,  constant ADEV
        elif self.b == -4:
            coeff = 2.0*pow(np.pi, 2)/3.0 #  A, RW FM, sqrt(tau)

        return np.sqrt(coeff*g_b*pow(2.0*np.pi, 2))

    def mdev_from_qd(self, tau: float = 1.):
        # FIXME: tau is unused here - can we remove it?
        """ prefactor for Modified Allan deviation for noise
        type defined by (qd, b, tau0)

        Colored noise generated with (qd, b, tau0) parameters will
        show an Modified Allan variance of:

        MVAR = prefactor * h_a * tau^c

        where a = b + 2 is the slope of the frequency PSD.
        and h_a is the frequency PSD prefactor S_y(f) = h_a * f^a

        The relation between a, b, c is:

        +---------+---------+---------+---------+
        |    a    |    b    | c(AVAR) | c(MVAR) |
        +=========+=========+=========+=========+
        |   -2    |   -4    |    1    |    1    |
        +---------+---------+---------+---------+
        |   -1    |   -3    |    0    |    0    |
        +---------+---------+---------+---------+
        |    0    |   -2    |   -1    |   -1    |
        +---------+---------+---------+---------+
        |   +1    |   -1    |   -2    |   -2    |
        +---------+---------+---------+---------+
        |   +2    |    0    |   -2    |   -3    |
        +---------+---------+---------+---------+

        Coefficients from:
        S. T. Dawkins, J. J. McFerran and A. N. Luiten, "Considerations on
        the measurement of the stability of oscillators with frequency
        counters," in IEEE Transactions on Ultrasonics, Ferroelectrics, and
        Frequency Control, vol. 54, no. 5, pp. 918-925, May 2007.
        doi: 10.1109/TUFFC.2007.337

        """
        g_b = self.phase_psd_from_qd()
        #f_h = 0.5/tau0 #unused!?

        if self.b == 0:
            coeff = 3.0/(8.0*pow(np.pi, 2)) # E, White PM, tau^-{3/2}
        elif self.b == -1:
            coeff = (24.0*np.log(2)-9.0*np.log(3))/8.0/pow(np.pi, 2) # D, Flicker PM, tau^-1
        elif self.b == -2:
            coeff = 0.25 # C, white FM,  1/sqrt(tau)
        elif self.b == -3:
            coeff = 2.0*np.log(3.0*pow(3.0, 11.0/16.0)/4.0) # B, flicker FM,  constant MDEV
        elif self.b == -4:
            coeff = 11.0/20.0*pow(np.pi, 2) #  A, RW FM, sqrt(tau)

        return np.sqrt(coeff*g_b*pow(2.0*np.pi, 2))


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