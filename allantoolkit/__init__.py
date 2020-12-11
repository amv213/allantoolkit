import logging

import allantoolkit.allantools
import allantoolkit.ci
import allantoolkit.dataset
import allantoolkit.noise
import allantoolkit.noise_kasdin
import allantoolkit.plot
import allantoolkit.realtime
import allantoolkit.testutils

# Setting package root-logger
logger = logging.getLogger(__name__)
logger.setLevel("WARNING")  # (default) best practice

# Uncomment if want to prevent ALL library’s logged events being output to
# sys.stderr in the absence of user-side logging configuration
# logger.addHandler(logging.NullHandler())
