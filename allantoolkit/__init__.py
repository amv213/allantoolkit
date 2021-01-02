import logging

import allantoolkit.tables
import allantoolkit.stats
import allantoolkit.noise_id
import allantoolkit.devs
import allantoolkit.ci
import allantoolkit.bias
import allantoolkit.api
import allantoolkit.noise
import allantoolkit.plot
import allantoolkit.realtime
import allantoolkit.testutils
import allantoolkit.utils

# Setting package root-logger
logger = logging.getLogger(__name__)
logger.setLevel("WARNING")  # (default) best practice

# Uncomment if want to prevent ALL library’s logged events being output to
# sys.stderr in the absence of user-side logging configuration
# logger.addHandler(logging.NullHandler())
