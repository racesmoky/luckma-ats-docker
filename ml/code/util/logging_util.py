########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   Last edited: 05/25/20
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################

import logging
import logging.config
from os import path

from ml.code import constants

log_file_path = path.join(path.dirname(path.abspath(__file__)), '..', '..', 'input', 'config', 'logging.conf')
logging.config.fileConfig(log_file_path)


def get_logger():
    """
    non-resource leaking logger implementation. need to use os lib for abs path.
    Otherwise, it fails to find config with error: keyError: formatters
    https://stackoverflow.com/questions/23161745/python-logging-file-config-keyerror-formatters
    https://stackoverflow.com/questions/40559667/how-to-redirect-tensorflow-logging-to-a-file
    """
    logger = logging.getLogger(constants.LOGGER_NAME)
    logger.name = __name__
    return logger
