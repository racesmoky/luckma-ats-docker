import traceback

import requests

from ml.code.util import logging_util

_logger = logging_util.get_logger()


# $ curl -d '{"signature_name":"predict",
#                     "inputs":{"Sex": ["male", "female"],
#                               "Pclass": [3, 3],
#                               "Fare": [7.8292,7.0000],
#                               "Age": [34.5, 47]}}'
#                               -X POST http://localhost:8501/v1/models/model:predict


def predict(x, symbol):
    """
        requests.exceptions.HTTPError is used to handle invalid http response such as 401 properly

        All exceptions that Requests explicitly raises inherit from requests.exceptions.RequestException
        requests.exceptions.RequestException is subclass of requests.exceptions.HTTPError
        so we catch HTTPError first

        traceback.print_exc() is used to print failing stacktrace to the logger for debugging

    :param x:
    :param symbol:
    :return:
    """
    try:
        # data must be JSON formatted string, do not use single quote
        data = {}
        host = ""
        response = requests.post(host, data="")
        response.raise_for_status()

        # Validate response code
        return response

    except requests.exceptions.HTTPError:
        _logger.error(f'failed to predict with {x}, unexpected http response, stacktrace={traceback.print_exc()}')
    except requests.exceptions.RequestException:
        _logger.error(f'failed to predict with {x}: request exception, stacktrace={traceback.print_exc()}')
