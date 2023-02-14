########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   Last edited: 05/25/20
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################

import datetime
import os
import platform
import shlex
import subprocess
import sys
import time

from ml.code import constants
from ml.code.util import logging_util

_logger = logging_util.get_logger()


def get_dir_sep():
    """
        Returns platform specific directory separator

    :return:
    """
    if platform.system() == "Windows":
        return '\\'
    if platform.system() == "Darwin":
        return '/'
    if platform.system() == "Linux":
        return '/'

    raise EnvironmentError(f'Unexpected system in project_dir_sep {platform.system()}')


def get_today_date():
    return datetime.datetime.now().strftime('%m-%d-%y')


def validate_date(date_str):
    try:
        spliited_date_str = date_str.rsplit('-', 1)
        reformatted_date = spliited_date_str[0] + '-20' + spliited_date_str[-1]
        datetime.datetime.strptime(reformatted_date, '%m-%d-%Y')
    except IndexError:
        print(f'Invalid date_str: {date_str}, expected mm-dd-yy', file=sys.stderr)
        return False
    except ValueError:
        print(f'Invalid date_str: {date_str}, expected mm-dd-yy', file=sys.stderr)
        return False
    return True


def run_command(command):
    """
        Runs given the shell command, command gets parsed with posix=False using shlex lib
        with default posix=True, backslash on Window gets omitted

        subprocess.Popen expected list(<command>, arg1, arg2, ...) which is provided via shlex.split

    :param command: shell command to run
    :return: return code from running the shell command
    """
    command_splits = shlex.split(command, posix=False)
    _logger.info(f'running popen command ={command_splits}')

    process = subprocess.Popen(command_splits, stdout=subprocess.PIPE, universal_newlines=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            _logger.info(output.strip())
    rc = process.poll()
    return rc


def verify_docker_image(docker_image_name):
    """
         verifies docker image exists in the current system

    :param docker_image_name: : formatted with docker <REPOSITORY>:<TAG>
    :return: None if :param docker_image_name exists in the system, else Exception
    """
    try:
        shell_output = subprocess.check_output(['docker', 'images'], universal_newlines=True)
        _logger.info(f'shell_output:\n {shell_output}')
        if docker_image_name.replace(':', '   ') not in shell_output:
            raise DockerImageNotFoundError(f'{docker_image_name} not found in the system, run build.py first')
        else:
            _logger.info(f'docker_image_name found={docker_image_name}')
    except (subprocess.CalledProcessError, DockerImageNotFoundError) as e:
        _logger.exception(f'failed to verify_docker_image: {docker_image_name}\n'
                          f'stacktrace: {e}')
        sys.exit(1)


def verify_system_command(command):
    """
        Verifies :param command: is installed and available in the system

    :param command: any platform dependent command
    :return:  None if :param command exists in the system, else Exception
    """
    try:
        shell_output: str
        if platform.system() == "Windows":
            shell_output = subprocess.check_output(['where', command], universal_newlines=True)
        else:
            shell_output = subprocess.check_output(['which', command], universal_newlines=True)
        _logger.debug(f'verify_system_command::shell_output:\n {shell_output}')
        if command not in shell_output:
            raise FileNotFoundError(f'{command} not found in the system')
        else:
            _logger.info(f'{command} installed at: {shell_output}')
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        _logger.exception(f'failed to verify_system_command: {command}\n'
                          f'stacktrace: {e}')
        sys.exit(1)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            _logger.info('%r  %2.2fs' % (method.__name__, (te - ts)))
        return result

    return timed


def is_debug():
    try:
        return True if os.environ[constants.DEBUGGING] and bool(os.environ[constants.DEBUGGING]) else False
    except KeyError:
        return False


class DockerImageNotFoundError(OSError):
    """ Docker Image not found. """

    def __init__(self, *args, **kwargs):
        pass
