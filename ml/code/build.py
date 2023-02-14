########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   Last edited: 05/31/20
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################

import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

from ml.code import constants
from ml.code.util import data_util, logging_util

_logger = logging_util.get_logger()

BUILD_SCRIPT_NAME = 'build_and_push'
VALID_DEPLOYMENT_ENV = ['aws', 'localhost']
VALID_DEPLOYMENT_MODULE = ['nvidia', 'tensorflow-serving']


def validate_env_vars():
    """
        Valiadtes the following envrionment variables exist in your .doc environment:
            - DEPLOYMENT_ENV
            - DOCKER_USER
            - DOCKER_REPO
    :return:
    """
    assert constants.DEPLOYMENT_ENV in os.environ, 'Failed to build, set DEPLOYMENT_ENV = localhost or aws in .env'
    assert os.environ[
               constants.DEPLOYMENT_ENV] in VALID_DEPLOYMENT_ENV, 'Failed to build, set DEPLOYMENT_ENV = localhost or aws in .env'
    assert constants.DOCKER_USER in os.environ, 'Failed to build, set DOCKER_USER = <your_docker_username> in .env'
    assert constants.DOCKER_REPO in os.environ, 'Failed to build, set DOCKER_REPO = <your_docker_repo> in .env'
    assert constants.DEPLOYMENT_MODULE in os.environ, 'Failed to build, set DEPLOYMENT_MODULE = ' \
                                                      '(e.g. sagamaker, nvidia, tensorflow-serving) in .env'
    assert os.environ[
               constants.DEPLOYMENT_MODULE] in VALID_DEPLOYMENT_MODULE, 'Failed to build, set DEPLOYMENT_MODULE = ' \
                                                                        '(e.g. sagamaker, nvidia, tensorflow-serving) in .env'


def get_build_script_file_name():
    """
        Returns platform dependent builld script file name
            - .sh for Linux
            - .bat for Windows

    :return: file name
    """
    if platform.system() == "Windows":
        return f'{BUILD_SCRIPT_NAME}.bat'
    elif platform.system() == "Darwin" or platform.system() == "Linux":
        return f'{BUILD_SCRIPT_NAME}.sh'
    else:
        raise EnvironmentError(f'Unexpected system in project_dir_sep {platform.system()}')


def get_build_dir_path():
    """
        Returns build dir path which build.py will execute from configurations from
            - .env
            - input/config/buildconfig.json

    :return: path to directory containing build file script
    """
    d_env = os.environ['DEPLOYMENT_ENV']
    d_module = os.environ['DEPLOYMENT_MODULE']

    config = data_util.load_build_config()
    version = config[d_env][d_module]['Version']

    return os.path.join(constants.PROJECT_HOME_PATH, 'ml', 'build', 'docker', d_env, d_module, version)


def execute_build_script(p, d, d_user, d_repo):
    """
        Executes build script :param f from directory :param d

    :param d: build directory name
    :param p: build script path
    :param d_user: docker user
    :param d_repo: docker repository
    :return:
    """
    # Make sure to execute with shell=False, True flag introduces security vulnerability
    # https://docs.python.org/2/library/subprocess.html#frequently-used-arguments
    try:
        subprocess.check_call([p, d, d_user, d_repo], shell=False)
    except subprocess.CalledProcessError as e:
        _logger.error(f'failed to execute build script = {build_script_path}, '
                      f'exit_code= {e.returncode}, '
                      f'stack_trace={traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    # load dotenv
    data_util.load_dotenv()

    # validate required environ vars exist in .env
    validate_env_vars()

    # get build script dir path
    build_script_dir = get_build_dir_path()

    # get platform dependent build script file name
    build_script_filename = get_build_script_file_name()

    # construct build_script_path and make sure it exists
    build_script_path = os.path.join(build_script_dir, build_script_filename)

    # check to make sure script exists
    assert Path(build_script_path).exists(), f'unexpected build_script_path={build_script_path}, check configs'

    # execute build script
    execute_build_script(build_script_path, build_script_dir, os.environ[constants.DOCKER_USER],
                         os.environ[constants.DOCKER_REPO])
