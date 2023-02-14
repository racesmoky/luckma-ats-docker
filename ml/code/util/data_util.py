########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   Last edited: 05/25/20
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################

import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv

from ml.code import constants


def load_config(file_path):
    """
        https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        yaml.load(f) is unsafe and is now deprecated. load now requires the usage of Loader
        which adds additional complexity. Therefore, use safe_load which has default loader
        capabilities built in.

    :param file_path:
    :return: dict of hyperparameters
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def find_load_dotenv():
    try:
        dot_env_path = find_dotenv(filename='.env', raise_error_if_not_found=True)
        print(f'Found dotenv config file: {dot_env_path}')
        load_dotenv()
        print(f'Loaded dotenv config file: {dot_env_path}')
    except IOError:
        print(f'Failed to find dotenv config file, create .env first')
        sys.exit(1)  # https://stackoverflow.com/questions/13424926/exit-gracefully-if-file-doesnt-exist


def load_df(file_path):
    return pd.read_csv(file_path, header=0, index_col=0)


def load_build_config():
    return load_config(
        os.path.join(os.environ[constants.INPUT_CONFIG_PATH], constants.BUILD_CONFIG_FILE))


def load_hyperparameters_config():
    return load_config(
        os.path.join(os.environ[constants.INPUT_CONFIG_PATH], constants.HYPERPARAMETERS_FILE))


def load_input_data_config():
    return load_config(
        os.path.join(os.environ[constants.INPUT_CONFIG_PATH], constants.INPUT_DATA_CONFIG_FILE))


def load_resource_config():
    return load_config(
        os.path.join(os.environ[constants.INPUT_CONFIG_PATH], constants.RESOURCE_CONFIG_FILE))


def get_channel_dir(channel, data_type, exchange, date):
    """ Returns the directory containing the channel data file(s) which is:
    - <INPUT_DATA_PATH>/<channel>
    Returns:
        (str) The input data directory for the specified channel.
    """
    return os.path.join(os.path.join(os.environ[constants.INPUT_DATA_PATH], channel, data_type, exchange, date))


def get_channel_file_path(file_type, channel, data_type, exchange, date, symbol, config):
    return os.path.join(os.environ[constants.INPUT_DATA_PATH], config[f'train'][f'{channel}_channel_name'], data_type,
                        exchange, date, f'{symbol}.{file_type}')


def calc_ds_path_len(channel, data_type, exchange, date, symbol, config):
    """
        Calcuates ds len and path for the input dataset. Currently only supports CSV source data

    :param channel:
    :param data_type:
    :param exchange:
    :param date:
    :param symbol:
    :param config:
    :return:
    """
    source_file_path = get_channel_file_path(constants.INPUT_FILE_TYPE_CSV, channel, data_type, exchange, date, symbol,
                                             config)
    source_df = load_df(source_file_path)
    ds_path = get_channel_file_path(constants.INPUT_FILE_TYPE, channel, data_type, exchange, date, symbol, config)
    return ds_path, len(source_df),


def calc_ds_shuffle_buffer_size(ds_len):
    ds_shuffle_size = int(ds_len * constants.TF_RECORD_SHUFFLE_RATIO)
    return ds_shuffle_size


def maybe_create_dir(dir_path):
    data_dir_exists = Path(dir_path).exists()
    if not data_dir_exists:
        Path(dir_path).mkdir(mode=0o755, parents=True, exist_ok=True)
    return dir_path
