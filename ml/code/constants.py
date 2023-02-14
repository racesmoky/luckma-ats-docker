########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   Last edited: 05/25/20
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################
from pathlib2 import Path

# logger must be named tensorflow and logger must be instantiated before importing Tensorflow to silence absl logger
LOGGER_NAME = 'tensorflow'
PROJECT_NAME = 'luckma-ats-docker'
PROJECT_HOME_PATH = Path(__file__).resolve().parents[2]

# data constants
INPUT_FILE_TYPE_CSV = 'csv'
INPUT_FILE_TYPE_TF_RECORD = 'tfrecords'

INPUT_FILE_TYPE = INPUT_FILE_TYPE_TF_RECORD

BUILD_CONFIG_FILE = 'buildconfig.json'
HYPERPARAMETERS_FILE = 'hyperparameters.json'
RESOURCE_CONFIG_FILE = 'resourceconfig.json'
INPUT_DATA_CONFIG_FILE = 'inputdataconfig.json'

# .env constants - must be defined in your .env file
CUDA_VISIBLE_DEVICES = 'CUDA_VISIBLE_DEVICES'
DEBUGGING = 'DEBUGGING'
DEPLOYMENT_ENV = 'DEPLOYMENT_ENV'
DEPLOYMENT_MODULE = 'DEPLOYMENT_MODULE'
DOCKER_USER = 'DOCKER_USER'
DOCKER_REPO = 'DOCKER_REPO'
INPUT_CONFIG_PATH = 'INPUT_CONFIG_PATH'
INPUT_DATA_PATH = 'INPUT_DATA_PATH'
MODEL_EXPORT_PATH = 'MODEL_EXPORT_PATH'
PROCESSED_DATA_PATH = 'PROCESSED_DATA_PATH'

# channel prefix
TRAIN_CHANNEL = 'train'
EVAL_CHANNEL = 'eval'
TEST_CHANNEL = 'test'

# Tensorflow constants
TF_RECORD_PARTS_PATTERN = 'part-r-*'
TF_RECORD_P_NUM_READS = 2
TF_RECORD_SHUFFLE_RATIO = 0.10

# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
TF_GLOBAL_RANDOM_SEED = 13

# Tensorflow model constants
OPTIMIZER_FUNC_ADAGRAD = 'Adagrad'
OPTIMIZER_FUNC_RMSPROP = 'RMSprop'
OPTIMIZER_FUNC_ADAM = 'Adam'

# Feature Columns - Contextual
FEATURE_OHLC_CLOSE = 'ohlc_close'
FEATURE_OHLC_PCR = 'ohlc_pcr'
FEATURE_OHLC_USW = 'ohlc_usw'
FEATURE_OHLC_LSW = 'ohlc_lsw'
FEATURE_BAS_RQ = 'bas_rq'
FEATURE_VOL_RQ = 'vol_rq'
FEATURE_CMMC_MAX = 'cmmc_max'

LABEL_COL = 'ohlc_close_label'

INDEX_COL = 's_ts'

FEATURE_COLS = [FEATURE_OHLC_CLOSE, FEATURE_OHLC_PCR, FEATURE_OHLC_USW, FEATURE_OHLC_LSW, FEATURE_BAS_RQ,
                FEATURE_VOL_RQ, FEATURE_CMMC_MAX]

DEBUG_STEPS = 50
