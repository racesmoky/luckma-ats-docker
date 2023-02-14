########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   Last edited: 05/25/20
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################

from __future__ import absolute_import, print_function

import glob
import os
import sys

import click

from ml.code import constants
from ml.code.util import data_util, logging_util, misc_util, train_util
from ml.code.util.misc_util import timeit

_logger = logging_util.get_logger()

import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.rnn import RNNEstimator


def build_context_feature_cols(col_names):
    """
    Builds context feature columns. Context feature columns for numeric columns must be defined as
    generic feature column (e.g. numeric_column) and not sequence_* because these are tensors going back for
    back prop, and tensorflow enforces it to be defined as DenseTensor

    :param col_names:
    :return:
    """
    return [tf.feature_column.numeric_column(col_name) for col_name in col_names]


def build_estimator(model_dir, config):
    """
      According to https://www.tensorflow.org/guide/migrate#make_the_code_20-native
        ***
        For compatibility reasons, a custom model_fn will still run in 1.x-style graph mode.
        This means there is no eager execution and no automatic control dependencies.

        It was recommended that you define your model using Keras, then use the tf.keras.estimator.model_to_estimator
        utility to turn your model into an estimator.
        ***
        Builds RNNEstimator

            - sequential_feature_columns - requires tf.Tensor.SparseTensor
            - context_feature_columns - requires tf.Tensor.DenseTensor

        Wraps tf.estimator.RegressionHead then builds tf.estimator.head.SequentialHead.

        tf.estimator.RegressionHead - built with specified label dimension (i.e. output_layer_dim)
                                    - uses MSE (Mean Squared Error) for loss function

        tf.estimator.Head.SequentialHead - required when building RNNEstimator with return_sequences=True

        :param model_dir: directory where model artifacts will save to
        :param config: hyperparameters
        :return: tensorflow_estimator.python.estimator.canned.rnn.RNNEstimator
        """
    # define sequence feature columns
    sequence_feature_column_names = constants.FEATURE_COLS
    sequence_feature_columns = build_sequence_feature_cols(sequence_feature_column_names)

    # define static regression head with expected label dimension
    static_head_label_dim = config['model']['output_units']
    static_head = tf.estimator.RegressionHead(label_dimension=static_head_label_dim)
    _logger.info(f'Built static_head(type=tf.estimator.RegressionHead, label_dim= : {static_head_label_dim}) ')

    # load tensorflow run config configurations from training hyper parameters
    save_summary_steps = config['train']['save_summary_steps']
    log_step_count_steps = config['train']['log_step_count_steps']
    tf_run_config = tf.estimator.RunConfig(model_dir=model_dir, save_summary_steps=save_summary_steps,
                                           log_step_count_steps=log_step_count_steps)

    # Builds RNNEstimator with specified head configuration, requires feature_columns
    estimator = RNNEstimator(
        head=static_head,
        sequence_feature_columns=sequence_feature_columns,
        context_feature_columns=None,
        rnn_cell_fn=rnn_cell_fn,
        return_sequences=False,
        model_dir=model_dir,
        optimizer=optimizer_fn(config),
        config=tf_run_config)

    _logger.info(f'build_estimator_tf_record: RNN_estimator created, model_dir = {model_dir}')
    return estimator


def build_sequence_feature_cols(col_names):
    """
    Builds sequence feature cols. NOT tensorflow.python.feature_column.feature_column_v2.DenseColumn. Valid column
    types are:
        - sequnce_numeric_column
        - categorical_column_with_*
        - embedding
    :param col_names:
    :return:
    """
    return [tf.feature_column.sequence_numeric_column(col_name) for col_name in col_names]


def build_model(config):
    """
        Legacy <2.1 model_to_estimator support method, doesn't support the usage of feature layers configuration

        IMPORTANT - Do not create feature layers configurations here as it requires dic of protos.
        IMPORTANT - features passed to this model must be 3d tensors.

        Warning - Do not define tf.keras.layers.Input (i.e. model.add(tf.keras.layers.Input(model_input_shape)))

        tf.Keras support variable (no input_shape) defined or fixed input_shape in the first input layer of
        neural network definition.

    :param config: hyperparameters
    :return: Compiled Model instance
    """
    model_input_shape = (config['train']['past_history'], config['train']['num_features'])
    _logger.info(f'model input shape: {model_input_shape}')
    model = tf.keras.models.Sequential()
    """
      recurrent_dropout - drops GPU training because tensorflow forces to use generic lstm kernel rather than
      performent gpu kernel, avoid using it unless it's absolutely necessary

      The requirements to use the cuDNN implementation are:

     1. `activation` == `tanh`
     2. `recurrent_activation` == `sigmoid`
     3. `recurrent_dropout` == 0
     4. `unroll` is `False`
     5. `use_bias` is `True`
     6. Inputs are not masked or strictly right padded.
    """
    model.add(tf.keras.layers.LSTM(config['model']['hidden_1_units'], input_shape=model_input_shape,
                                   return_sequences=config['model']['hidden_1_return_sequences'],
                                   kernel_regularizer=config['model']['hidden_1_kernel_regularizer'],
                                   recurrent_regularizer=config['model']['hidden_1_recurrent_regularizer'],
                                   dropout=config['model']['hidden_1_dropout']))  # Hidder layer 1
    model.add(tf.keras.layers.LSTM(config['model']['hidden_2_units'],
                                   return_sequences=config['model']['hidden_2_return_sequences'],
                                   kernel_regularizer=config['model']['hidden_2_kernel_regularizer'],
                                   recurrent_regularizer=config['model']['hidden_2_recurrent_regularizer'],
                                   dropout=config['model']['hidden_2_dropout']))  # Hidden layer 2

    # sigmoid > relu performed better on the model, only choose relu iff exploding or vanishing gradient
    model.add(tf.keras.layers.Dense(config['model']['output_units'], ))  # Output layer
    model.compile(optimizer=optimizer_fn(config), loss=config['model']['loss_func'])
    _logger.info(model.summary())
    return model


def configure_tf_memory_growth(is_enabled):
    """
        Configures tf memory growth settings
            # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth

    :param is_enabled: flag to enable/disable
    :return: None
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            memory_limit = int(os.environ['TF_GPU_MEMORY_LIMIT'])
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, is_enabled)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            _logger.info(f'{len(gpus)} "Physical GPUs,", {len(logical_gpus)} Logical GPUs')
            _logger.info(
                f'successfully allocated memory_limit={memory_limit}mb to gpus={gpus}, logical_gpus={logical_gpus}')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            _logger.info(e)
            sys.exit(1)


@timeit
def evaluate(estimator, file_path, ds_len, config):
    """
     steps: Number of steps for which to evaluate model. If `None`, evaluates
        until `input_fn` raises an end-of-input exception.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the evaluation call.
      checkpoint_path: Path of a specific checkpoint to evaluate. If `None`, the
        latest checkpoint in `model_dir` is used.  If there are no checkpoints
        in `model_dir`, evaluation is run with newly initialized `Variables`
        instead of ones restored from checkpoint.
      name: Name of the evaluation if user needs to run multiple evaluations on
        different data sets, such as on training data vs test data. Metrics for
        different evaluations are saved in separate folders, and appear
        separately in tensorboard.

    Returns:
      A dict containing the evaluation metrics specified in `model_fn` keyed by
      name, as well as an entry `global_step` which contains the value of the
      global step for which this evaluation was performed. For canned
      estimators, the dict contains the `loss` (mean loss per mini-batch) and
      the `average_loss` (mean loss per sample). Canned classifiers also return
      the `accuracy`. Canned regressors also return the `label/mean` and the
      `prediction/mean`.
    """
    num_epoch = config['train']['num_epoch']
    batch_size = config['train']['batch_size']
    eval_steps = train_util.calc_max_steps(ds_len, num_epoch, batch_size)
    _logger.info(f'evaluating for {eval_steps} steps from {file_path}')
    eval_output = estimator.evaluate(input_fn=lambda: input_fn(file_path, ds_len, config),
                                     steps=constants.DEBUG_STEPS if misc_util.is_debug() else eval_steps,
                                     hooks=None,
                                     checkpoint_path=None,
                                     name=None)

    _logger.info(f'evaluation complete, eval_output={eval_output}')
    return estimator


def export_model(estimator, symbol):
    """
        tf.estimator.export.build_parsing_serving_input_receiver_fn
            - Requires same feature specifications provided for input_fn

        There is no replacement for placeholder in Tf2 as its default mode is eager execution

        tf.estimator.export.build_raw_serving_input_receiver_fn
            - Requires feature specifications with tf.keras.backend.placeholder

    :param estimator:
    :param symbol:
    :return:
    """
    sequence_features_example_spec = tf.feature_column.make_parse_example_spec(
        build_sequence_feature_cols(constants.FEATURE_COLS))
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(sequence_features_example_spec)
    model_save_path = train_util.get_model_export_dir(symbol)
    estimator.export_saved_model(model_save_path, serving_input_fn)


def input_fn(file_path, ds_len, config):
    '''
           input_fn to be used for the model training, evaluation, and prediction.
           Expected file format: file path where parted (i.e. part-r-*) tf.Data.tfrecords is located

           By default, tf.data.TFRecordDataset files are read sequentially, setting num_parallel_reads > 1
                       will enable to read files in parallel

           https://stackoverflow.com/questions/49915925/output-differences-when-changing-order-of-batch-shuffle-and
           -repeat
           repeat -> shuffle: faster than shuffer -> repeats, causes epoch boundaries to get blurred (x)
           shuffle -> repeat: slower than repeat-> shuffer, stronger ordering guarantee (o)

           batch should be the last operation

       :param config:
       :param ds_len:
       :param file_path:
       :return: deserialized tf.data.Dataset
       '''
    file_names = sorted(glob.glob(os.path.join(file_path, constants.TF_RECORD_PARTS_PATTERN)))
    _logger.info(f'processing  {len(file_names)} serialized proto files...')
    assert len(file_names) > 0, f'Found no records at: {file_path}'

    # create tensorflow dataset containing all file names
    dataset = tf.data.Dataset.from_tensor_slices(file_names)

    sequence_features_example_spec = tf.feature_column.make_parse_example_spec(
        build_sequence_feature_cols(constants.FEATURE_COLS))
    sequence_features_example_spec[constants.LABEL_COL] = tf.io.FixedLenSequenceFeature([], dtype=tf.float32)

    def parse_fn(serialized_example):
        parsed_context_feature, parsed_sequence_feature = tf.io.parse_single_sequence_example(
            serialized=serialized_example,
            sequence_features=sequence_features_example_spec
        )
        labels = parsed_sequence_feature.pop(constants.LABEL_COL)
        return parsed_sequence_feature, labels

    dataset = dataset.interleave(lambda x:
                                 tf.data.TFRecordDataset(x, num_parallel_reads=constants.TF_RECORD_P_NUM_READS)
                                 .map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE),
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if config['train']['shuffle_repeat']:
        return dataset.cache() \
            .shuffle(data_util.calc_ds_shuffle_buffer_size(ds_len), reshuffle_each_iteration=True) \
            .repeat() \
            .batch(batch_size=config['train']['batch_size'], drop_remainder=True) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        return dataset.cache() \
            .batch(batch_size=config['train']['batch_size'], drop_remainder=True) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


@click.command()
@click.argument('input_data_path', envvar='INPUT_DATA_PATH', type=click.Path(exists=True))
@click.argument('train_pred', envvar='TRAIN_PREDICATE')
def main(input_data_path, train_pred, data_type='price', exchange='bn'):
    # find train data from the given input path and extract data and symbol
    parent_dir = os.path.join(input_data_path, constants.TRAIN_CHANNEL, data_type, exchange)
    search_pattern = os.path.join(parent_dir, '*')
    dirs = glob.glob(search_pattern)
    if len(dirs) > 0:
        dates = [directory.rsplit(misc_util.get_dir_sep(), 1)[-1] for directory in dirs]
        dates = train_util.get_valid_training_dates(dates=dates, how=train_pred)
        _logger.info(f'Found {len(dates)} date(s) to process - {[os.path.join(parent_dir, d) for d in dates]}')
        for date in dates:
            search_symbols = os.path.join(parent_dir, date, '*.csv')
            symbol_paths = glob.glob(search_symbols)
            _logger.info(f'processing date: {date}, found {len(symbol_paths)} symbols to train - {symbol_paths}')
            for symbol_path in symbol_paths:
                symbol = symbol_path.rsplit(misc_util.get_dir_sep(), 1)[-1].split('.')[0]
                _logger.info(f'Proceessing input data for date={date}, symbol={symbol}')
                start(data_type, exchange, date, symbol)
    else:
        _logger.info(f'found no datasets @ {parent_dir}')
        return


def optimizer_fn(config):
    if config['model']['optimizer_func'] == constants.OPTIMIZER_FUNC_ADAGRAD:
        """
        Adagrad

        Default optimizer function for tensorflow_estimator.python.estimator.canned.rnn.RNNEstimator 
            ```
                _DEFAULT_LEARNING_RATE = 0.05
                _DEFAULT_CLIP_NORM = 5.0
            ```
        """
        return tf.keras.optimizers.Adagrad(learning_rate=config['model']['learning_rate'],
                                           initial_accumulator_value=0.1,
                                           epsilon=1e-7,
                                           clipnorm=5.0)

    if config['model']['optimizer_func'] == constants.OPTIMIZER_FUNC_RMSPROP:
        return tf.keras.optimizers.RMSprop(clipvalue=1.0,
                                           learning_rate=config['model']['learning_rate'],
                                           rho=0.9,
                                           momentum=0.0,
                                           epsilon=1e-7,
                                           centered=False,
                                           name="RMSprop")
    elif config['model']['optimizer_func'] == constants.OPTIMIZER_FUNC_ADAM:
        return tf.keras.optimizers.Adam(learning_rate=config['model']['learning_rate'],
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=1e-7,
                                        amsgrad=False,
                                        name='Adam')
    else:
        raise ValueError(f'unexpected optimizer_func in model_hyperparameters={config["model"]["optimizer_func"]}')


@timeit
def predict(estimator, file_path, ds_len, config):
    """
    redict_keys: list of `str`, name of the keys to predict. It is used if
        the `tf.estimator.EstimatorSpec.predictions` is a `dict`. If
        `predict_keys` is used then rest of the predictions will be filtered
        from the dictionary. If `None`, returns all.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the prediction call.
      checkpoint_path: Path of a specific checkpoint to predict. If `None`, the
        latest checkpoint in `model_dir` is used.  If there are no checkpoints
        in `model_dir`, prediction is run with newly initialized `Variables`
        instead of ones restored from checkpoint.
      yield_single_examples: If `False`, yields the whole batch as returned by
        the `model_fn` instead of decomposing the batch into individual
        elements. This is useful if `model_fn` returns some tensors whose first
        dimension is not equal to the batch size.

    Yields:
      Evaluated values of `predictions` tensors.

    Raises:
      ValueError: If batch length of predictions is not the same and
        `yield_single_examples` is `True`.
      ValueError: If there is a conflict between `predict_keys` and
        `predictions`. For example if `predict_keys` is not `None` but
        `tf.estimator.EstimatorSpec.predictions` is not a `dict`.
    """
    num_epoch = config['train']['num_epoch']
    batch_size = config['train']['batch_size']
    predict_steps = train_util.calc_max_steps(ds_len, num_epoch, batch_size)
    _logger.info(f'predicting for {predict_steps} steps from {file_path}')
    predictions = estimator.predict(input_fn=lambda: input_fn(file_path, ds_len, config),
                                    predict_keys=None,
                                    hooks=None,
                                    checkpoint_path=None,
                                    yield_single_examples=True)
    _logger.info(f'prediction complete')
    return estimator


def rnn_cell_fn():
    config = data_util.load_hyperparameters_config()
    lstm_cell_hidden1 = tf.keras.layers.LSTMCell(units=config['model']['hidden_1_units'],
                                                 kernel_initializer='glorot_uniform',
                                                 recurrent_initializer='orthogonal',
                                                 bias_initializer='zeros',
                                                 unit_forget_bias=True,
                                                 kernel_regularizer=None,
                                                 recurrent_regularizer=None,
                                                 bias_regularizer=None,
                                                 kernel_constraint=None,
                                                 recurrent_constraint=None,
                                                 bias_constraint=None,
                                                 dropout=config['model']['hidden_1_dropout'],
                                                 implementation=2)

    lstm_cell_hidden2 = tf.keras.layers.LSTMCell(units=config['model']['hidden_2_units'],
                                                 kernel_initializer='glorot_uniform',
                                                 recurrent_initializer='orthogonal',
                                                 bias_initializer='zeros',
                                                 unit_forget_bias=True,
                                                 kernel_regularizer=None,
                                                 recurrent_regularizer=None,
                                                 bias_regularizer=None,
                                                 kernel_constraint=None,
                                                 recurrent_constraint=None,
                                                 bias_constraint=None,
                                                 dropout=config['model']['hidden_2_dropout'],
                                                 implementation=2)
    return tf.keras.layers.StackedRNNCells([lstm_cell_hidden1, lstm_cell_hidden2])


def start(data_type, exchange, date, symbol):
    # load hyper parameters
    config = data_util.load_hyperparameters_config()
    _logger.info(f'model hyper parameters: {config["model"]}')
    _logger.info(f'train hyper parameters: {config["train"]}')

    # set global tf seed to ensure reproducability
    tf.random.set_seed(constants.TF_GLOBAL_RANDOM_SEED)

    # build an estimator
    model_dir = train_util.get_model_dir(symbol)
    estimator = build_estimator(model_dir, config)

    # training
    train_file_path, train_ds_len = data_util.calc_ds_path_len(constants.TRAIN_CHANNEL, data_type, exchange, date,
                                                               symbol, config)
    estimator = train(estimator, train_file_path, train_ds_len, config)

    # evaluation
    eval_file_path, eval_ds_len = data_util.calc_ds_path_len(constants.EVAL_CHANNEL, data_type, exchange, date, symbol,
                                                             config)
    estimator = evaluate(estimator, eval_file_path, eval_ds_len, config)

    # prediction
    test_file_path, test_ds_len = data_util.calc_ds_path_len(constants.TEST_CHANNEL, data_type, exchange, date, symbol,
                                                             config)
    predict(estimator, test_file_path, test_ds_len, config)

    # export model
    export_model(estimator, symbol)


@timeit
def train(estimator, file_path, ds_len, config):
    # calculate training steps using provided hyper parameters
    batch_size = config['train']['batch_size']
    num_epoch = config['train']['num_epoch']

    train_data_length = ds_len
    train_max_steps = train_util.calc_max_steps(train_data_length, num_epoch, batch_size)

    """
     hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the training loop.
      steps: Number of steps for which to train the model. If `None`, train
        forever or train until `input_fn` generates the `tf.errors.OutOfRange`
        error or `StopIteration` exception. `steps` works incrementally. If you
        call two times `train(steps=10)` then training occurs in total 20 steps.
        If `OutOfRange` or `StopIteration` occurs in the middle, training stops
        before 20 steps. If you don't want to have incremental behavior please
        set `max_steps` instead. If set, `max_steps` must be `None`.
      max_steps: Number of total steps for which to train model. If `None`,
        train forever or train until `input_fn` generates the
        `tf.errors.OutOfRange` error or `StopIteration` exception. If set,
        `steps` must be `None`. If `OutOfRange` or `StopIteration` occurs in the
        middle, training stops before `max_steps` steps. Two calls to
        `train(steps=100)` means 200 training iterations. On the other hand, two
        calls to `train(max_steps=100)` means that the second call will not do
        any iteration since first call did all 100 steps.
      saving_listeners: list of `CheckpointSaverListener` objects. Used for
        callbacks that run immediately before or after checkpoint savings.

    Returns:
      `self`, for chaining.
    # """
    _logger.info(f'training for {train_max_steps} max_steps from {file_path}')
    estimator.train(input_fn=lambda: input_fn(file_path, ds_len, config),
                    max_steps=constants.DEBUG_STEPS if misc_util.is_debug() else train_max_steps,
                    saving_listeners=None)
    _logger.info(f'training completed')
    return estimator


# find .env automagically by walking up directories until it's found, then load up the .env entries as env vars
if __name__ == '__main__':
    """
    ***IMPORTANT***: NEVER commit your .dotenv file into the source repository. 
     Prereq: create .env configuration file at your project home directory. 
        <.env> file_sample:
        
        List of parameters required:
        ***   
        # Tensorflow memory configuration
        TF_GPU_MEMORY_LIMIT=<your gpu memory limit>
        # Train configurations
        TRAIN_PREDICATE=today
        
        # Local configurations
        MODEL_PATH=<...>\ml\model
        INPUT_PATH=<...>\ml\input\data\processed
        INPUT_DATA_PATH=<...>\ml\input\data
        
        OUTPUT_PATH==<...>\ml\output
        OUTPUT_DATA_PATH==<...>\ml\output\data
        
        INPUT_CONFIG_PATH==<...>\ml\input\config
        
        AWS_SECRET=<your_aws_secret>
        ***
     (https://click.palletsprojects.com/en/7.x/arguments/#file-arguments)
     runs defined function as command line command (ex. python make_dataset.py)
    """
    # load dot env
    data_util.find_load_dotenv()

    # configure gpu memory growth
    configure_tf_memory_growth(True)
    main()
