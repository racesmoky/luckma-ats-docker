########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   Last edited: 05/13/20
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################

import os
import sys
import time

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import *

from luckma.ats.ml.data import constants
from luckma.ats.ml.data.denv_vars import denv_check
from luckma.ats.ml.data.pipeline.processed.data_processor_base import ProcessedDataProcessorBase
from luckma.ats.ml.data.utils import file_utils
from luckma.ats.ml.data.utils.data_util import timeit


class ProcessedDataProcessorTFRSpark(ProcessedDataProcessorBase):
    '''
        https://stackoverflow.com/questions/43532083/pyspark-import-user-defined-module-or-py-files

        spark-submit --master spark://<local_host>:7077 <optional: --packages ...><python_file>
        requires PYTHON_PATH to be set to your local project's home directory to look for packages

    '''

    @denv_check(denv_vars=['SPARK_MASTER_URL', 'SPARK_DRIVER_MEMORY', 'SPARK_EXECUTOR_MEMORY', 'SPARK_EXECUTOR_CORES'])
    def __init__(self, data_type, exchange, date, symbol):
        super().__init__(data_type, exchange, date, symbol)

    @property
    def get_file_type(self):
        return constants.INPUT_FILE_TYPE_TF_RECORD

    def save(self, df, save_to):
        '''
            Transforms given pandas dataframe into TFRecord by utilizing parallel write capability of Spark.

            Spark's context lifecycle should be consistent with this function. It's recommended to SparkSession
            to end with return statement and spark.streaming.stopGracefullyOnShutdown = True

        :param df: pandas dataframe
        :param save_to: location where TFRecord will be saved to
        :return: save_to
        '''

        # save df to the file system for data analysis / for direct read to spark dataframe
        save_dir = save_to.rsplit(file_utils.project_dir_sep(), 1)[0]
        file_utils.maybe_create_dir(save_dir)

        save_to_csv = save_to.replace(constants.INPUT_FILE_TYPE_TF_RECORD, constants.INPUT_FILE_TYPE_CSV)
        df.to_csv(save_to_csv, index=True, header=True)
        self.logger.debug(f'Prepared csv file for spark processing (len={len(df)}) to {save_to_csv}')

        # pass the saved data file path for direct spark dataframe read
        self.to_tf_record_spark(save_to_csv, save_to)

        return save_to

    @timeit
    def to_tf_record_spark(self, csv_file_path, save_to):
        '''
           Reads csv file from given path then produces serialized tensorflow.proto tfrecords using spark partitions.
           Partitioning serves the following purposes:

           1) Improves performance on multiple spark executioners to process RDDs in //
           2) Improves performance when processing dataset during input_fn by utilizing
              tensorflow's prefetch and interleave to reduce training time
           3) Eases management of large single dataset causes inefficiency on storage units

           Causes irrelevant WARN WindowExec: No Partition Defined for Window operation!
           Spark dataframe gets partitioned before write, and entire windowing operation takes less than 1s
           on large dataset. Can silence warning by dummy partitioning but it's rather redundant code

        :param csv_file_path: source
        :param save_to: destination
        :return: saved destination
        '''
        # Extract panda numpy series from pandas df, then create spark dataframe rows
        past_history = self.config[constants.PROCESSED_DATA_DIR_NAME]['past_history']
        future_target = self.config[constants.PROCESSED_DATA_DIR_NAME]['future_target']

        # start spark session
        spark = self.start_spark_session()

        # prepare spark dataframe directly from csv_file_path given
        try:
            sdf = spark.read.csv(csv_file_path,
                                 header=True,
                                 schema=self.build_spark_rdd_schema(),
                                 timestampFormat=constants.INDEX_COL_TS_FORMAT)
            self.logger.debug(f'original rdd schema: {sdf._jdf.schema().treeString()}')
        except FileNotFoundError:
            self.logger.error(f'Failed to read spark dataframe from {csv_file_path}')
            sys.exit(-1)

        # define window spec for feature cols and label
        feature_window_spec = Window.orderBy(constants.INDEX_COL).rowsBetween(-(past_history - 1), 0)
        label_window_spec = Window.orderBy(constants.INDEX_COL).rowsBetween(1, future_target)

        # define udf function for sequencing the entries to how Tensorflow proto expects [[x],[x]]
        to_seq_udf = F.udf(lambda seq: [[x] for x in seq], ArrayType(ArrayType(FloatType(), False)))

        # transform label
        sdf = sdf.withColumn(constants.LABEL_COL, F.collect_list(constants.LABEL_BASE_COL).over(label_window_spec)) \
            .withColumn(constants.LABEL_COL, to_seq_udf(constants.LABEL_COL))

        # transform features
        for feature_col in constants.FEATURE_COLS:
            sdf = sdf.withColumn(feature_col, F.collect_list(feature_col).over(feature_window_spec)) \
                .withColumn(feature_col, to_seq_udf(feature_col))

        # clean up dataframe for transformation
        row_num_col = 'row_num'
        sdf = sdf.withColumn(row_num_col, F.row_number().over(Window.orderBy(constants.INDEX_COL)))

        # filter out tail rows that doesn't have proper shape, label requires future values
        sdf = sdf.filter(sdf[row_num_col] <= sdf.count() - future_target)

        # filter out head rows that doesn't have proper shape, feature sequence requires previous values
        sdf = sdf.filter(past_history <= sdf[row_num_col])

        # drop unneeded cols to prevent from getting tranformed into tfrecord
        sdf = sdf.drop(constants.INDEX_COL).drop(row_num_col)

        self.logger.debug(f'transformed rdd schema: {sdf._jdf.schema().treeString()}')

        # write to tf record
        save_dir = save_to.rsplit(file_utils.project_dir_sep(), 1)[0]
        file_utils.maybe_create_dir(save_dir)

        self.logger.info(f'started tf_record transformation to {save_to}')
        start_time = time.time()

        # repartition spark dataframe to take advantage of // provided by spark
        sdf.repartition(constants.SPARK_PARTITION_NUMS).write.format(constants.INPUT_FILE_TYPE_TF_RECORD) \
            .option('recordType', constants.SPARK_CONFIG_TF_RECORD_TYPE) \
            .save(path=save_to, mode='overwrite')

        end_time = time.time()
        self.logger.info(f'Saved tf_record data to {save_to}, transform duration = {round(end_time - start_time, 2)}s')

        return save_to

    def load(self, load_from):
        '''
            Loads spark dataframe from the given path

        :param load_from:
        :return: spark_df
        '''
        # start spark session
        spark = self.start_spark_session()

        sdf = spark.read.format(constants.INPUT_FILE_TYPE_TF_RECORD) \
            .option("recordType", constants.SPARK_CONFIG_TF_RECORD_TYPE) \
            .load(load_from)
        self.logger.info(sdf.printSchema())
        return sdf

    def start_spark_session(self):
        """
            Expects the following entries in your .env file

            SPARK_MASTER_URL = url of your spark host (e.g. spark://127.0.0.1:7077)
                               spark doesn't like localhost and forces you to set environment variable
                               called SPARK_LOCAL_IP, just use ip in your configuration
            SPARK_DRIVER_MEMORY = amount of memory to allocate for spark driver (e.g. 4g or 1200m)
                                  in practice, spark driver memory is not used for processing.
                                  need just enough for driving RDDs within spark. This depends
                                  on the size of the processing dataset
            SPARK_EXECUTOR_MEMORY = amount of memory to allocate for spark executor (e.g. 4g or 1200m)
                                    memory used for processing. Higher the better.
            SPARK_EXECUTOR_CORES = number of spark cores to use for processing (e.g. 8)
            SPARK_TENSORFLOW_JAR_PATH = path where spark tensorflow jar is located
                                                    (e.g. <path>//spark-tensorflow-connector_2.11-1.15.0.jar)

            .config passes key,value pair into spark context configuration

            https://www.tutorialspoint.com/pyspark/pyspark_sparkcontext.htm
            Spark configurations:
                spark.master - local master cluster with 4 cores (i.e. .master('local[4]') or all .master('local[*]')
                spark.jars - configures extra jars to be consumed for the spark
                spark.streaming.stopGracefullyOnShutdown - default: False, gracefully shuts down spark context,
                                                           no need to call SparkSession.stop() when used

                The master is a Spark, Mesos or YARN cluster URL, or a special "local" string to run in local mode

            During tuning it's observed that partitioning the windows caused more processing resources and took
            longer time.
            The average processing time during benchmark was 7m with window partitioning size of 100 using large dataset
            whereas running single partition with single executor consistently yields par 3m processing time.

            This seems due to the fact the dataset gets expanded by multiples of history for every single example and
            the input data size is rather small for map and reduce to benefit from multiple clusters.

            Will revisit later if there is need to process this on multiple clusters.

        :return: initialized SparkSession with configurations above

        # Instead of initializing as SparkSession, might have to initialize as SparkContext(URI_master)
        i.e. spark://<master>. Creating a new spark session binds a new port with a separate UI
        """

        spark_session: pyspark.sql.SparkSession
        spark_tf_jar_path = file_utils.get_spark_tf_jar_path()
        try:
            self.logger.info('starting spark session...')
            spark_session = pyspark.sql.SparkSession \
                .builder \
                .appName(constants.SPARK_CONFIG_APP_NAME) \
                .config('spark.master', os.environ['SPARK_MASTER_URL']) \
                .config('spark.jars', spark_tf_jar_path) \
                .config('spark.streaming.stopGracefullyOnShutdown', True) \
                .config('spark.driver.memory', os.environ['SPARK_DRIVER_MEMORY']) \
                .config('spark.executor.memory', os.environ['SPARK_EXECUTOR_MEMORY']) \
                .config("spark.executor.cores", os.environ['SPARK_EXECUTOR_CORES']) \
                .getOrCreate()
        except:
            self.logger.error(f'Failed to create spark session with {spark_tf_jar_path}')
            sys.exit(1)
        self.logger.info(f'spark session initialized: {spark_session.sparkContext.getConf().getAll()}')
        return spark_session

    def build_spark_rdd_schema(self):
        all_fields = [StructField(constants.INDEX_COL, TimestampType())]
        feature_fields = [StructField(col, FloatType(), False) for col in constants.FEATURE_COLS]
        all_fields.extend(feature_fields)

        schema = StructType(all_fields)
        self.logger.debug(f'Generated spark rdd schema = {schema}')

        return schema
