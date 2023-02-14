########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   Last edited: 06/01/20
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################
import glob
import os
import sys
import traceback

import ruamel.yaml as ruamel
from google.protobuf import text_format
from google.protobuf.text_format import ParseError
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from tensorflow_serving.config import model_server_config_pb2

from ml.code import constants
from ml.code.util import data_util, logging_util, misc_util

__author__ = 'jae.lim@luckma.io (Jae Lim)'
_logger = logging_util.get_logger()

"""Generates tensorflow model serve configuration files for linux docker deployment

    Configuration files generated:
        - <project_home>/model_config.config
        - <project_home>/docker-compose.yml

    Prerequisites:
        .env with the following entries defined:
            - MODEL_EXPORT_PATH: model (saved_model.pb) artifacts base directory. 
                                 Expects MODEL_EXPORT_PATH/<symbol>/<time_stamp>/saved_model.pb

    By default, configuration files will be saved to project's home directory
"""


class ModelServeConfigGenerator:
    def __init__(self):
        data_util.find_load_dotenv()
        self.config = data_util.load_build_config()['localhost']['tensorflow-serving']

    def generate_configs(self):
        """
            Configurations generated:
                - <project_home>/model_config.config
                - <project_home>/docker-compose.yml
        :return: None
        """
        model_export_dir = os.environ[constants.MODEL_EXPORT_PATH]

        # create symbol path dictionary
        symbol_path_dic = self._create_symbol_path_dic(model_export_dir)

        # generate model_config.config
        model_config_path = self._create_model_config(symbol_path_dic.keys(),
                                                      os.path.join(constants.PROJECT_HOME_PATH,
                                                                   self.config['ModelConfigFileName']))

        # generate docker-compose.yml
        return self._create_docker_compose_yml(model_config_path, symbol_path_dic,
                                               os.path.join(constants.PROJECT_HOME_PATH,
                                                            self.config['DockerComposeFileName']))

    @staticmethod
    def is_saved_model_present(symbol_path, model_name='saved_model.pb'):
        """
            Checks if saved_model.pb exists from given symbol_path

        :param model_name: name of the saved_model. In Tensorflow, default is saved_model.pb
        :param symbol_path: <model_export_dir>/<symbol>
        :return: True if saved_model.pb exists within <model_export_dir>/<symbol>, False otherwise
        """
        return True if len(glob.glob(os.path.join(symbol_path, '*', model_name))) > 0 else False

    @staticmethod
    def _create_symbol_path_dic(model_export_dir):
        """
            Creates symbol_path_dic { <symbol>:<symbol_path> } for given :param model_export_dir
            if the model_export_dir/<symbol> doesn't contain expected model_name (e.g. saved_model.pb)
            in the dir, it will be excluded from the serving pipeline.

            This is to prevent tensorflow_serving from serving the empty directory with no model file present.
            This occurs if the training loop terminated without saving the model via export_model method

        :param model_export_dir:
        :return: { <symbol>:<symbol_path> } for given :param model_export_dir
        """
        symbol_paths = glob.glob(os.path.join(model_export_dir, '*'))
        filtered_symbol_paths = filter(ModelServeConfigGenerator.is_saved_model_present, symbol_paths)
        return {ModelServeConfigGenerator.extract_symbol(symbol_path): symbol_path for symbol_path
                in filtered_symbol_paths}

    @staticmethod
    def extract_symbol(symbol_path):
        """
            Extracts symbol from given :param symbol_path

        :param symbol_path:
        :return: symbol
        """
        return symbol_path.rsplit(misc_util.get_dir_sep(), 1)[1]

    def _create_model_config(self, symbols, save_to):
        """
            Creates google.proto formatted model_config.config for all symbols defined at
                - .env::MODEL_EXPORT_PATH/<symbol>

            - model_config.config
                - model configurations for tensorflow_serving
                - passed through docker volumn defined at docker-compose.yml
                - docker-compose passes command via --model_config_file flag to tensorflow_model_server

        :param save_to: path where model_config.config will be saved to
        :param symbols: list(directory_name under .env::MODEL_EXPORT_PATH)
        :return: save_to
        """
        config_list = model_server_config_pb2.ModelConfigList()

        # Implicit model_config.model version_policy = latest (default)
        for symbol in symbols:
            model_config = config_list.config.add()
            model_config.name = symbol
            model_config.base_path = f"{self.config['DockerContainerModelDir']}{symbol}"
            model_config.model_platform = self.config['TFServingModelPlatform']

        # create model_server_config_pb2.ModelServerConfig text
        prefix = "model_config_list {"
        suffix = "}"
        model_config_text = f"{prefix}{config_list}{suffix}"
        model_server_config = model_server_config_pb2.ModelServerConfig()
        try:
            # Validates the model config using google.protobuf.text_format
            model_server_config = text_format.Parse(text=model_config_text, message=model_server_config)
        except ParseError:
            _logger.error(
                f'failed to model_config_text, model_config_text={model_config_text}, stack_trace='
                f'{traceback.print_exc()}')
            sys.exit(1)

        # save model_server_config_pb2.ModelServerConfig to specified save_to path
        try:
            with open(save_to, "w") as f:
                f.write(text_format.MessageToString(model_server_config, as_utf8=True))
        except IOError:
            _logger.error(f'failed to create_model_config, save_to={save_to}, stack_trace={traceback.print_exc()}')
            sys.exit(1)

        _logger.info(f"created {self.config['ModelConfigFileName']}.yml to {save_to}")
        return save_to

    def get_docker_compose_service_image_name(self):
        """
            Reads variables from .env and deployconfig.json to create image tag for docker-compose.yml

        :return: name of tensorflow serving image name used for docker compose
        """
        return f"{os.environ[constants.DOCKER_USER]}/{os.environ[constants.DOCKER_REPO]}:" \
               f"{self.config['DockerServiceName']}-{self.config['TFServingBranch']}-cpu-mkl"

    def _create_docker_compose_yml(self, model_config_path, symbol_path_dic, save_to,
                                   restart='unless-stopped'):
        """
            Creates docker-compose.yml for docker deployment.

        :param model_config_path: path where model_config.config was saved to
        :param symbol_path_dic: { <symbol>:<symbol_path> }
        :param save_to: destination where docker-compose.yml will be saved to
        :param restart: docker services::restart configuration
        :return: :param save_to
        """
        volumes = [f"{symbol_path_dic[symbol]}:{os.path.join(self.config['DockerContainerModelDir'], symbol)}" for
                   symbol in symbol_path_dic.keys()]
        volumes.append(
            f"{model_config_path}:{os.path.join(self.config['DockerContainerModelDir'], self.config['ModelConfigFileName'])}")
        ports = [self.config['TFServingGPCPortRange'], self.config['TFServingRestPortRange']]

        # https://docs.docker.com/compose/compose-file/
        docker_compose_dic = {"version": DoubleQuotedScalarString(self.config['DockerComposeVersion']),
                              "services": {
                                  f"{self.config['DockerServiceName']}": {
                                      "image": self.get_docker_compose_service_image_name(),
                                      "restart": restart,
                                      "ports": ports,
                                      "volumes": volumes,
                                      "command": f"--model_config_file="
                                                 f"{os.path.join(self.config['DockerContainerModelDir'], self.config['ModelConfigFileName'])}"
                                  }
                              }
                              }

        # save model_server_config_pb2.ModelServerConfig to specified save_to path
        try:
            with open(save_to, "w") as f:
                ruamel.YAML().dump(docker_compose_dic, f)
        except IOError:
            _logger.error(f'failed to create_model_config, save_to={save_to}, stack_trace={traceback.print_exc()}')
            sys.exit(1)

        _logger.info(f"created {self.config['DockerComposeFileName']} to {save_to}")
        return save_to


"""
    serve.py::main 
        - Verifies system requirements
            : docker installation, expected tensorflow-serving docker image exists in the system
        - Generates model_config.config for all the compiled model exists in .env::MODEL_EXPORT_DIR\<symbol>
        - Generates docker-compose.yml with above generated model_config.config
        - Runs docker-compose up
"""
if __name__ == '__main__':
    # make sure docker is installed in the system
    misc_util.verify_system_command('docker')

    # initialize ModelServeConfigGenerator and get docker image name
    model_serve_config_gen = ModelServeConfigGenerator()
    docker_image_name = model_serve_config_gen.get_docker_compose_service_image_name()

    # make sure docker image exists in the system
    misc_util.verify_docker_image(docker_image_name)

    # generate model service configs
    docker_compose_file_path = model_serve_config_gen.generate_configs()

    # run docker-compose
    misc_util.run_command(f'docker-compose -f {docker_compose_file_path} up')
