########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   Last edited: 05/29/20
#
#   Original from: https://github.com/tensorflow/serving/tree/r2.1/tensorflow_serving/tools/docker/Dockerfile.mkl
#
#   Configured the following from original:
#       - Dockerfile.mkl -> mkl.Dockerile
#       - Adjusted build image pulling is from tensorflow/serving rather than from --build_args
#       - ARG TF_SERVING_VERSION_GIT_BRANCH=master -> ARG TF_SERVING_VERSION_GIT_BRANCH=r2.1
#       - ENV TENSORFLOW_INTRA_OP_PARALLELISM=2 -> ENV TENSORFLOW_INTRA_OP_PARALLELISM=8
#       - ENV TENSORFLOW_INTER_OP_PARALLELISM=2 -> ENV TENSORFLOW_INTRA_OP_PARALLELISM=8
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################

# Parse TF_SERVING_BUILD_IMAGE from --build_args, fails without arg assignment
ARG TF_SERVING_BUILD_IMAGE=${TF_SERVING_BUILD_IMAGE}
FROM ${TF_SERVING_BUILD_IMAGE} as build_image
FROM ubuntu:18.04

MAINTAINER Jae Lim <jae.lim@luckma.io>

ARG TF_SERVING_VERSION_GIT_BRANCH=r2.1
ARG TF_SERVING_VERSION_GIT_COMMIT=head
ENV TF_SERVING_GRPC_PORT=8500
ENV TF_SERVING_REST_PORT=8501

LABEL tensorflow_serving_github_branchtag=${TF_SERVING_VERSION_GIT_BRANCH}
LABEL tensorflow_serving_github_commit=${TF_SERVING_VERSION_GIT_COMMIT}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install TF Serving pkg
COPY --from=build_image /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# Install MKL libraries
COPY --from=build_image /usr/local/lib/libiomp5.so /usr/local/lib
COPY --from=build_image /usr/local/lib/libmklml_gnu.so /usr/local/lib
COPY --from=build_image /usr/local/lib/libmklml_intel.so /usr/local/lib

ENV LIBRARY_PATH '/usr/local/lib:$LIBRARY_PATH'
ENV LD_LIBRARY_PATH '/usr/local/lib:$LD_LIBRARY_PATH'

# Expose gRPC ports
EXPOSE ${TF_SERVING_GRPC_PORT}

# Expose REST ports
EXPOSE ${TF_SERVING_REST_PORT}

# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models
RUN mkdir -p ${MODEL_BASE_PATH}

# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=model

################################ KML Optimization ################################
# https://www.tensorflow.org/guide/performance/overview#tuning_mkl_for_the_best_performance
# https://github.com/tensorflow/tensorflow/commit/d1823e2e966e96ee4ea7baa202ad9f292ac7427b
# https://ark.intel.com/content/www/us/en/ark/products/88196/intel-core-i7-6700-processor-8m-cache-up-to-4-00-ghz.html
#
# Run the following unix command to see your cpu architecture and configure below
#   > lscpu
#
# Architecture:        x86_64
# CPU op-mode(s):      32-bit, 64-bit
# Byte Order:          Little Endian
# CPU(s):              8            (aka logical processors = physical cores * num_threads_per_core)
# On-line CPU(s) list: 0-7
# Thread(s) per core:  2
# Core(s) per socket:  4            (aka physical cores)
# Socket(s):           1
# Vendor ID:           GenuineIntel
# CPU family:          6
# Model:               94
# Model name:          Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
#
###################################################################################

# Verbose for KML message
ENV MKLDNN_VERBOSE=1

# Recommended settings for OMP_NUM_THREADS = num physical cores
ENV OMP_NUM_THREADS=4

# Recommended settings for non-CNN→ KMP_BLOCKTIME=1 (user should verify empirically)
ENV KMP_BLOCKTIME=1

# Recommended settings for KMP_SETTINGS = 1 (enable)
ENV KMP_SETTINGS=1

# Recommended settings for KMP_AFFINITY=granularity=fine,verbose,compact,1,0
ENV KMP_AFFINITY='granularity=fine,verbose,compact,1,0'

# Recommended settings for TENSORFLOW_INTRA_OP_PARALLELISM=#No.of Physical cores
ENV TENSORFLOW_INTRA_OP_PARALLELISM=4

# Recommended settings for TENSORFLOW_INTER_OP_PARALLELISM=#No.of Sockets
ENV TENSORFLOW_INTER_OP_PARALLELISM=1

# Create a script that runs the model server so we can use environment variables
# while also passing in arguments from the docker command line
RUN echo '#!/bin/bash \n\n\
tensorflow_model_server --port=${TF_SERVING_GRPC_PORT} --rest_api_port=${TF_SERVING_REST_PORT} \
--tensorflow_intra_op_parallelism=${TENSORFLOW_INTRA_OP_PARALLELISM} \
--tensorflow_inter_op_parallelism=${TENSORFLOW_INTER_OP_PARALLELISM} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
