########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   Last edited: 05/31/20
#
#   Original from: https://github.com/aws/sagemaker-tensorflow-container
#
#   Refactored out to minimal tensorflow Dockerfile for the following specifications:
#
#       - Nvidia Cuda DNN 10.1
#       - Ubuntu 18.04
#       - Tensorflow 2.1
#
#   Requires nvidia-docker installed at your unix os
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################

FROM nvidia/cuda:10.1-base-ubuntu18.04

MAINTAINER Jae Lim <jae.lim@luckma.io>

# prevent stopping by user interaction
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# See https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html
# Intel MKL environment variables, redundant since training on GPU
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV KMP_BLOCKTIME=1
ENV KMP_SETTINGS=0

RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
    apt-utils \
    python3-dev \
    python3-pip \
    python3-setuptools \
    ca-certificates \
    cuda-command-line-tools-10-1 \
    cuda-cudart-dev-10-1 \
    cuda-cufft-dev-10-1 \
    cuda-curand-dev-10-1 \
    cuda-cusolver-dev-10-1 \
    cuda-cusparse-dev-10-1 \
    curl \
    libcudnn7=7.6.2.24-1+cuda10.1 \
    # TensorFlow doesn't require libnccl anymore but Open MPI still depends on it
    libnccl2=2.4.7-1+cuda10.1 \
    libgomp1 \
    libnccl-dev=2.4.7-1+cuda10.1 \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libpng-dev \
    libzmq3-dev \
    git \
    wget \
    vim \
    build-essential \
    openssh-client \
    openssh-server \
    zlib1g-dev \
    # Install dependent library for OpenCV
    libgtk2.0-dev \
    #cuda-cublas-dev not available with 10-1, install libcublas instead
    #it will downgrade the cublas from 10-2 to 10-1
    #adding an extra flag --allow-downgrades for it
    && apt-get update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated --allow-downgrades \
    libcublas10=10.1.0.105-1 \
    libcublas-dev=10.1.0.105-1 \
    # The 'apt-get install' of nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0
    # adds a new list which contains libnvinfer library, so it needs another
    # 'apt-get update' to retrieve that list before it can actually install the
    # library.
    # We don't install libnvinfer-dev since we don't need to build against TensorRT,
    # and libnvinfer4 doesn't contain libnvinfer.a static library.
    # nvinfer-runtime-trt-repo doesn't have a 1804-cuda10.1 version yet. see:
    # https://developer.download.nvidia.cn/compute/machine-learning/repos/ubuntu1804/x86_64/
 && apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated  \
    nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 \
 && apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated  \
    libnvinfer6=6.0.1-1+cuda10.1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/luckma
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
COPY ../../../../../../requirements.txt .

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python \
 && ln -s $(which ${PIP}) /usr/bin/pip

ARG TF_URL=https://tensorflow-aws.s3-us-west-2.amazonaws.com/2.1/AmazonLinux/gpu/final/tensorflow_gpu-2.1.0-cp36-cp36m-manylinux2010_x86_64.whl

RUN ${PIP} install -r /usr/luckma/requirements.txt
RUN ${PIP} install --no-cache-dir -U \
    ${TF_URL}

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/nvidia/bin:$PATH
