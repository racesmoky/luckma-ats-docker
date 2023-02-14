#!/usr/bin/env bash

########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   Last edited: 06/02/20
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################
if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

# OS CPU ARCHITECTURE
CPU_ARCH="x86_64"

# SOFTWARE PACKAGE VERSIONS
ANACONDA3_VERSION="2020.02"
HADOOP_VERSION="2.7"
JAVA_VERSION="8"
NVIDIA_CUDA_VERSION="10-1"
NVIDIA_UBUNTU_DIST="ubuntu18.04"
NVIDIA_UBUNTU_REPO="ubuntu1804"

# Generic python 3 preferred since it gets latest stable
PYTHON_VERSION="3"
SPARK_VERSION="2.4.5"

# prevent stopping by user interaction
EXPORT DEBIAN_FRONTEND noninteractive
EXPORT DEBCONF_NONINTERACTIVE_SEEN true

# See https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html
# Intel MKL environment variables, redundant since training on GPU
EXPORT KMP_AFFINITY=granularity=fine,compact,1,0
EXPORT KMP_BLOCKTIME=1
EXPORT KMP_SETTINGS=0

EXPORT LANG=C.UTF-8
EXPORT LC_ALL=C.UTF-8

EXPORT PIP_DISABLE_PIP_VERSION_CHECK 1
EXPORT PYTHONDONTWRITEBYTECODE=1

# http://blog.stuart.axelbrooke.com/python-3-on-spark-return-of-the-pythonhashseed
EXPORT PYTHONHASHSEED 0
EXPORT PYTHONIOENCODING=UTF-8
EXPORT PYTHONUNBUFFERED=1


########################################################################################################################
#                                          prerequisite packages
########################################################################################################################
# docker prerequisite packages
apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
    apt-utils \
    apt-transport-https \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg-agent \
    linux-headers-"$(uname -r)" \
    openssh-client \
    openssh-server \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-pip \
    python${PYTHON_VERSION}-setuptools \
    software-properties-common \
    systemd \
    vim \
    wget \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

########################################################################################################################
#                                                DOCKER
########################################################################################################################
# uninstall old docker versions
apt-get remove docker docker-engine docker.io containerd runc

# add-apt-repository:: docker gpg
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# install docker engine
apt-get update && apt-get  install -y --no-install-recommends --allow-unauthenticated \
    docker-ce \
    docker-ce-cli \
    containerd.io \
 && rm -rf /var/lib/apt/lists/*

########################################################################################################################
#                                                NVIDIA DRIVER
########################################################################################################################
# first get the PPA repository driver
add-apt-repository ppa:graphics-drivers/ppa
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/${NVIDIA_UBUNTU_REPO}/${CPU_ARCH}/7fa2af80.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/${NVIDIA_UBUNTU_REPO}/${CPU_ARCH}/" | tee /etc/apt/sources.list.d/cuda.list

# installing CUDA
apt-get update -o Dpkg::Options::="--force-overwrite" && apt-get install -y --no-install-recommends --allow-unauthenticated \
    cuda-${NVIDIA_CUDA_VERSION} \
    cuda-drivers \
 && rm -rf /var/lib/apt/lists/*

# set LD_LIBRARY_PATH and update PATH
echo "EXPORT LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc
echo "EXPORT PATH=$PATH:/usr/local/nvidia/bin" >> /etc/bash.bashrc

########################################################################################################################
#                                                NVIDIA DOCKER
########################################################################################################################
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/${NVIDIA_UBUNTU_DIST}/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get  install -y --no-install-recommends --allow-unauthenticated \
    nvidia-container-toolkit \
 && rm -rf /var/lib/apt/lists/*
systemctl restart docker

########################################################################################################################
#                                                JAVA
########################################################################################################################
echo oracle-java${JAVA_VERSION}-installer shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
add-apt-repository -y ppa:webupd8team/java && \
apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
    oracle-java${JAVA_VERSION}-installer \
 && rm -rf /var/lib/apt/lists/*
rm -rf /var/cache/oracle-jdk${JAVA_VERSION}-installer

# JAVA_HOME
EXPORT JAVA_HOME /usr/lib/jvm/java-${JAVA_VERSION}-oracle

# SPARK_HOME
echo "EXPORT JAVA_HOME=/usr/lib/jvm/java-${JAVA_VERSION}-oracle" >> /etc/bash.bashrc
echo "EXPORT PATH=$PATH:$JAVA_HOME/bin" >> /etc/bash.bashrc

########################################################################################################################
#                                                SPARK
########################################################################################################################
mkdir /usr/local/spark
curl -s https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz | tar -xz -C /usr/local/spark

# SPARK_HOME
echo "EXPORT SPARK_HOME=/usr/local/spark/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" >> /etc/bash.bashrc
echo "EXPORT PATH=$PATH:$SPARK_HOME/bin" >> /etc/bash.bashrc

########################################################################################################################
#                                               ANACONDA3
########################################################################################################################
mkdir /usr/local/anaconda
ANACONDA3_INSTALL_SCRIPT="Anaconda3-${ANACONDA3_VERSION}-Linux-${CPU_ARCH}.sh"
ANACONDA3_HOME="/usr/local/anaconda3/${ANACONDA3_VERSION}"
curl -Ok https://repo.continuum.io/archive/${ANACONDA3_INSTALL_SCRIPT}
bash ${ANACONDA3_INSTALL_SCRIPT} -b -p ${ANACONDA3_HOME}
rm ${ANACONDA3_INSTALL_SCRIPT}

# ANACONDA_HOME
echo "EXPORT ANACONDA_HOME=${ANACONDA3_HOME}" >> /etc/bash.bashrc
echo "EXPORT PATH=$PATH:${ANACONDA3_HOME}/bin" >> /etc/bash.bashrc

########################################################################################################################
#                                              Update PYTHON AND PIP
########################################################################################################################
PYTHON=python${PYTHON_VERSION}
PIP=pip${PYTHON_VERSION}
${PIP} --no-cache-dir install --upgrade pip setuptools
ln -s "$(which ${PYTHON})" /usr/local/bin/python && ln -s "$(which ${PIP})" /usr/bin/pip

########################################################################################################################
#                       Create Conda environment from environment.yml & install python packages
########################################################################################################################
conda env create -f ../../environment.yml
