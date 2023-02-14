########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   https://aws.amazon.com/releasenotes/available-deep-learning-containers-images/
#   tensorFlow 2.1 | training | gpu | 3.6 (py36)
#
#   763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.1.0-gpu-py27-cu101-ubuntu18.04
#
#   You must login to access to the Deep Learning Containers image repository before pulling the image.
#   Specify your region and its corresponding ECR Registry from the previous table in the following command:
#   --------------------------------------------------------------------------------------------------------------------
#       aws ecr get-login-password --region us-east-1 |
#       docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
#   --------------------------------------------------------------------------------------------------------------------

#   You can then pull these Docker images from Amazon ECR by running:
#   --------------------------------------------------------------------------------------------------------------------
#       docker pull tensorflow-training:2.1.0-gpu-py36-cu101-ubuntu18.04
#   --------------------------------------------------------------------------------------------------------------------
#
#   Last edited: 05/31/20
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################
FROM tensorflow-training:2.1.0-gpu-py36-cu101-ubuntu18.04

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY ../../../../../code /opt/ml/code
COPY ../../../../../input /opt/input
WORKDIR /opt/ml/code