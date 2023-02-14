::########################################################################################################################
::#   Proprietary and confidential.
::#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
::#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
::#
::#   IMPORTANT:
::#     - Make sure your docker is configured with enough memory and your cpu architecture supports intel MKL instructions
::#     - The memory cap is defined at devel-mkl.Dockerfile::BAZEL_BUILD_RAM_RESOURCES, must be set prior execution
::#     - Make sure you are logged into DockerHub before executing this script.
::#     - Script automatically pushes the built docker images into the docker hub repository
::#     - If not logged in, the images will still get built. However, you will get an error message at the end.
::#
::#   Last edited: 05/29/20
::#
::#   @author Jae Lim, <jae.lim@luckma.io>
::########################################################################################################################

@echo off

:: Ensure BUILD_DIR is passed
if [%1]==[] goto usage

:: Ensure BUILD_DIR exists
if not exist %1\* @echo BUILD_DIR=%1 doesn't exist && goto :eof

:: Ensure DOCKER_USER is passed
if [%2]==[] goto usage

:: Ensure DOCKER_REPO is passed
if [%3]==[] goto usage

:: Docker configurations
set DOCKER_USER=%2
set DOCKER_REPO=%3

:: Docker image tags
set TF_SERVING_GIT_BRANCH=r2.1
set TF_SERVING_BUILD_IMAGE_DEVEL_CPU_MKL=%DOCKER_USER%/%DOCKER_REPO%:tensorflow-serving-devel-%TF_SERVING_GIT_BRANCH%-cpu-mkl_
set TF_SERVING_BUILD_IMAGE_CPU_MKL=%DOCKER_USER%/%DOCKER_REPO%:tensorflow-serving-%TF_SERVING_GIT_BRANCH%-cpu-mkl_

:: Dockerfile configurations
set TF_SERVING_DOCKERFILE_BASE=%1
set TF_SERVING_DOCKERFILE_DEVEL_MKL=devel-mkl.Dockerfile
set TF_SERVING_DOCKERFILE_MKL=mkl.Dockerfile
set TF_SERVING_DOCKERFILE_DEVEL_MKL_PATH=%TF_SERVING_DOCKERFILE_BASE%\%TF_SERVING_DOCKERFILE_DEVEL_MKL%
set TF_SERVING_DOCKERFILE_MKL_PATH=%TF_SERVING_DOCKERFILE_BASE%\%TF_SERVING_DOCKERFILE_MKL%

:: Ensure devel-mkl.Dockerfile exists
if not exist %TF_SERVING_DOCKERFILE_DEVEL_MKL_PATH% @echo Failed to build_and_push, BUILD_DIR=%TF_SERVING_DOCKERFILE_DEVEL_MKL_PATH% doesn't exist && goto :eof

:: Ensure mkl.Dockerfile exists
if not exist %TF_SERVING_DOCKERFILE_MKL_PATH% @echo Failed to build_and_push, BUILD_DIR=%TF_SERVING_DOCKERFILE_MKL_PATH% doesn't exist && goto :eof

setlocal enableDelayedExpansion

:: Verify system docker installation
@echo Checking system requirement: docker
where docker > NUL 2>&1 && set "DOCKER_REQUIREMENT_SATISFIED=true"
IF not "%DOCKER_REQUIREMENT_SATISFIED%"=="true" (
  echo Download and install docker from https://www.docker.com/
  goto :eof
)
@echo System requirement satisfied: docker, BASE_DIR=%TF_SERVING_DOCKERFILE_BASE% DOCKER_USER=%DOCKER_USER%, DOCKER_REPO=%DOCKER_REPO%

:build_tf_serving_devel
    :: go to serving directory
    @echo building %TF_SERVING_BUILD_IMAGE_DEVEL_CPU_MKL% using bazel with Dockerfile: %TF_SERVING_DOCKERFILE_DEVEL_MKL_PATH%
    docker build --pull -t %TF_SERVING_BUILD_IMAGE_DEVEL_CPU_MKL% -f %TF_SERVING_DOCKERFILE_DEVEL_MKL_PATH% . && set "TF_SERVING_DEVEL_BUILT=true"
    IF not "%TF_SERVING_DEVEL_BUILT%"=="true" (
      @echo Failed to build %TF_SERVING_BUILD_IMAGE_DEVEL_CPU_MKL%. Exiting...
      goto :eof
    )
    @echo building %TF_SERVING_BUILD_IMAGE_DEVEL_CPU_MKL% was successful

    goto build_tf_serving

:build_tf_serving
    @echo building %TF_SERVING_BUILD_IMAGE_CPU_MKL% with Dockerfile: %TF_SERVING_DOCKERFILE_MKL_PATH%
    docker build -t %TF_SERVING_BUILD_IMAGE_CPU_MKL% --build-arg TF_SERVING_BUILD_IMAGE=%TF_SERVING_BUILD_IMAGE_DEVEL_CPU_MKL% -f %TF_SERVING_DOCKERFILE_MKL_PATH% . && set "TF_SERVING_BUILT=true"
    IF not "%TF_SERVING_BUILT%"=="true" (
      @echo Failed to build %TF_SERVING_BUILD_IMAGE_CPU_MKL%. Exiting...
      goto :eof
    )
    @echo building %TF_SERVING_BUILD_IMAGE_CPU_MKL% was successful

    goto :push_docker_images

:push_docker_images
    :: Push tensorflow-serving-devel image to docker hub
    @echo pushing %TF_SERVING_BUILD_IMAGE_DEVEL_CPU_MKL% to dockerhub repository: %DOCKER_USER%/%DOCKER_REPO%
    docker push %TF_SERVING_BUILD_IMAGE_DEVEL_CPU_MKL%
    @echo pushing %TF_SERVING_BUILD_IMAGE_DEVEL_CPU_MKL% was successful

    :: Push tensorflow-serving image to docker pub
    @echo pushing %TF_SERVING_BUILD_IMAGE_CPU_MKL% to dockerhub repository: %DOCKER_USER%/%DOCKER_REPO%
    docker push %TF_SERVING_BUILD_IMAGE_CPU_MKL%
    @echo pushing %TF_SERVING_BUILD_IMAGE_CPU_MKL% was successful

    goto :eof

:usage
    @echo Usage: ^<build_dir>^ ^<docker_user^> ^<docker_repo^>
exit /B 1