# Amazon SageMaker RL Containers


> [!WARNING]  
> As of April 2024, SageMaker RL containers no longer accepts new pull requests. Please follow [Building Your Image](#building-your-image) to build your own RL images.


A set of Dockerfiles that enables Reinforcement Learning (RL) solutions to be used in SageMaker.

The SageMaker team uses this repository to build its official RL images. On how to use any of these images on SageMaker,
see [Python SDK](https://github.com/aws/sagemaker-python-sdk).
For end users, this repository is typically of interest if you need implementation details of
the official image, or if you want to use it to build your own customized RL image.

For information on running RL jobs on SageMaker: [SageMaker RLEstimators](https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/rl).

For notebook examples: [SageMaker Notebook Examples](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/reinforcement_learning).


## Table of Contents

1. [Getting Started](#getting-started)
1. [RL Images Provided by SageMaker](#rl-images-provided-by-sagemaker)
1. [Building Your Image](#building-your-image)
1. [Running the Tests](#running-the-tests)


## Getting Started

### Prerequisites

Make sure you have installed all of the following prerequisites on your
development machine:

* [Docker](https://www.docker.com/)

#### For Testing on GPU

* [Nvidia-Docker2](https://github.com/NVIDIA/nvidia-docker>)

#### Recommended

A Python environment management tool (e.g. [PyEnv](https://github.com/pyenv/pyenv), [VirtualEnv](https://virtualenv.pypa.io/en/stable/).

### Terminologies

#### Toolkit

Toolkits are libraries that provide specific algorithms to train a Reinforcement Learning model. We currently provide Dockerfiles for these three toolkits:

* [Ray](https://github.com/ray-project/ray)
* [Coach](https://github.com/NervanaSystems/coach)
* [VW](https://github.com/VowpalWabbit/vowpal_wabbit)

#### Framework

Framework refers to a Deep Learning framework/library that a toolkit may need in order to train an algorithm. We use Sagemaker created framework images/prebuilt Amazon SageMaker Docker images as base images in a Toolkit's Dockerfile (whenever required). Currently we are using these two frameworks:

* TensorFlow (used for Ray and Coach)
* PyTorch (used for Ray)
* MXNet (used for Coach)

Note: VW doesn't require a framework


## RL Images Provided by SageMaker


### MXNet Coach Images:

* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-mxnet:coach0.11-cpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-mxnet:coach0.11.0-cpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-mxnet:coach0.11-gpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-mxnet:coach0.11.0-gpu-py3

### TensorFlow Coach Images:

* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:coach0.10-cpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:coach0.10.1-cpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:coach0.10-gpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:coach0.10.1-gpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:coach0.11-cpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:coach0.11.0-cpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:coach0.11.1-cpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:coach0.11-gpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:coach0.11.0-gpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:coach0.11.1-gpu-py3
* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-coach-container:coach-1.0.0-tf-cpu-py3
* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-coach-container:coach-1.0.0-tf-gpu-py3

### TensorFlow Ray Images:

* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:ray0.6-cpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:ray0.6.5-cpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:ray0.6-gpu-py3
* 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-tensorflow:ray0.6.5-gpu-py3
* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.2-tf-cpu-py36
* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.2-tf-gpu-py36
* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.5-tf-cpu-py36
* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.5-tf-gpu-py36
* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-ray-container:ray-1.6.0-tf-cpu-py37
* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-ray-container:ray-1.6.0-tf-gpu-py37

### PyTorch Ray Images:

* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.5-torch-cpu-py36
* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.5-torch-gpu-py36
* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-ray-container:ray-1.6.0-torch-cpu-py36
* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-ray-container:ray-1.6.0-torch-gpu-py36

### Vowpal Wabbit Images:

* 462105765813.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-vw-container:vw-8.7.0-cpu


[List of supported SageMaker regions](https://docs.aws.amazon.com/general/latest/gr/rande.html#sagemaker_region>).

## Building Your Image

[Amazon SageMaker](https://aws.amazon.com/documentation/sagemaker/) utilizes Docker containers to run all training jobs and inference endpoints.

The Docker images are built from the Dockerfiles specified in this repository at:

* [coach/docker](https://github.com/aws/sagemaker-rl-container/tree/master/coach/docker)
* [ray/docker](https://github.com/aws/sagemaker-rl-container/tree/master/ray/docker)
* [vw/docker](https://github.com/aws/sagemaker-rl-container/tree/master/vw/docker)

The Dockerfiles are grouped by RL toolkit and toolkit version. Within that, they are separated 
by framework (if needed). For e.g., the Dockerfile for Coach v0.11.0 with MXNet framework can be found at: `coach/docker/0.11.0/Dockerfile.mxnet`.


For toolkits Ray and Coach, the Dockerfiles use deep learning framework images provided by SageMaker as their "base" images.

These "base" images are specified with the following naming convention:

```
520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-<framework>:<framework_version>-<processor>-py3
```

* `<framework>` can be `tensorflow-scriptmode` (with `<framework_version>` `1.11.0` or higher depending on the toolkit requirements)
  or `mxnet` (with `<framework_version>` `1.3.0` or higher depending on the toolkit requirements);
* ``<processor>`` can be `cpu` or `gpu`;
* for valid `<region>` values please see `list of supported SageMaker regions <https://docs.aws.amazon.com/general/latest/gr/rande.html#sagemaker_region).

Before building images:

Pull deep learning framework "base" image, which require [Docker](https://www.docker.com/), [AWS credentials](https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html), and [AWS CLI](https://aws.amazon.com/cli/).

```
# Login into SageMaker ECR account
$(aws ecr get-login --no-include-email --region <region> --registry-ids 520713654638)
# Pull docker image from ECR
docker pull 520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-<framework>:<framework_version>-<processor>-py3
```

```
# Example

$(aws ecr get-login --no-include-email --region us-west-2 --registry-ids 520713654638)

# CPU TensorFlow image
docker pull 520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-scriptmode:1.11.0-cpu-py3

# GPU MXNet image
docker pull 520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:1.3.0-gpu-py3
```


To build RL Docker image:

```
# All build instructions assume you're building from the root directory of the sagemaker-rl-container.

# CPU
docker build -t <image_name>:<tag> -f <rl_toolkit>docker/<rl_toolkit_version>/Dockerfile.<framework> --build-arg processor=<cpu_or_gpu> .

# GPU
docker build -t <image_name>:<tag> -f <rl_toolkit>/docker/<rl_toolkit_version>/Dockerfile.<framework> --build-arg processor=<cpu_or_gpu> .
```

```
# Example

# Ray TensorFlow CPU
docker build -t tf-ray:0.6.5-cpu-py3 -f ray/docker/0.6.5/Dockerfile.tf --build-arg processor=cpu .

# Coach TensorFlow GPU
docker build -t tf-coach:0.11.0-gpu-py3 -f coach/docker/0.11.0/Dockerfile.tf --build-arg processor=gpu .

# Coach MXNet CPU
docker build -t mxnet-coach:0.11.0-cpu-py3 -f coach/docker/0.11.0/Dockerfile.mxnet --build-arg processor=cpu .

# VW CPU
docker build -t vw:8.7.0-cpu -f vw/docker/8.7.0/Dockerfile .
```



## Running the Tests
Running the tests requires installation of test dependencies.

```
git clone https://github.com/aws/sagemaker-rl-container.git
cd sagemaker-rl-container
pip install .
```


Tests are defined in [test/](https://github.com/aws/sagemaker-rl-container/tree/master/test) and include local integration and SageMaker integration tests.


### Local Integration Tests

Running local integration tests require [Docker](https://www.docker.com/) and [AWS credentials](https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html), as the local integration tests make calls to a couple of AWS services. The local integration tests and SageMaker integration tests require configurations specified within their respective [conftest.py](https://github.com/aws/sagemaker-rl-container/tree/master/test/conftest.py).

Local integration tests on GPU require [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker).

Before running local integration tests:

1. Build your Docker image.
1. Pass in the correct pytest arguments to run tests against your Docker image.

If you want to run local integration tests, then use:

```
# Required arguments for integration tests are found in test/conftest.py
pytest test/integration/local --toolkit <toolkit_to_run_tests_for> \
                              --docker-base-name <your_docker_image> \
                             --tag <your_docker_image_tag> \
                              --processor <cpu_or_gpu>
```

```
# Example
pytest test/integration/local --toolkit coach \
                              --docker-base-name custom-rl-coach-image \
                              --tag 1.0 \
                              --processor cpu
```

### SageMaker Integration Tests


SageMaker integration tests require your Docker image to be within an [Amazon ECR repository](https://docs
.aws.amazon.com/AmazonECS/latest/developerguide/ECS_Console_Repositories.html).

The Docker base name is your [ECR repository namespace](https://docs.aws.amazon
.com/AmazonECR/latest/userguide/Repositories.html).

The instance type is your specified [Amazon SageMaker Instance Type](https://aws.amazon.com/sagemaker/pricing/instance-types/) that the SageMaker integration test will run on.

Before running SageMaker integration tests:

1. Build your Docker image.
1. Push the image to your ECR repository.
1. Pass in the correct pytest arguments to run tests on SageMaker against the image within your ECR repository.

If you want to run a SageMaker integration end to end test on [Amazon SageMaker](https://aws.amazon.com/sagemaker/), then use:

```
# Required arguments for integration tests are found in test/conftest.py
pytest test/integration/sagemaker --toolkit <toolkit_to_run_tests_for> \
                                  --aws-id <your_aws_id> \
                                  --docker-base-name <your_docker_image> \
                                  --instance-type <amazon_sagemaker_instance_type> \
                                  --tag <your_docker_image_tag> \
```

```
# Example
pytest test/integration/sagemaker --toolkit coach \
                                  --aws-id 12345678910 \
                                  --docker-base-name custom-rl-coach-image \
                                  --instance-type ml.m4.xlarge \
                                  --tag 1.0
```


## Contributing

Please read [CONTRIBUTING.md](https://github.com/aws/sagemaker-rl-container/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This library is licensed under the Apache 2.0 License. 

Note: Specific license for Toolkits/Frameworks, if any, can be found in <toolkit>/docker/LICENSE or in the Framework's image
