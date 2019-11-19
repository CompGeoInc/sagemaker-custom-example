# Sagemaker Custom Algorithm Example

This repo demonstrates a simple linear regression algorithm in Amazon SageMaker.

Its purpose is to show how to build a [custom SageMaker algorithm container](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html) from scratch that obeys the SageMaker container contract.

It also contains some example data to simulate a SageMaker training job locally.  All the data under `ml` is simply to provide a local testing example, it is not part of the algorithm.

## Build

`docker build -t linear .`

## Train

Train the model using local data to simulate the SageMaker environment:

`docker run -v $(pwd)/ml:/opt/ml -it linear train`

## Test

Test the model you built in the previous step.  In real use with SageMaker this would be done by the `serve` command.  Building proper `serve` support into this example is a TODO - I've added my own `test` command just to demonstrate locally that the model was built properly.

The correct answer is 31 :)

`docker run -v $(pwd)/ml:/opt/ml -it linear test`

# Notes

* Note that SageMaker will deliver all hyperparameters as strings so you must convert them to integer.

# Important Paths

* Hyperparameters JSON file: `/opt/ml/input/config/hyperparameters.json`
* Default input data directory:  `/opt/ml/input/data/train/`
* Default training output (i.e. model) directory:  `/opt/ml/model/`
* Error message output file:  `/opt/ml/output/failure`

# TODO

* Demonstrate `serve` functionality
