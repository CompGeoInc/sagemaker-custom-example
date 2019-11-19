# Sagemaker Custom Algorithm Example

This repo demonstrates a simple linear regression algorithm in Amazon SageMaker.

Its purpose is to show how to build a [custom SageMaker algorithm container](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html) from scratch that obeys the SageMaker container contract.

The overall contract is as follows:

* Set your Dockerfile ENTRYPOINT to be your script or binary
* In your script, check argv[1] for the SageMaker command
* Load the hyperparameters.json file
* Load input data file(s) from the /opt/ml/input/data directory (this directory is automatically pulled from S3 by SageMaker before it runs your container)
* Run your algorithm, compute your output (in an ML situation the output would be a model, but SageMaker doesn't tell you what your output should be - it can be anything)
* Save all your output files to /opt/ml/model directory (after your algo has run, SageMaker will automatically compress this directory into a tarball and upload to S3)
* If there's an error, save the error message to the file /opt/ml/output/failure.
* Exit with zero for success non-zero for error

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

* Note that you don't need to use Python and TensorFlow to use SageMaker, this is just an example.  Your Docker container can be any framework or language as long as it obeys the SageMaker commands and loads/saves data to the correct places.
* Note that SageMaker will deliver all hyperparameters as strings so you must convert them to integer.

# Important Paths

* Hyperparameters JSON file: `/opt/ml/input/config/hyperparameters.json`
* Default input data directory:  `/opt/ml/input/data/train/`
* Default training output (i.e. model) directory:  `/opt/ml/model/`
* Error message output file:  `/opt/ml/output/failure`

# TODO

* Demonstrate `serve` functionality
