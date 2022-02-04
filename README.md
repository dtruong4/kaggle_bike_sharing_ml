# Training a Sagemaker Model to Predict Bike Sharing Demand

In this project, we will use AWS Sagemaker to train a pretrained model that can perform image classification by using various practices and tools in Sagemaker and PyTorch:
* Sagemaker profiling
* Sagemaker debugger
* Hyperparameter tuning/optimization (HPO)
* Integration with S3

For this project, we will be using this [dataset of dog images](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).

## Project Specifications
The notebook and endpoint were run in an AWS `ml.t3.medium` instance, with the notebook additionally using `Python 3 (MXNet 1.8 Python 3.7 CPU Optimized)`.

## Key Files
* `project_notebook.ipynb` - The main notebook used to run commands and interact with SageMaker.
* `README.md` - Contains key information about set up and post-project analysis.
* `project_report.md` - Discusses the results of the project, as well as analyzes the data.

## Project Set Up and Installation
1. Enter AWS through the gateway in the course and open SageMaker Studio.
2. Download the starter notebook file.
3. Proceed by running the cells once in consecutive order.
4. If the images are not rendering, or you need a closer look, you can check out all of them in the `./img` folder.

## Sources:
* Template code and dataset images from Udacity's AWS course
