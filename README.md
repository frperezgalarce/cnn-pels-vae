# CNN-PELSVAE

This repository contains the code for the CNN-PELSVAE project. The project, which is focused on facing the data shift problem, involves building a Convolutional Neural Network classifier with a embebbed deep generative modeling of periodic variable stars. 


## Conda instructions

### Setting Up the Environment
To run the CNN-PELSVAE project using Conda, follow the steps below to create a Conda environment and activate it:

### Install Conda:
If you haven't installed Conda, download and install Anaconda or Miniconda from the official website: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

### Clone the Repository:
Clone this repository to your local machine using Git or download the repository as a ZIP file and extract it.

### Navigate to the Project Directory:
Open a terminal or Anaconda Prompt and change the current directory to the root of the CNN-PELSVAE project.

### Create a Conda Environment:
Run the following command to create a new Conda environment named cnnpelsvae and install the required packages listed in the requirements.txt file:

`conda create --name cnnpelsvae --file requirements.txt`


Activate the Conda Environment:
After the environment is created, activate it using the following command:

`conda activate cnnpelsvae`

Now that the Conda environment is set up and activated, you are ready to run the CNN-PELSVAE model:

### Execute the Main Script:
Use the python command to execute the main script (main.py or any other appropriate script) that contains the CNN-PELSVAE model and other relevant code:

`python main.py`

### Deactivate the Environment:
After you are done with the CNN-PELSVAE model execution, deactivate the Conda environment:

`conda deactivate`

By following these instructions, you can create an isolated Conda environment with the required dependencies to run the CNN-PELSVAE project. This ensures consistency and reproducibility across different environments.

## Docker Instructions
To run the CNN-PELSVAE model, we provide a Dockerfile that simplifies the setup process. Please follow these steps:

### Build the Docker Image:
Use the following command to build the Docker image:

`docker build -t cnnpelsvae .`

This command will create a Docker image named cnnpelsvae based on the instructions provided in the Dockerfile.

### Run the Docker Container:

Once the Docker image is built, you can run the CNN-PELSVAE model in a container using the following command:

`docker run -it --rm cnnpelsvae bash`

This will start the model within the Docker container, and it will execute the necessary processes for the CNN-PELSVAE.

Make sure you have Docker installed on your machine before executing these commands. The Docker image contains all the required dependencies, so you don't need to worry about installing them separately.

For additional details on the CNN-PELSVAE model and how to use it, please refer to the documentation or code in this repository.

`docker stop cnnpelsvae`
