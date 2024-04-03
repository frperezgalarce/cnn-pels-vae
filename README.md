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
