# CNN-PELSVAE

This repository contains the code for the CNN-PELSVAE project. The project involves building a Convolutional Neural Network with a Probabilistic Embedding Learning Shared Variational Autoencoder (CNN-PELSVAE) model.
## Docker Instructions
To run the CNN-PELSVAE model, we provide a Dockerfile that simplifies the setup process. Please follow these steps:

### Build the Docker Image:
Use the following command to build the Docker image:

docker build -t cnnpelsvae .

This command will create a Docker image named cnnpelsvae based on the instructions provided in the Dockerfile.

### Run the Docker Container:

Once the Docker image is built, you can run the CNN-PELSVAE model in a container using the following command:

docker run cnnpelsvae .

This will start the model within the Docker container, and it will execute the necessary processes for the CNN-PELSVAE.

Make sure you have Docker installed on your machine before executing these commands. The Docker image contains all the required dependencies, so you don't need to worry about installing them separately.

For additional details on the CNN-PELSVAE model and how to use it, please refer to the documentation or code in this repository.
