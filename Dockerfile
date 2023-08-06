# Use the official PyTorch Docker image as the base
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . /app
