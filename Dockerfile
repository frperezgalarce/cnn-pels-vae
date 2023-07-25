# Use the official Python image as the base image
#FROM python:3.7
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . /app

# Expose a port (replace 8000 with the appropriate port number)
EXPOSE 8000

# Define the command to run your application
CMD ["python", "main.py"]
