# Dockerfile for the api

# Use an official Python runtime as a parent image
FROM python:3.10-buster

# Set the working directory in the container
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Copy the current directory contents into the container at /app
COPY src /app/src
COPY tasks /app/tasks

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
  pip install -r requirements.txt && \
  pip install .  

# Invoke the training model and evaluation
RUN invoke run

# Make port 80 availableZ to the world outside this container
EXPOSE 8000

#Invoke the server 
CMD [ "invoke", "serve" ]



