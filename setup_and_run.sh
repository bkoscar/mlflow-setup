#!/bin/bash

# Script Name: setup_and_run.sh
# Description: This script creates necessary directories, runs Docker Compose in detached mode, and notifies the user.

# Create necessary directories
echo "Creating required directories..."

# Define paths
SRC_DIR="./src"
MLRUNS_DIR="${SRC_DIR}/mlruns"
MODELS_DIR="${SRC_DIR}/models"

# Create the 'src' directory if it does not exist
if [ ! -d "$SRC_DIR" ]; then
    mkdir "$SRC_DIR"
    echo "'src' directory created."
else
    echo "'src' directory already exists."
fi

# Create the 'src/mlruns' directory if it does not exist
if [ ! -d "$MLRUNS_DIR" ]; then
    mkdir "$MLRUNS_DIR"
    echo "'mlruns' directory inside 'src' created."
else
    echo "'mlruns' directory inside 'src' already exists."
fi

# Create the 'src/models' directory if it does not exist
if [ ! -d "$MODELS_DIR" ]; then
    mkdir "$MODELS_DIR"
    echo "'models' directory inside 'src' created."
else
    echo "'models' directory inside 'src' already exists."
fi

# Run Docker Compose in detached mode
echo "Starting Docker Compose..."
docker-compose up --build -d

# Notify the user
echo "Docker Compose is running in detached mode."
echo "You can visit the MLflow UI at: http://localhost:5001"

# Completion message
echo "Script executed successfully."
