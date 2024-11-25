# Use a Python base image. Replace '3.11-slim' with your actual Python version if different.
FROM python:3.11.6-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache for dependency installation
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# No default command or entrypoint