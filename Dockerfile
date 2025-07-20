# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV
# This is the key fix for the "libGL.so.1" error
RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg libsm6 libxext6

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Make port 7860 available, as expected by Hugging Face Spaces
EXPOSE 7860

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "app:app"]
