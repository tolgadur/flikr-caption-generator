# Use PyTorch as base image
FROM --platform=linux/arm64 pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/models

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the application code and models
COPY src/ ./src/
COPY models/ ./models/

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["python", "src/main.py"] 
