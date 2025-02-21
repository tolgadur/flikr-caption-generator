# Use PyTorch as base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/models

# Set working directory
WORKDIR /app

# Install system dependencies - rarely changes
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/models

# Copy requirements first and install dependencies - changes occasionally
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the model file - large but changes rarely
COPY models/model.pth /app/models/model.pth

# Copy the application code - changes frequently, keep this last
COPY src/ ./src/

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["python", "src/main.py"] 
