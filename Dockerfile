# Use PyTorch as base image with CUDA support
FROM --platform=$BUILDPLATFORM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and model files
COPY src/ ./src/
COPY models/ ./models/

# Production stage
FROM --platform=$TARGETPLATFORM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS runner

WORKDIR /app

# Copy from builder
COPY --from=builder /app /app

# Set environment variables
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API
CMD ["python", "src/main.py"] 
