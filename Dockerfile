# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Maximilian Herbert Dressler"
LABEL description="Surgical Instrument Segmentation Pipeline"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install package
RUN pip install --upgrade pip && \
    pip install -e .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs

# Default command
CMD ["python", "-c", "import surgical_segmentation; print(f'v{surgical_segmentation.__version__}')"]
