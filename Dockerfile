# Use official Python runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 
    PYTHONDONTWRITEBYTECODE=1 
    PORT=8000

# Create app directory
WORKDIR /app

# Install system dependencies (for numpy, pillow, torch, etc.)
RUN apt-get update && apt-get install -y 
    build-essential 
    git 
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Start with Gunicorn + Uvicorn workers
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8000"]
