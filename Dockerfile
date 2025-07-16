FROM python:3.10-slim

# Prevent .pyc files and enable real-time logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy only required files (CHANGE 1)
COPY . /app/

# Set environment variable for GCP auth (used by google-cloud-storage)
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/nlpcoursework.json

# Install system dependencies (CHANGE 2: no bloat)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Flask app port
EXPOSE 8080

# Start the application
CMD ["python", "app.py"]