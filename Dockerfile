FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Expose ports for API and UI
EXPOSE 8000
EXPOSE 8501

# Run as non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Default command launches the API
CMD ["axon-api"]
