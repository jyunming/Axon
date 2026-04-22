FROM python:3.11-slim

WORKDIR /app

# Install OS deps needed for common python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Copy full repository into image (build context is repo root)
COPY . /app

# Install Python deps
RUN python3 -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi && \
    pip install --no-cache-dir -e . || true

# Ensure runner script is executable
RUN chmod +x /app/docker/two_user_test/runner.sh || true

EXPOSE 8000 8001

# Default to a shell; services override the command
CMD ["bash"]
