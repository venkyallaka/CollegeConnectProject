FROM python:3.10-slim

# Prevent creation of .pyc files and ensure stdout/stderr are not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system packages that some Python libraries need. Adjust if unnecessary.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy pinned requirements which includes a CPU PyTorch wheel and then installs the repo's requirements
COPY requirements-pinned.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-pinned.txt

# Copy the application code
COPY . .

# Expose the port used by Uvicorn
EXPOSE 8000

# Default command (can be overridden by the host platform)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
