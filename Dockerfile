FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy all environment files
COPY . /app/env

WORKDIR /app/env

# Install Python dependencies directly via pip
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "beautifulsoup4>=4.12.0" \
    "requests>=2.31.0" \
    "openai>=1.0.0" \
    "fastapi>=0.111.0" \
    "uvicorn>=0.30.1" \
    "pydantic>=2.0.0"

# Set PYTHONPATH so server/ imports resolve from /app/env
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Expose port
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
