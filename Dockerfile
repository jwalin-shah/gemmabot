FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml requirements.txt uv.lock ./
COPY src/ ./src/
COPY robot_video/ ./robot_video/
COPY examples/ ./examples/
COPY .env.example ./

# Install Python deps
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "src.web.server:app", "--host", "0.0.0.0", "--port", "8000"]
