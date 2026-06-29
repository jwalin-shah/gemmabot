FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY robot_video/ ./robot_video/
COPY examples/ ./examples/
COPY .env.example ./

# Install Python deps
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8002
EXPOSE 8003

# Run server
CMD ["uvicorn", "src.web.robosuite_server:app", "--host", "0.0.0.0", "--port", "8002"]
