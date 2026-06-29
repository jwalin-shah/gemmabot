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
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn[standard] sse-starlette jinja2 aiofiles

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "src.web.server:app", "--host", "0.0.0.0", "--port", "8000"]
