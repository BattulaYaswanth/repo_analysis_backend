# Use official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# 1. Install system dependencies (Required for GitPython)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy your application code
COPY . .

# 4. Expose the port (Render uses 10000, HF uses 7860)
# We will use an environment variable for flexibility
ENV PORT=1000

# 5. Run the application
# We use --host 0.0.0.0 so external traffic can reach it
CMD uvicorn main:app --host 0.0.0.0 --port $PORT