FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# 2. Remove default nginx config to prevent port conflicts
RUN rm /etc/nginx/sites-enabled/default

# 3. Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. App code
COPY . .

# 5. Configs
COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENV PORT=10000
EXPOSE 10000

# 6. Run supervisor with explicit config path
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
