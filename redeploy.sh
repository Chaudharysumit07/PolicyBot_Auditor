#!/bin/bash

# 1. Stop and remove the old container
echo "🛑 Stopping old container..."
docker stop policy-api || true
docker rm policy-api || true

# 2. Build the new image
echo "🏗️ Building new image..."
docker build -t policy-backend .

# 3. Run the new container with the OLLAMA_URL environment variable
echo "🚀 Starting new container..."
docker run -d \
  --name policy-api \
  -p 8000:8000 \
  -e OLLAMA_BASE_URL="http://host.docker.internal:11434" \
  --add-host=host.docker.internal:host-gateway \
  policy-backend

echo "📝 Following logs..."
docker logs -f policy-api