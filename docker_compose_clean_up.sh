#!/bin/bash

echo "🚀 Starting deep cleanup for $(basename $PWD)..."

# 1. Stop and remove project-specific containers, networks, images, and volumes
docker-compose down --volumes --rmi all --remove-orphans

# 2. Remove any remaining 'dangling' images (the <none>:<none> ones)
echo "🧹 Removing dangling images..."
docker image prune -f

# 3. Remove unused networks
echo "🌐 Cleaning up unused networks..."
docker network prune -f

# 4. Optional: Global cleanup (active and inactive)
# Uncomment the line below if you want to wipe EVERYTHING on your system
docker system prune -a --volumes -f

echo "✨ System is now clean."
