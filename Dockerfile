# --- STAGE 1: Builder ---
FROM python:3.11-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies (Compilers/Headers)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into a local folder to keep the final image clean
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# --- STAGE 2: Final Production Image ---
FROM python:3.11-slim AS runner

# 1. Standard Python Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/home/appuser/.local/bin:$PATH

# 2. FIX: AI & Library Cache Redirection
# Redirects all library-specific cache/data folders to writable app directories.
# This prevents 'Permission Denied' errors for the non-root appuser.
ENV MPLCONFIGDIR=/app/.matplotlib
ENV NLTK_DATA=/app/nltk_data
ENV FONTCONFIG_PATH=/app/.fontconfig
ENV HF_HOME=/app/.cache
ENV XDG_CACHE_HOME=/app/.cache
ENV ONNXRUNTIME_CACHE_DIR=/app/.cache

# Create a restricted non-root system user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -s /sbin/nologin appuser

WORKDIR /app

# Install runtime-only dependencies (The "Headless AI Suite")
# These are essential for PDF processing, threading, and OpenGL support.
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    fontconfig \
    libxcb1 \
    libx11-6 \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    libsm6 \
    libice6 \
    libgl1 \
    libglx0 \
    && rm -rf /var/lib/apt/lists/*

# Transfer the installed dependencies from the builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Verification Step: Ensures binaries were transferred correctly
RUN ls /home/appuser/.local/bin && echo "Build-time Check: Dependencies found in local/bin"

# Copy the application source code
COPY . .

# 3. FIX: Permissions Management
# Pre-create all required hidden/data folders and hand over ownership to appuser.
RUN mkdir -p uploads vector_store nltk_data .matplotlib .fontconfig .cache && \
    chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Automated Healthcheck to monitor system availability
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Launch with 4 workers to handle concurrent RAG requests in production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]