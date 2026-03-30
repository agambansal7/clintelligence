# TrialIntel Dockerfile
# Multi-stage build for smaller production image

# ==============================================================================
# Build stage
# ==============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Production stage
# ==============================================================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="TrialIntel Team <support@trialintel.com>"
LABEL description="Clinical Trial Intelligence Platform"
LABEL version="1.0.0"

# Create non-root user for security
RUN groupadd -r trialintel && useradd -r -g trialintel trialintel

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=trialintel:trialintel . .

# Create data directory
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R trialintel:trialintel /app/data /app/models /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    TRIALINTEL_DATA_DIR=/app/data \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Switch to non-root user
USER trialintel

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/api/v1/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Default command - start API server
CMD ["python", "run.py", "api"]

# ==============================================================================
# Dashboard variant
# ==============================================================================
FROM production as dashboard

# Override command for dashboard
CMD ["python", "run.py", "dashboard"]
