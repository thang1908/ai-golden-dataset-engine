# ============================================================================
# Golden Dataset Harness — Dockerfile
# ============================================================================
# Multi-stage build: install dependencies in a builder stage, then copy
# only the venv to a slim runtime image.
# ============================================================================

# --- Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN pip install --no-cache-dir --upgrade pip hatchling

# Copy project files (context is golden_dataset_harness/)
COPY . /build/golden_dataset_harness/

# Build wheel and install into a venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir /build/golden_dataset_harness/

# --- Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Copy application code and configs into package directory
COPY . /app/golden_dataset_harness/

# Expose API port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "golden_dataset_harness.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
