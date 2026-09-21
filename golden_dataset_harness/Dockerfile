FROM python:3.11-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip setuptools

COPY pyproject.toml ./
COPY golden_dataset_harness ./golden_dataset_harness
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir /build

FROM python:3.11-slim AS runtime

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY golden_dataset_harness ./golden_dataset_harness

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8000
CMD ["uvicorn", "golden_dataset_harness.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
