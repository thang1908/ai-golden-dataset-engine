# Golden Dataset Harness

AI-powered annotation pipeline for generating high-quality golden datasets from person images (CCTV/person search scenarios).

## Architecture

```
Image Dataset → Data Loader → Annotation Orchestrator (LangGraph)
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              Caption Agent   Attribute Agent  Grounding Agent
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                            Consensus Engine
                                    │
                            Quality Judge
                                    │
                          Confidence Scoring
                                    │
                        ┌───────────┴───────────┐
                   Auto Accept            Human Review
                        │                       │
                        └───────────┬───────────┘
                                    │
                          Golden Dataset Storage
```

## Quick Start

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e ".[dev]"
```

### 2. Run with Mock VLM (No GPU Required)

```bash
# Process images in a directory
python -m golden_dataset_harness.workflow.runner \
    --input-dir ./sample_images/ \
    --output-dir ./output/

# Output: ./output/dataset.jsonl
```

### 3. Run the API Server

```bash
uvicorn golden_dataset_harness.api.main:app --reload --port 8000
```

Then open http://localhost:8000/docs for the interactive API.

### 4. Run with Docker Compose

```bash
cd golden_dataset_harness
docker-compose up --build
```

This starts:
- **App** on port 8000 (FastAPI)
- **PostgreSQL** on port 5432
- **MinIO** on ports 9000 (API) / 9001 (Console)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/annotate` | Annotate a single image |
| `POST` | `/annotate/batch` | Annotate multiple images |
| `GET` | `/annotations/{image_id}` | Get annotation by ID |
| `GET` | `/annotations?status=pending_review` | Query by status |
| `GET` | `/export/label-studio` | Export for Label Studio |
| `POST` | `/review/{image_id}` | Submit human review |
| `GET` | `/stats` | Pipeline statistics |

## Configuration

### Model Provider

Edit `configs/settings.yaml`:

```yaml
model:
  provider: "mock"        # Options: mock, qwen-vl, internvl
  api_base_url: "http://localhost:8001/v1"
  model_name: "Qwen2.5-VL-7B-Instruct"
```

### Attribute Taxonomy

Edit `configs/attributes.yaml` to add/remove attributes and allowed values.

### Confidence Threshold

```yaml
confidence:
  judge_weight: 0.4
  consensus_weight: 0.3
  grounding_weight: 0.3
  acceptance_threshold: 0.75
```

## Swapping Models

The system uses an abstract `BaseVisionLanguageModel` interface. To add a new model:

1. Create a new file in `models/` (e.g., `models/my_model.py`)
2. Implement `BaseVisionLanguageModel` with methods:
   - `generate_caption(image, prompt)`
   - `extract_attributes(image, taxonomy)`
   - `verify_claim(image, claim)`
   - `judge_quality(image, caption, attributes)`
3. Register in `models/factory.py`
4. Set `provider: "my-model"` in `settings.yaml`

## Output Format

Each annotation in `dataset.jsonl`:

```json
{
  "image_id": "person_001",
  "caption": "A woman wearing a black jacket and carrying a backpack.",
  "attributes": {
    "gender": "female",
    "upper_clothing": "jacket",
    "upper_color": "black",
    "bag": "backpack",
    "hair": "long_hair"
  },
  "confidence": 0.87,
  "consensus_score": 0.85,
  "grounding_score": 0.90,
  "judge_score": 0.88,
  "review_status": "auto_accepted",
  "model_version": "mock-vlm-v1"
}
```

## Human Review (Label Studio)

Low-confidence annotations are exported for human review:

```bash
curl http://localhost:8000/export/label-studio | jq .
```

Import the JSON output into Label Studio for efficient review.

## Project Structure

```
golden_dataset_harness/
├── agents/                 # LangGraph pipeline nodes
│   ├── caption_agent.py    # Multi-caption generation
│   ├── attribute_agent.py  # Structured attribute extraction
│   ├── grounding_agent.py  # Hallucination detection
│   ├── judge_agent.py      # Quality evaluation
│   ├── consensus.py        # Semantic clustering & voting
│   └── confidence.py       # Score fusion & routing
├── api/                    # FastAPI HTTP interface
│   ├── main.py             # Endpoints
│   └── label_studio.py     # Label Studio integration
├── configs/                # YAML configuration
│   ├── attributes.yaml     # Attribute taxonomy
│   └── settings.yaml       # Pipeline settings
├── evaluation/             # Metrics
│   └── metrics.py
├── models/                 # VLM abstraction layer
│   ├── base.py             # Abstract interface
│   ├── mock_vlm.py         # Mock (testing)
│   ├── qwen_vl.py          # Qwen2.5-VL
│   ├── internvl.py         # InternVL
│   └── factory.py          # Model registry
├── schemas/                # Pydantic data contracts
│   ├── annotation.py       # Domain models
│   └── state.py            # LangGraph state
├── storage/                # Persistence layer
│   ├── postgres.py         # PostgreSQL ORM
│   ├── minio_client.py     # S3/MinIO abstraction
│   └── dataset_writer.py   # Output writer
├── workflow/               # Pipeline orchestration
│   ├── graph.py            # LangGraph StateGraph
│   └── runner.py           # Batch runner
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Development

```bash
# Run tests
pytest tests/ -v

# Type checking
mypy golden_dataset_harness/

# Lint
ruff check golden_dataset_harness/
```
