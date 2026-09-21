"""FastAPI gateway for the platform's OAuth-protected OpenAI-compatible vLLM APIs.

Run from the repository root:
    uvicorn golden_dataset_harness.api.vllm_gateway:app --reload --port 8001

All endpoints obtain and refresh the OAuth token internally. Clients never need
to receive ``VLLM_CLIENT_SECRET`` or the access token.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from golden_dataset_harness.models.oauth import OAuthTokenProvider
from golden_dataset_harness.models.provider_settings import ProviderSettings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = ProviderSettings()
    client = httpx.AsyncClient(timeout=settings.vllm_timeout_seconds)
    app.state.settings = settings
    app.state.client = client
    app.state.tokens = OAuthTokenProvider(settings, client)
    yield
    await client.aclose()


app = FastAPI(
    title="vLLM API Gateway",
    description="Internal FastAPI wrapper for the OAuth-protected vLLM platform.",
    version="1.0.0",
    lifespan=lifespan,
)


def _url(path: str) -> str:
    return f"{app.state.settings.service_base_url}{path}"


def _prepare_payload(payload: dict[str, Any], *, stream: bool | None = None) -> dict[str, Any]:
    """Apply platform defaults and flatten the OpenAI SDK's ``extra_body`` field."""
    body = dict(payload)
    body.setdefault("model", app.state.settings.vllm_model)
    if stream is not None:
        body["stream"] = stream
    extra_body = body.pop("extra_body", None)
    if extra_body is not None:
        if not isinstance(extra_body, dict):
            raise HTTPException(422, "extra_body must be a JSON object")
        body.update(extra_body)
    return body


async def _request(
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    data: dict[str, str] | None = None,
    files: dict[str, Any] | None = None,
) -> httpx.Response:
    """Send an authenticated request and refresh the token once on HTTP 401."""
    for retry in range(2):
        token = await app.state.tokens.get_token(force_refresh=retry == 1)
        response = await app.state.client.request(
            method,
            _url(path),
            headers={"Authorization": f"Bearer {token}"},
            json=json,
            data=data,
            files=files,
        )
        if response.status_code != 401 or retry == 1:
            return response
        await app.state.tokens.invalidate()
    raise AssertionError("unreachable")


def _json_response(response: httpx.Response) -> Any:
    if response.status_code >= 400:
        raise HTTPException(response.status_code, response.text)
    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(502, "vLLM returned invalid JSON") from exc


@app.get("/health")
async def health() -> dict[str, str]:
    """Confirm that environment configuration was loaded without exposing secrets."""
    return {"status": "ok", "model": app.state.settings.vllm_model}


@app.get("/vllm/models")
async def list_models() -> Any:
    return _json_response(await _request("GET", "/v1/models"))


@app.post("/vllm/chat")
async def chat(payload: dict[str, Any] = Body(...)) -> Any:
    """Chat Completions, including thinking, guardrail, vision and JSON Schema."""
    if "messages" not in payload:
        raise HTTPException(422, "messages is required")
    return _json_response(
        await _request("POST", "/v1/chat/completions", json=_prepare_payload(payload, stream=False))
    )


@app.post("/vllm/chat/stream")
async def chat_stream(payload: dict[str, Any] = Body(...)) -> StreamingResponse:
    """Stream Chat Completions as the upstream server-sent-event response."""
    if "messages" not in payload:
        raise HTTPException(422, "messages is required")
    body = _prepare_payload(payload, stream=True)

    async def stream() -> AsyncIterator[bytes]:
        for retry in range(2):
            token = await app.state.tokens.get_token(force_refresh=retry == 1)
            request = app.state.client.build_request(
                "POST", _url("/v1/chat/completions"),
                headers={"Authorization": f"Bearer {token}"}, json=body,
            )
            response = await app.state.client.send(request, stream=True)
            if response.status_code == 401 and retry == 0:
                await response.aclose()
                await app.state.tokens.invalidate()
                continue
            if response.status_code >= 400:
                error = await response.aread()
                await response.aclose()
                raise HTTPException(response.status_code, error.decode(errors="replace"))
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
            return

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/vllm/responses")
async def responses_api(payload: dict[str, Any] = Body(...)) -> Any:
    return _json_response(await _request("POST", "/v1/responses", json=_prepare_payload(payload)))


@app.post("/vllm/embeddings")
async def embeddings(payload: dict[str, Any] = Body(...)) -> Any:
    if "input" not in payload:
        raise HTTPException(422, "input is required")
    return _json_response(await _request("POST", "/v1/embeddings", json=_prepare_payload(payload)))


@app.post("/vllm/rerank")
async def rerank(payload: dict[str, Any] = Body(...)) -> Any:
    if "query" not in payload or "documents" not in payload:
        raise HTTPException(422, "query and documents are required")
    response = await _request("POST", "/vinsoc/v1/rerank", json=_prepare_payload(payload))
    return _json_response(response)


@app.post("/vllm/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = "v-llm-v1-asr",
    language: str = "vi",
    response_format: str = "verbose_json",
) -> Any:
    content = await file.read()
    return _json_response(await _request(
        "POST",
        "/v1/audio/transcriptions",
        data={"model": model, "language": language, "response_format": response_format},
        files={"file": (file.filename or "audio.wav", content, file.content_type)},
    ))


@app.post("/vllm/files")
async def upload_file(file: UploadFile = File(...), purpose: str = "batch") -> Any:
    content = await file.read()
    return _json_response(await _request(
        "POST",
        "/v1/files",
        data={"purpose": purpose},
        files={"file": (file.filename or "upload.jsonl", content, file.content_type)},
    ))


@app.get("/vllm/files")
async def list_files() -> Any:
    return _json_response(await _request("GET", "/v1/files"))


@app.get("/vllm/files/{file_id}")
async def get_file(file_id: str) -> Any:
    return _json_response(await _request("GET", f"/v1/files/{file_id}"))


@app.get("/vllm/files/{file_id}/content")
async def download_file(file_id: str) -> Response:
    response = await _request("GET", f"/v1/files/{file_id}/content")
    if response.status_code >= 400:
        raise HTTPException(response.status_code, response.text)
    return Response(response.content, media_type=response.headers.get("content-type"))


@app.delete("/vllm/files/{file_id}")
async def delete_file(file_id: str) -> Any:
    return _json_response(await _request("DELETE", f"/v1/files/{file_id}"))


@app.post("/vllm/batches")
async def create_batch(payload: dict[str, Any] = Body(...)) -> Any:
    return _json_response(await _request("POST", "/v1/batches", json=payload))


@app.get("/vllm/batches/{batch_id}")
async def get_batch(batch_id: str) -> Any:
    return _json_response(await _request("GET", f"/v1/batches/{batch_id}"))


@app.post("/vllm/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str) -> Any:
    return _json_response(await _request("POST", f"/v1/batches/{batch_id}/cancel"))
