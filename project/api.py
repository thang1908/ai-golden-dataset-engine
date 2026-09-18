"""FastAPI transport. Set PERSON_QUERY_CHECKPOINT to a trained run or epoch bundle."""
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator
from .inference import QueryParser

logger = logging.getLogger(__name__)


class ParseRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: StrictStr = Field(min_length=1, max_length=2000)

    @field_validator('text')
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError('text must not be blank')
        return value


class UpperClothing(BaseModel):
    type: Literal['shirt', 't-shirt', 'jacket', 'hoodie', 'coat', 'dress', 'unknown']
    color: Literal['black', 'white', 'red', 'blue', 'green', 'yellow', 'gray', 'brown', 'unknown']


class Hair(BaseModel):
    color: Literal['black', 'brown', 'blonde', 'unknown']
    length: Literal['short', 'long'] | None = None


class Accessory(BaseModel):
    type: Literal['bag', 'backpack', 'glasses', 'hat', 'mask']


class ParseResponse(BaseModel):
    gender: Literal['male', 'female', 'unknown']
    upper_clothing: UpperClothing
    hair: Hair
    accessories: list[Accessory]


def error(status, code, message, details=None):
    return JSONResponse(status_code=status, content={'error': {
        'code': code, 'message': message, 'details': details or []}})


def create_app(parser=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.parser = parser
        if parser is None:
            checkpoint = os.getenv('PERSON_QUERY_CHECKPOINT')
            if checkpoint:
                try:
                    app.state.parser = QueryParser.from_checkpoint(checkpoint,
                        os.getenv('PERSON_QUERY_DEVICE', 'cpu'))
                except Exception as exc:
                    logger.error('Parser startup failed (%s): %s', type(exc).__name__, exc)
            else:
                logger.error('PERSON_QUERY_CHECKPOINT is not set. Set it to a trained run '
                             'directory or epoch bundle before starting the API.')
        yield
        app.state.parser = None

    app = FastAPI(title='Person Query Understanding', lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc):
        return error(422, 'invalid_request', 'Invalid request', [
            {'field': '.'.join(str(v) for v in item['loc']), 'reason': item['type']}
            for item in exc.errors()])

    @app.get('/health')
    def health(request: Request):
        ready = request.app.state.parser is not None
        return JSONResponse(status_code=200 if ready else 503,
                            content={'status': 'ready' if ready else 'unavailable'})

    @app.post('/parse_query', response_model=ParseResponse, response_model_exclude_none=True)
    def parse_query(body: ParseRequest, request: Request):
        active = request.app.state.parser
        if active is None:
            return error(503, 'model_unavailable', 'Parser is unavailable')
        try:
            return active.parse(body.text)
        except ValueError as exc:
            return error(422, 'invalid_request', str(exc))
        except Exception as exc:
            logger.error('Inference failed: %s', type(exc).__name__)
            return error(500, 'inference_failed', 'Unable to parse query')

    return app


app = create_app()
