class EvaluationError(Exception):
    """Base error exposed by the local evaluation harness."""


class ConfigurationError(EvaluationError):
    """Configuration is missing or unsafe."""


class DatasetError(EvaluationError):
    """The query manifest, labels, or prediction files are inconsistent."""


class GeminiError(EvaluationError):
    """Gemini could not produce a usable evaluation response."""


class ResponseValidationError(EvaluationError):
    """Gemini returned JSON that violates the local review contract."""
