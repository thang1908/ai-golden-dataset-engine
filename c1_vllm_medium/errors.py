"""C1 error types safe to show to command-line users."""


class C1Error(RuntimeError):
    """Base error. Messages must never include credentials or provider response bodies."""


class ConfigurationError(C1Error):
    """Missing or malformed local configuration."""


class ImageError(C1Error):
    """The input image cannot be decoded or safely compressed."""


class AuthenticationError(C1Error):
    """OAuth token could not be retrieved or refreshed."""


class ApiError(C1Error):
    """The remote Chat Completions request did not succeed."""


class ResponseValidationError(C1Error):
    """The model response violates the 21-attribute contract."""
