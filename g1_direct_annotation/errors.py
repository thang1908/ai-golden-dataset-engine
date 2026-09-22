class G1Error(RuntimeError):
    """Base error whose text is safe to show to a CLI user."""


class ConfigurationError(G1Error):
    pass


class ImageError(G1Error):
    pass


class AuthenticationError(G1Error):
    pass


class ApiError(G1Error):
    pass


class ResponseValidationError(G1Error):
    pass
