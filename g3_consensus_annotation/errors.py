class G3Error(RuntimeError):
    """Base error whose text is safe to show to a CLI user."""


class ConfigurationError(G3Error):
    pass


class ImageError(G3Error):
    pass


class AuthenticationError(G3Error):
    pass


class ApiError(G3Error):
    pass


class ResponseValidationError(G3Error):
    pass


