class G2Error(RuntimeError):
    """Base error whose text is safe to show to a CLI user."""


class ConfigurationError(G2Error):
    pass


class ImageError(G2Error):
    pass


class AuthenticationError(G2Error):
    pass


class ApiError(G2Error):
    pass


class ResponseValidationError(G2Error):
    pass

