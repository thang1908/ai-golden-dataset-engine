class G5Error(RuntimeError):
    """Base error whose text is safe to show to a CLI user."""


class ConfigurationError(G5Error):
    pass


class ImageError(G5Error):
    pass


class AuthenticationError(G5Error):
    pass


class ApiError(G5Error):
    pass


class ResponseValidationError(G5Error):
    pass



