class G4Error(RuntimeError):
    """Base error whose text is safe to show to a CLI user."""


class ConfigurationError(G4Error):
    pass


class ImageError(G4Error):
    pass


class AuthenticationError(G4Error):
    pass


class ApiError(G4Error):
    pass


class ResponseValidationError(G4Error):
    pass



