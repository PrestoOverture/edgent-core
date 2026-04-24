"""Core exception hierarchy."""


class CoreRuntimeError(Exception):
    """Base exception for all public runtime errors."""


class RouteError(CoreRuntimeError):
    """Raised when routing fails or produces an invalid decision."""


class ToolError(CoreRuntimeError):
    """Raised when tool lookup or tool execution fails."""


class BackendError(CoreRuntimeError):
    """Raised when a backend cannot satisfy a request."""


class ValidationError(CoreRuntimeError):
    """Raised when public input or output schema validation fails."""
