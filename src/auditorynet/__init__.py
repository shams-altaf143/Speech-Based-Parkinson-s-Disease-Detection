"""AuditoryNet: biologically motivated CNN for speech-based Parkinson's detection."""

__version__ = "1.2.0"
__all__ = ["Config", "AuditoryNet"]


def __getattr__(name):  # lazy: importing the package must not require torch
    if name == "Config":
        from .config import Config
        return Config
    if name == "AuditoryNet":
        from .model import AuditoryNet
        return AuditoryNet
    raise AttributeError(name)
