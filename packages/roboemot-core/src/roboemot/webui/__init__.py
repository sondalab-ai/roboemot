"""Local web UI for enrollment and registry management (SPEC 6, 11A, 12, 12A).

Robot-agnostic: it talks to an :class:`EnrollmentPort` (the app's enrollment service) and a
diagnostics registry. A robot app points ``custom_app_url`` at its port (SPEC 4A).
"""

from .api import DebugReport, EnrollmentPort
from .server import Server, make_server

__all__ = ["DebugReport", "EnrollmentPort", "Server", "make_server"]
