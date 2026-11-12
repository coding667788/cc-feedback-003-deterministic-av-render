"""Typed errors so callers can distinguish validation from I/O failures."""

from __future__ import annotations


class TimelineError(Exception):
    """Base class for all engine errors."""


class ProjectValidationError(TimelineError):
    """The project JSON is missing information or is self-contradictory."""


class SourceProbeError(TimelineError):
    """A source file cannot be probed or lacks required time information."""


class SourceDecodeError(TimelineError):
    """Decoding a source into the normalized work format failed."""


class RenderError(TimelineError):
    """A render segment could not be produced."""


class IntegrityError(TimelineError):
    """A cached/published artifact fails its content hash verification."""


class IncompleteJobError(TimelineError):
    """The job still has unfinished segments; publishing is refused."""
