"""Deterministic multi-timebase A/V timeline rendering engine.

The engine is organized in strict layers:

``rational``     exact rational time arithmetic (no float authority)
``model``        JSON project schema and in-memory objects
``errors``       typed validation/render errors
``ffmpeg_tools`` version capture, probing, subprocess wrappers
``source``       source decode into normalized work format, content cache
``timeline``     exact frame/sample planning, source time mapping, blending
``segment``      dependency graph, bounded parallel scheduler, segment cache
``render``       job orchestration, atomic publish/resume
``validator``    independent decode-based verification

All authoritative timeline arithmetic uses :class:`fractions.Fraction` or
integer tick math derived from it.  Floating point only appears inside
vectorized numpy sample math whose error budget is declared in the docs.
"""

__all__ = [
    "rational",
    "model",
    "errors",
    "ffmpeg_tools",
    "source",
    "timeline",
    "segment",
    "render",
    "validator",
]

__version__ = "1.0.0"
