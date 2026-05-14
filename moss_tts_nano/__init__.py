"""MOSS-TTS-Nano: realtime multilingual tiny TTS with internal phonemization."""

__all__ = ["__version__", "synthesize", "synthesize_stream"]

__version__ = "0.1.0"

from .synthesize import synthesize, synthesize_stream
