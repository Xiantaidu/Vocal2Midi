"""Custom exception hierarchy for Vocal2Midi.

Provides structured error types that the GUI and application layers
can catch and handle consistently.
"""


class Vocal2MidiError(Exception):
    """Base exception for all Vocal2Midi-specific errors."""

    def __init__(self, message: str = "", *, details: str = ""):
        super().__init__(message)
        self.details = details


class ModelNotFoundError(Vocal2MidiError):
    """Raised when a required model path does not exist or is invalid."""


class ASRError(Vocal2MidiError):
    """Raised when the ASR subprocess fails or returns invalid output."""


class AlignmentError(Vocal2MidiError):
    """Raised when forced alignment fails."""


class ExportError(Vocal2MidiError):
    """Raised when format export fails."""


class CancellationError(Vocal2MidiError):
    """Raised when the user cancels a running pipeline."""