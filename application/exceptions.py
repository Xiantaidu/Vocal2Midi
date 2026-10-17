class Vocal2MidiError(Exception):
    """Base exception for all Vocal2Midi-specific errors."""

    def __init__(self, message: str = "", *, details: str = ""):
        # Fold details into the message so user-facing tracebacks/dialogs carry them.
        super().__init__(f"{message}: {details}" if details else message)
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