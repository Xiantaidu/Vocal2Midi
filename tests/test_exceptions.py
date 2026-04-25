"""Unit tests for Vocal2Midi exception hierarchy."""

from application.exceptions import (
    Vocal2MidiError,
    ModelNotFoundError,
    ASRError,
    AlignmentError,
    ExportError,
    CancellationError,
)


class TestVocal2MidiError:
    """Tests for base exception and subclasses."""

    def test_base_error_message(self):
        """Base error should store message."""
        err = Vocal2MidiError("Something failed")
        assert str(err) == "Something failed"
        assert err.details == ""

    def test_base_error_with_details(self):
        """Base error should store details."""
        err = Vocal2MidiError("Something failed", details="Extra info")
        assert str(err) == "Something failed"
        assert err.details == "Extra info"

    def test_model_not_found_is_vocal2midi_error(self):
        """ModelNotFoundError should inherit from Vocal2MidiError."""
        err = ModelNotFoundError("Model not found")
        assert isinstance(err, Vocal2MidiError)
        assert isinstance(err, Exception)

    def test_model_not_found_with_details(self):
        """ModelNotFoundError should support details."""
        err = ModelNotFoundError("Model check failed", details="path/to/model not found")
        assert err.details == "path/to/model not found"

    def test_asr_error(self):
        """ASRError should be a Vocal2MidiError."""
        err = ASRError("ASR failed", details="Subprocess exit code 1")
        assert isinstance(err, Vocal2MidiError)
        assert str(err) == "ASR failed"

    def test_alignment_error(self):
        """AlignmentError should be a Vocal2MidiError."""
        err = AlignmentError("Alignment failed")
        assert isinstance(err, Vocal2MidiError)

    def test_export_error(self):
        """ExportError should be a Vocal2MidiError."""
        err = ExportError("Export failed")
        assert isinstance(err, Vocal2MidiError)

    def test_cancellation_error(self):
        """CancellationError should be a Vocal2MidiError."""
        err = CancellationError("User cancelled")
        assert isinstance(err, Vocal2MidiError)

    def test_all_exceptions_can_be_caught_by_base(self):
        """All custom exceptions should be catchable by Vocal2MidiError."""
        for exc_cls in [ModelNotFoundError, ASRError, AlignmentError, ExportError, CancellationError]:
            try:
                raise exc_cls("test")
            except Vocal2MidiError:
                pass  # expected
            except Exception:
                pytest.fail(f"{exc_cls.__name__} should be caught by Vocal2MidiError")