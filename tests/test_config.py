"""Unit tests for PipelineConfig dataclass."""

from pathlib import Path

import pytest

from application.config import (
    DEFAULT_SLICE_MAX_SEC,
    DEFAULT_SLICE_MIN_SEC,
    PipelineConfig,
    validate_slice_bounds,
)


class TestPipelineConfig:
    """Tests for PipelineConfig creation and validation."""

    @pytest.fixture
    def base_kwargs(self):
        """Minimum required kwargs for PipelineConfig."""
        return {
            "audio_path": "test.wav",
            "output_filename": "output",
            "output_dir": Path("/tmp/output"),
            "game_model_dir": "/models/game",
            "hfa_model_dir": "/models/hfa",
            "asr_model_path": "/models/asr",
            "device": "cpu",
            "language": "zh",
            "ts": [0.0, 0.1, 0.2],
        }

    def test_required_fields_only(self, base_kwargs):
        """Should create with only required fields, using defaults for the rest."""
        cfg = PipelineConfig(**base_kwargs)

        assert cfg.audio_path == "test.wav"
        assert cfg.language == "zh"
        assert cfg.tempo == 120.0  # default
        assert cfg.batch_size == 8  # default
        assert cfg.slice_min_sec == DEFAULT_SLICE_MIN_SEC
        assert cfg.slice_max_sec == DEFAULT_SLICE_MAX_SEC
        assert cfg.output_lyrics is True  # default
        assert cfg.debug_mode is False  # default
        assert cfg.cancel_checker is None  # default

    def test_default_output_formats(self, base_kwargs):
        """Should default to ['mid'] output format."""
        cfg = PipelineConfig(**base_kwargs)
        assert cfg.output_formats == ["mid"]

    def test_custom_output_formats(self, base_kwargs):
        """Should accept custom output formats."""
        cfg = PipelineConfig(**base_kwargs, output_formats=["mid", "ustx", "csv"])
        assert cfg.output_formats == ["mid", "ustx", "csv"]

    def test_to_kwargs_method(self, base_kwargs):
        """to_kwargs() should return a dict with all fields."""
        cfg = PipelineConfig(**base_kwargs)
        result = cfg.to_kwargs()

        assert isinstance(result, dict)
        assert result["audio_path"] == "test.wav"
        assert result["output_dir"] == Path("/tmp/output")
        assert result["language"] == "zh"
        assert result["tempo"] == 120.0
        assert result["slice_min_sec"] == DEFAULT_SLICE_MIN_SEC
        assert result["slice_max_sec"] == DEFAULT_SLICE_MAX_SEC

    def test_mutable_default_is_safe(self, base_kwargs):
        """Default output_formats should not be shared across instances."""
        cfg1 = PipelineConfig(**base_kwargs)
        cfg2 = PipelineConfig(**base_kwargs)

        cfg1.output_formats.append("ustx")
        assert cfg2.output_formats == ["mid"]

    def test_cancel_checker_is_callable(self, base_kwargs):
        """Cancel checker should accept a callable."""
        was_called = False

        def checker():
            nonlocal was_called
            was_called = True
            return False

        cfg = PipelineConfig(**base_kwargs, cancel_checker=checker)
        assert cfg.cancel_checker() is False
        assert was_called is True

    def test_rmvpe_model_path_default(self, base_kwargs):
        """rmvpe_model_path should default to empty string."""
        cfg = PipelineConfig(**base_kwargs)
        assert cfg.rmvpe_model_path == ""

    def test_to_kwargs_includes_rmvpe(self, base_kwargs):
        """to_kwargs() should include rmvpe_model_path."""
        cfg = PipelineConfig(**base_kwargs, rmvpe_model_path="/path/to/rmvpe.pt")
        result = cfg.to_kwargs()
        assert result["rmvpe_model_path"] == "/path/to/rmvpe.pt"

    def test_optional_fields_have_sensible_types(self, base_kwargs):
        """Optional fields should have expected types."""
        cfg = PipelineConfig(**base_kwargs)

        assert isinstance(cfg.lyric_output_mode, str)
        assert isinstance(cfg.tempo, float)
        assert isinstance(cfg.quantization_step, int)
        assert isinstance(cfg.batch_size, int)
        assert isinstance(cfg.seg_threshold, float)
        assert isinstance(cfg.rmvpe_model_path, str)

    def test_validate_slice_bounds_accepts_zero_min(self):
        """Zero minimum slice duration should be allowed when max is positive."""
        validate_slice_bounds(0.0, 22.0)

    def test_validate_slice_bounds_rejects_zero_max(self):
        """Zero maximum slice duration should be rejected."""
        with pytest.raises(ValueError, match="slice_max_sec"):
            validate_slice_bounds(0.0, 0.0)

    def test_validate_slice_bounds_rejects_min_greater_than_max(self):
        """Minimum slice duration cannot exceed maximum."""
        with pytest.raises(ValueError, match="slice_min_sec"):
            validate_slice_bounds(24.0, 12.0)
