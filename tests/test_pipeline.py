"""Unit tests for application/pipeline.py with PipelineConfig."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from application.config import PipelineConfig
from application.pipeline import run_auto_lyric_job, _validate_model_paths
from application.exceptions import (
    ModelNotFoundError,
    CancellationError,
    Vocal2MidiError,
)


class TestValidateModelPaths:
    """Tests for _validate_model_paths function."""

    @pytest.fixture
    def base_cfg(self):
        """PipelineConfig with non-existent paths."""
        return PipelineConfig(
            audio_path="test.wav",
            output_filename="output",
            output_dir=Path("/tmp"),
            game_model_dir="/does/not/exist/game",
            hfa_model_dir="/does/not/exist/hfa",
            asr_model_path="/does/not/exist/asr",
            device="cpu",
            language="zh",
            ts=torch.tensor([0.0]),
        )

    def test_missing_all_paths_raises(self, base_cfg):
        """Should raise ModelNotFoundError when all paths are missing."""
        with pytest.raises(ModelNotFoundError) as exc_info:
            _validate_model_paths(base_cfg)
        assert "GAME" in exc_info.value.details
        assert "HubertFA" in exc_info.value.details
        assert "ASR" in exc_info.value.details

    def test_missing_partial_paths_raises(self, base_cfg, tmp_path):
        """Should raise when some paths are missing."""
        existing = str(tmp_path)
        base_cfg.game_model_dir = existing  # valid
        # hfa_model_dir and asr_model_path still invalid

        with pytest.raises(ModelNotFoundError) as exc_info:
            _validate_model_paths(base_cfg)
        assert "GAME" not in exc_info.value.details  # valid path not in errors
        assert "HubertFA" in exc_info.value.details
        assert "ASR" in exc_info.value.details

    def test_all_valid_paths_passes(self, base_cfg, tmp_path):
        """Should not raise when all paths exist."""
        existing = str(tmp_path)
        base_cfg.game_model_dir = existing
        base_cfg.hfa_model_dir = existing
        base_cfg.asr_model_path = existing

        # Should not raise
        _validate_model_paths(base_cfg)

    def test_empty_string_path_raises(self, base_cfg):
        """Empty string paths should be treated as invalid."""
        base_cfg.game_model_dir = ""
        base_cfg.hfa_model_dir = "/exists"
        base_cfg.asr_model_path = "/exists"

        with pytest.raises(ModelNotFoundError):
            _validate_model_paths(base_cfg)


class TestRunAutoLyricJob:
    """Tests for run_auto_lyric_job."""

    @pytest.fixture
    def valid_cfg(self, tmp_path):
        """PipelineConfig with valid paths."""
        return PipelineConfig(
            audio_path="test.wav",
            output_filename="output",
            output_dir=tmp_path,
            game_model_dir=str(tmp_path),
            hfa_model_dir=str(tmp_path),
            asr_model_path=str(tmp_path),
            device="cpu",
            language="zh",
            ts=torch.tensor([0.0]),
        )

    @patch("application.pipeline.auto_lyric_hybrid_pipeline")
    def test_successful_execution(self, mock_pipeline, valid_cfg):
        """Should call the pipeline with correct arguments."""
        run_auto_lyric_job(valid_cfg)

        mock_pipeline.assert_called_once()
        call_args = mock_pipeline.call_args.kwargs
        assert call_args["audio_path"] == "test.wav"
        assert call_args["language"] == "zh"
        assert call_args["device"] == "cpu"

    @patch("application.pipeline.auto_lyric_hybrid_pipeline")
    def test_cancellation_before_start(self, mock_pipeline, valid_cfg):
        """Should raise CancellationError if cancel_checker returns True."""
        valid_cfg.cancel_checker = lambda: True

        with pytest.raises(CancellationError) as exc_info:
            run_auto_lyric_job(valid_cfg)

        assert "cancelled before starting" in str(exc_info.value).lower()
        mock_pipeline.assert_not_called()

    @patch("application.pipeline.auto_lyric_hybrid_pipeline")
    def test_cancel_checker_not_called_when_none(self, mock_pipeline, valid_cfg):
        """Should not raise when cancel_checker is None."""
        valid_cfg.cancel_checker = None

        run_auto_lyric_job(valid_cfg)
        mock_pipeline.assert_called_once()

    @patch("application.pipeline.auto_lyric_hybrid_pipeline")
    def test_pipeline_interrupted_error_converted(self, mock_pipeline, valid_cfg):
        """InterruptedError should be converted to CancellationError."""
        mock_pipeline.side_effect = InterruptedError("interrupted")

        with pytest.raises(CancellationError):
            run_auto_lyric_job(valid_cfg)

    @patch("application.pipeline.auto_lyric_hybrid_pipeline")
    def test_generic_exception_converted(self, mock_pipeline, valid_cfg):
        """Generic exceptions should be wrapped in Vocal2MidiError."""
        mock_pipeline.side_effect = RuntimeError("Something went wrong")

        with pytest.raises(Vocal2MidiError) as exc_info:
            run_auto_lyric_job(valid_cfg)

        assert "Pipeline execution failed" in str(exc_info.value)
        assert "Something went wrong" in exc_info.value.details