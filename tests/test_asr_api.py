from types import SimpleNamespace

import numpy as np
import pytest

from inference.API import asr_api


class _NeverReadyResult:
    def ready(self):
        return False


class _FakePool:
    def __init__(self):
        self.terminated = False
        self.joined = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def apply_async(self, *args, **kwargs):
        return _NeverReadyResult()

    def terminate(self):
        self.terminated = True

    def join(self):
        self.joined = True


def test_subprocess_timeout_terminates_pool(monkeypatch, tmp_path):
    pool = _FakePool()
    ctx = SimpleNamespace(Pool=lambda *args, **kwargs: pool)
    monkeypatch.setattr(asr_api.mp, "get_context", lambda method: ctx)

    chunks = [{"waveform": np.zeros(16, dtype=np.float32)}]

    with pytest.raises(TimeoutError, match="timed out"):
        asr_api.batch_transcribe_asr(
            chunks,
            sr=16000,
            asr_model=None,
            temp_dir_path=tmp_path,
            asr_batch_size=1,
            language="zh",
            asr_model_path="asr",
            device="cpu",
            force_subprocess=True,
            asr_timeout_sec=0.0,
        )

    assert pool.terminated is True
    assert pool.joined is True


def test_clear_phoneme_cache_releases_cuda_cache(monkeypatch):
    asr_api._PHONEME_MODEL_CACHE["x"] = object()
    empty_cache = []
    monkeypatch.setattr(asr_api.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(asr_api.torch.cuda, "empty_cache", lambda: empty_cache.append(True))

    asr_api.clear_phoneme_model_cache()

    assert asr_api._PHONEME_MODEL_CACHE == {}
    assert empty_cache == [True]
