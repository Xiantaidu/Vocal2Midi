import numpy as np
import pytest

from inference.API import asr_api


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _FakeProcess:
    def __init__(self):
        self.started = False
        self.terminated = False
        self.joined = False
        self.alive = False

    def terminate(self):
        self.terminated = True
        self.alive = False

    def start(self):
        self.started = True
        self.alive = True

    def join(self, timeout=None):
        self.joined = True

    def is_alive(self):
        return self.alive


class _FakeContext:
    def __init__(self, process, queues):
        self._process = process
        self._queues = list(queues)

    def Queue(self):
        return self._queues.pop(0)

    def Process(self, *args, **kwargs):
        return self._process


def test_subprocess_timeout_terminates_pool(monkeypatch, tmp_path):
    worker = _FakeProcess()
    task_queue = _FakeQueue()
    result_queue = _FakeQueue()
    ctx = _FakeContext(worker, [task_queue, result_queue])
    monkeypatch.setattr(asr_api.mp, "get_context", lambda method: ctx)
    wait_calls = iter([{"type": "ready"}])

    def fake_wait(*args, **kwargs):
        try:
            return next(wait_calls)
        except StopIteration:
            raise asr_api.mp.TimeoutError

    monkeypatch.setattr(asr_api, "_wait_for_worker_message", fake_wait)

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

    assert worker.started is True
    assert worker.terminated is True
    assert worker.joined is True
    assert task_queue.items[0]["type"] == "transcribe"


def test_clear_phoneme_cache_clears_shared_cache():
    asr_api._PHONEME_MODEL_CACHE["x"] = object()

    asr_api.clear_phoneme_model_cache()

    assert asr_api._PHONEME_MODEL_CACHE == {}


def test_load_qwen_model_uses_dml_runtime_cache(monkeypatch):
    calls = []

    class _DummyModel:
        device = "dml+cpu"

        def shutdown(self):
            calls.append("shutdown")

    dummy_model = _DummyModel()

    def fake_from_model_path(cls, model_path, device="dml", verbose=False):
        calls.append((model_path, device, verbose))
        return dummy_model

    monkeypatch.setattr(
        asr_api.Qwen3ASRDmlModel,
        "from_model_path",
        classmethod(fake_from_model_path),
    )

    first = asr_api.load_qwen_model("experiments/Qwen3-ASR-1.7B-dml", device="cpu", use_cache=True)
    second = asr_api.load_qwen_model("experiments/Qwen3-ASR-1.7B-dml", device="cpu", use_cache=True)

    assert first is dummy_model
    assert second is dummy_model
    assert calls == [("experiments/Qwen3-ASR-1.7B-dml", "cpu", False)]

    asr_api.clear_qwen_model_cache()

    assert calls[-1] == "shutdown"
