"""The GUI must not report success when every queued song failed."""
from types import SimpleNamespace

from gui.fluent_worker import classify_batch_result


def _item(name, failed_stage=None, error=None):
    return SimpleNamespace(name=name, failed_stage=failed_stage, error=error)


def test_success_when_every_item_finished():
    items = [_item("a.wav"), _item("b.wav")]
    outcome, total, failed, first_error = classify_batch_result(items)
    assert outcome == "success"
    assert total == 2
    assert failed == []
    assert first_error is None


def test_all_failed_is_not_reported_as_success():
    items = [
        _item("a.wav", "slicing", "boom a"),
        _item("b.wav", "slicing", "boom b"),
    ]
    outcome, total, failed, first_error = classify_batch_result(items)
    assert outcome == "all_failed"
    assert total == 2
    assert len(failed) == 2
    assert first_error == "boom a"


def test_partial_failure_reports_counts():
    items = [
        _item("a.wav"),
        _item("b.wav", "alignment", "boom"),
        _item("c.wav"),
    ]
    outcome, total, failed, first_error = classify_batch_result(items)
    assert outcome == "partial"
    assert total == 3
    assert len(failed) == 1
    assert failed[0].name == "b.wav"


def test_empty_queue_is_success_not_all_failed():
    # An empty list must not be classified as "all failed" (0 of 0).
    outcome, total, failed, _ = classify_batch_result([])
    assert outcome == "success"
    assert total == 0
    assert failed == []


def test_none_items_is_tolerated():
    outcome, total, failed, _ = classify_batch_result(None)
    assert outcome == "success"
    assert total == 0
