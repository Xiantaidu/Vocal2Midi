from inference.t2s_utils import traditional_to_simplified


def test_traditional_to_simplified_converts_traditional_characters():
    text = "冷風偏偏吹雪，你凍嗎？這顆卑微的心它不再跳，前面硬石塊隨着雪落下來。"
    expected = "冷风偏偏吹雪，你冻吗？这颗卑微的心它不再跳，前面硬石块随着雪落下来。"
    assert traditional_to_simplified(text) == expected


def test_traditional_to_simplified_preserves_non_chinese_and_already_simplified():
    text = "你好 Hello 123! 简单测试。"
    assert traditional_to_simplified(text) == text


def test_traditional_to_simplified_empty_or_none():
    assert traditional_to_simplified("") == ""
    assert traditional_to_simplified(None) == ""
