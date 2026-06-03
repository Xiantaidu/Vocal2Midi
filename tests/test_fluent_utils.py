from gui.fluent_utils import parse_quantization_mode


def test_parse_quantization_mode_accepts_sv_fitted_label():
    assert parse_quantization_mode("SV拟合量化（推荐）") == "bayes"


def test_parse_quantization_mode_keeps_bayes_label_compatible():
    assert parse_quantization_mode("Bayes量化（节拍感知 + 重复先验）") == "bayes"


def test_parse_quantization_mode_keeps_disabled_ui_label_stable():
    assert parse_quantization_mode("开发中") == "bayes"
