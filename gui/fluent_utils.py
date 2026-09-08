def parse_quantization(quantize_option: str) -> int:
    if "1/4 音符" in quantize_option:
        return 480
    elif "1/8 音符" in quantize_option:
        return 240
    elif "1/16 音符" in quantize_option:
        return 120
    elif "1/32 音符" in quantize_option:
        return 60
    elif "1/64 音符" in quantize_option:
        return 30
    return 0


def parse_quantization_mode(mode_option: str) -> str:
    mode_option = (mode_option or "").strip()
    # Exact-label mapping first; the legacy "开发中" placeholder maps to bayes
    # for settings compatibility. Fallback substring matches cover variants,
    # with "dp" matched exactly (case-insensitive) so "SV-style" strings can
    # never shadow other labels.
    exact = {
        "节奏修复": "repair",
        "贝叶斯": "bayes",
        "DP": "dp",
        "简单": "simple",
        "智能": "smart",
        "开发中": "bayes",
    }
    if mode_option in exact:
        return exact[mode_option]
    lowered = mode_option.lower()
    if "修复" in mode_option or "repair" in lowered:
        return "repair"
    if "贝叶斯" in mode_option or "bayes" in lowered or "拟合量化" in mode_option:
        return "bayes"
    if "动态规划" in mode_option or lowered == "dp":
        return "dp"
    if "智能" in mode_option:
        return "smart"
    return "simple"


def t0_nstep_to_ts(t0: float, nsteps: int) -> list:
    if nsteps <= 0:
        return [t0]
    step = (1 - t0) / nsteps
    return [t0 + i * step for i in range(nsteps)]
