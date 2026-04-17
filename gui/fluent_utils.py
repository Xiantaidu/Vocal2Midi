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
    if "智能" in (mode_option or ""):
        return "smart"
    return "simple"


def t0_nstep_to_ts(t0: float, nsteps: int) -> list:
    step = (1 - t0) / nsteps
    return [t0 + i * step for i in range(nsteps)]
