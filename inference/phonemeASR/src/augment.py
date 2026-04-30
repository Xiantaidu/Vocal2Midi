# File: src/augment.py
import numpy as np
import random
from audiomentations import (
    Compose,
    Gain,
    AddGaussianSNR,
    PolarityInversion,
    LowPassFilter,
    HighPassFilter,
)

# ApplyImpulseResponse 可能因缺少 RIR 数据集而不可用，做安全导入
try:
    from audiomentations import ApplyImpulseResponse
    _HAS_RIR = True
except ImportError:
    _HAS_RIR = False


class AudioAugmenter:
    def __init__(self, config: dict):
        self.config = config
        self.prob = config.get("waveform", {}).get("prob", 0.0)

        if self.prob > 0.0 and config.get("waveform", {}).get("enable", False):
            wf_config = config["waveform"]
            transforms = []

            # Gain 增强
            if "gain_db" in wf_config:
                transforms.append(
                    Gain(
                        min_gain_db=wf_config["gain_db"][0],
                        max_gain_db=wf_config["gain_db"][1],
                        p=0.5,
                    )
                )

            # 高斯噪声（使用 SNR 控制，从 config 读取范围）
            if "noise_snr_db" in wf_config:
                snr_range = wf_config["noise_snr_db"]
                transforms.append(
                    AddGaussianSNR(
                        min_snr_db=snr_range[0],
                        max_snr_db=snr_range[1],
                        p=0.5,
                    )
                )

            # 极性反转
            if wf_config.get("polarity_inversion_prob", 0.0) > 0.0:
                transforms.append(
                    PolarityInversion(p=wf_config["polarity_inversion_prob"])
                )

            # 均衡器模拟（LowPass + HighPass 组合）
            if wf_config.get("eq_prob", 0.0) > 0.0:
                eq_p = wf_config["eq_prob"]
                transforms.append(LowPassFilter(p=eq_p / 2))
                transforms.append(HighPassFilter(p=eq_p / 2))

            # RIR（房间冲击响应）
            rir_prob = wf_config.get("rir_prob", 0.0)
            if rir_prob > 0.0:
                rir_path = wf_config.get("rir_path", None)
                if _HAS_RIR and rir_path:
                    transforms.append(
                        ApplyImpulseResponse(
                            ir_path=rir_path,
                            p=rir_prob,
                        )
                    )
                else:
                    if not _HAS_RIR:
                        print(
                            "Warning: ApplyImpulseResponse not available in audiomentations. "
                            "RIR augmentation disabled."
                        )
                    if not rir_path:
                        print(
                            "Warning: rir_prob > 0 but no rir_path specified in config. "
                            "RIR augmentation disabled. "
                            "Set augment.waveform.rir_path to a directory of IR wav files."
                        )

            self.augmenter = Compose(transforms) if transforms else None
        else:
            self.augmenter = None

    def apply_waveform(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if self.augmenter and random.random() < self.prob:
            return self.augmenter(samples=wav, sample_rate=sr)
        return wav