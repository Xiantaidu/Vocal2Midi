# 致谢

Vocal2Midi 项目建立在以下优秀的开源工作的基础上：

## 核心依赖

| 项目 | 说明 | 链接 |
|------|------|------|
| **GAME** | 歌声合成音符提取模型 | [GitHub](https://github.com/openvpi/GAME) |
| **HubertFA** | 基于 HuBERT 的音素级强制对齐工具 | [GitHub](https://github.com/Soulter/HubertFA) |
| **LyricFA** | 歌词强制对齐与音素匹配工具 | [GitHub](https://github.com/Anya1010/LyricFA) |
| **FunASR / Qwen3-ASR** | 端到端语音识别模型，提供高质量的多语言 ASR 能力 | [GitHub](https://github.com/modelscope/FunASR) |

## 其他依赖

| 项目 | 说明 |
|------|------|
| **RMVPE** | 歌声基频（F0）估计模型 |
| **qfluentwidgets** | 基于 PyQt5 的现代化 Fluent Design 组件库 |
| **PyTorch** | 深度学习框架 |
| **ONNX Runtime** | 跨平台高性能推理引擎 |
| **librosa** | 音频分析库 |
| **soundfile** | 音频文件读写库 |
| **pydantic** | 数据验证与模型定义 |
| **mido** | MIDI 文件读写库 |
| **tqdm** | 进度条显示 |

## 项目作者

- **Team OpenVPI** — 项目发起与主要维护

特别感谢所有为歌声合成与音乐信息检索领域做出贡献的研究者和开发者。

---

如需查看完整的依赖列表，请参阅 `requirements.txt` 和 `environment.yml`。