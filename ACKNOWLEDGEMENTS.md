# Acknowledgements

This project, **Vocal2Midi**, builds upon and integrates code from several excellent open-source projects. We express our deep gratitude to their authors and contributors.

## Upstream Repositories

### 1. [GAME (Generative Adaptive MIDI Extractor)](https://github.com/openvpi/GAME)
Vocal2Midi is heavily based on GAME for its core generative boundary extraction and pitch estimation capabilities. 
- **License**: MIT License
- **Modifications**: We utilized its ONNX inference pipeline and extended it with automatic lyric alignment and full GUI support.

### 2. [HubertFA](https://github.com/Soulter/HubertFA)
Used for robust acoustic forced alignment in the automatic lyric transcription pipeline.
- **Modifications**: We extracted the essential ONNX inference and decoding scripts to `third_party/inference_vendor/HubertFA/` to keep third-party code isolated from first-party runtime modules.

### 3. [LyricFA](https://github.com/Anya1010/LyricFA)
Used for G2P (Grapheme-to-Phoneme) conversion and intelligent lyric-to-ASR matching.
- **Modifications**: Extracted the core alignment logic and dictionaries to `third_party/inference_vendor/LyricFA/`.

### 4. [FunASR](https://github.com/modelscope/FunASR)
Historically evaluated in earlier experimentation branches for ASR exploration.
It is not part of the current primary runtime pipeline.
- **License**: MIT License

---

*All extracted third-party code remains under the copyright of their respective owners and is subject to their original open-source licenses.*
