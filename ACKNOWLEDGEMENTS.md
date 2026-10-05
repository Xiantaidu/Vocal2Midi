# Acknowledgements

Vocal2Midi builds on several open-source projects for transcription, alignment, pitch extraction, runtime support, and UI tooling.

## Repository license

The overall Vocal2Midi repository is distributed under the **Apache License 2.0**.

Third-party components, vendored code, bundled data, dictionaries, and other embedded materials may retain their own original licenses, copyright notices, and attribution requirements. Where such notices are present, they remain applicable to the corresponding materials.

## Core upstream projects

| Project | Role in Vocal2Midi | Upstream repository |
| --- | --- | --- |
| GAME | note and pitch extraction | https://github.com/openvpi/GAME |
| TIFA | default forced alignment engine for Chinese, Japanese, English, and Cantonese | https://github.com/openvpi/TIFA |
| RomajiASR | Japanese mora / romaji singing ASR used by the Japanese lyric path | https://github.com/Xiantaidu/RomajiASR |
| PinyinASR | lightweight Chinese pinyin singing ASR used by the Chinese lyric path | https://github.com/Xiantaidu/PinyinASR |
| kashi-g2p | Japanese G2P runtime and lattice beam search for forced alignment | https://github.com/Xiantaidu/kashi-g2p |
| HubertFA | phoneme-level forced alignment | https://github.com/wolfgitpr/HubertFA |
| LyricFA | lyric matching and G2P-based lyric alignment helpers | https://github.com/wolfgitpr/LyricFA |
| FunASR | broader ASR foundation referenced by the Qwen3-ASR integration path | https://github.com/modelscope/FunASR |
| llama.cpp | Platform-aware decoder runtime used by the Qwen3-ASR path | https://github.com/ggml-org/llama.cpp |
| ONNX Runtime | DirectML and CPU inference execution | https://github.com/microsoft/onnxruntime |
| PyQt-Fluent-Widgets | Fluent-style desktop UI components | https://github.com/zhiyiYo/PyQt-Fluent-Widgets |

## Vendored components in this repository

The repository currently includes local copies or adapted subsets of some upstream projects:

- `inference/TiFA/`
- `inference/romaji_asr/`
- `inference/pinyin_asr/`
- `inference/kashi_g2p_ja/`
- `inference/HubertFA/`
- `inference/LyricFA/`
- `inference/qwen3asr_dml/gguf/`

These copies may contain project-specific edits for integration, runtime changes, or interface compatibility.

## Additional libraries

Vocal2Midi also relies on widely used libraries from the Python audio and scientific stack, including `librosa`, `numpy`, `scipy`, `soundfile`, `textgrid`, `PyYAML`, and related tooling listed in `requirements.txt`.

## Thanks

Thanks to the maintainers and contributors of the upstream projects above, and to the singing synthesis / music information retrieval community whose open tooling makes this kind of integration work possible.
