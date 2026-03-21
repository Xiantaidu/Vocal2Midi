# Vocal2Midi

**Vocal2Midi** is a streamlined, all-in-one inference tool designed to automatically convert raw singing voice audio into precise MIDI files with perfectly aligned lyrics. It builds upon the Generative Adaptive MIDI Extractor (GAME) and deeply integrates state-of-the-art Automatic Speech Recognition (ASR) and Acoustic Forced Alignment technologies.

This tool is specifically tailored for virtual singer creators, music producers, and developers. It can handle raw vocals (including those with some background noise) and accurately extract note boundaries, pitches, and lyrics, providing an extremely accessible WebUI for batch processing.

---

## Key Features

- **Automated Vocal to MIDI Transcription**
  Powered by an advanced D3PM architecture (derived from the GAME model), it extracts note boundaries and pitches from audio with high precision and exports them as standard MIDI files.
- **Auto Lyric Alignment**
  Deeply integrated with `FunASR` and `HubertFA`. It not only extracts musical notes but also automatically transcribes lyrics and aligns them to each specific note. It also supports custom reference lyrics for intelligent error correction.
- **Lightweight ONNX Engine**
  Features a pure ONNX inference pipeline. There is no need to install massive PyTorch environments. It supports DirectML (`dml`) on Windows for efficient GPU hardware acceleration, allowing fast inference even on low-VRAM machines.
- **User-Friendly Web Interface**
  Includes a Gradio-based graphical interface. Users can visually adjust parameters, batch process audio files, and generate outputs seamlessly by simply running the startup script.
- **Dedicated Inference Repository**
  Stripped of all complex training and preprocessing code from the original deep learning models, this repository is highly streamlined and ready to use out of the box.

---

## Installation

This project primarily supports Windows. You can set up the environment using the provided batch script or manually via Conda/UV.

### Method 1: Quick Installation Script (Windows)
1. Ensure you have [Python 3.10+](https://www.python.org/downloads/) or a Conda distribution installed on your system.
2. Double-click the `安装环境.bat` (Install Environment) script in the project directory.
3. The script will automatically create a virtual environment and install all necessary dependencies.

### Method 2: Manual Installation
1. Create and activate a Python virtual environment:
   ```bash
   conda create -n vocal2midi python=3.12
   conda activate vocal2midi
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements_onnx.txt
   ```
   *(Note: If you intend to use the PyTorch inference engine, please install `requirements.txt` instead and follow the official PyTorch guide to install the version matching your CUDA toolkit.)*

---

## Usage

### Launch the Web Interface
Double-click `start.bat` in the project directory, or run the following command in your terminal:
```bash
python app.py
```
Once the application starts, it will automatically open the Web UI in your default browser (Default address: `http://127.0.0.1:7860`).

### Core Functions Overview:
1. **Extract Audio (Raw MIDI Extraction)**
   - The fundamental vocal-to-MIDI feature. Upload your audio files to generate MIDI files. It also supports exporting pitch information in `.txt` and `.csv` formats.
2. **Auto Lyric (Transcription & Alignment)**
   - The highlight feature. Upload an audio file, and the system will automatically perform ASR -> Forced Phoneme Alignment -> GAME Pitch Extraction -> MIDI Generation with injected lyrics.
   - It is highly recommended to use this feature with the `HubertFA` ONNX model.
3. **Align Datasets**
   - Designed for dataset creators. Supports batch processing of DiffSinger format `transcriptions.csv` files, extracting note information and injecting it directly back into the CSV labels.

### Command-Line Interface (CLI)
You can also run the Auto Lyric pipeline directly from the command line without opening the WebUI, which is ideal for batch scripting and automation:
```bash
python inference/auto_lyric.py --help
```
Example usage:
```bash
python inference/auto_lyric.py input.wav -gm experiments/GAME-1.0.3-medium-onnx -hm experiments/1218_hfa_model_new_dict/model.onnx -o ./output/ -f mid,txt
```

---

## Recommended Model Paths

For ease of management, we recommend placing your downloaded models in the `experiments` folder:
- **GAME ONNX Model**: `experiments/GAME-1.0.3-medium-onnx/`
- **HubertFA ONNX Model**: `experiments/1218_hfa_model_new_dict/model.onnx`
- **FunASR Model**: `experiments/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/`

You can specify these paths in the "Model Path" fields within the WebUI. The application is configured to load them from these default locations for a complete offline experience.

---

## Acknowledgements

The core capabilities of this project rely on several outstanding open-source projects. We express our sincere gratitude to their authors and contributors:

- [**GAME**](https://github.com/openvpi/GAME) - Provided the core generative adaptive MIDI extraction model and inference pipeline.
- [**HubertFA**](https://github.com/Soulter/HubertFA) - Provided robust acoustic forced alignment technology. This project deeply integrates its inference scripts.
- [**LyricFA**](https://github.com/Anya1010/LyricFA) - Utilized for Grapheme-to-Phoneme (G2P) conversion and intelligent ASR lyric matching and correction.
- [**FunASR**](https://github.com/modelscope/FunASR) - Provided accurate Chinese speech recognition capabilities.

For detailed open-source licenses and attribution, please refer to the [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) file in this repository.

---

## License

This project inherits and follows the [MIT License](LICENSE).

> **Disclaimer**: 
> Any organization or individual is prohibited from using the functionalities provided in this repository to generate, convert, or publish someone else's singing or speech without their explicit consent. This includes, but is not limited to, government leaders, political figures, and public figures. By using this software, users agree to assume full responsibility for any copyright disputes or legal liabilities that may arise from their usage.
