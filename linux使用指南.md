# Vocal2Midi Linux 使用指南

## 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- GCC / Clang + CMake（编译 llama.cpp 用）
- Qt5 系统库（PyQt5 需要）

**Arch Linux：**

```bash
sudo pacman -S qt5-base qt5-tools
```

**Ubuntu / Debian：**

```bash
sudo apt install qt5-qmake qtbase5-dev
```

---

## 1. 安装依赖

```bash
# 创建虚拟环境并安装核心依赖
uv sync --no-dev

# 如果还需运行测试:
uv sync --group dev
```

> 也可以直接用 pip：`pip install -r requirements.txt`

---

## 2. 准备模型

Vocal2Midi 需要 5 个模型，共约 3.5 GB。提供两种方式：

### 方式 A：自动下载（推荐）

```bash
bash download_models.sh
```

脚本会自动下载全部模型到 `experiments/` 目录并编译 llama.cpp。

### 方式 B：复用 OpenUTAU 已有模型 + 补下载

> 如果你安装了 OpenUTAU，`game` 和 `rmvpe` 模型可以直接复用。

```bash
bash download_models.sh --from-openutau
```

脚本会自动检测 `~/.local/share/OpenUtau/Dependencies/` 下的已有模型，只下载缺失的部分。

### 方式 C：手动链接 + 下载

```bash
mkdir -p experiments

# 链接 OpenUTAU 已有模型（可选）
ln -sfn ~/.local/share/OpenUtau/Dependencies/game experiments/GAME-1.0.3-medium-onnx
mkdir -p experiments/RMVPE
ln -sfn ~/.local/share/OpenUtau/Dependencies/rmvpe/rmvpe.onnx experiments/RMVPE/rmvpe.onnx

# 下载其余模型
bash download_models.sh
```

### 模型清单

下载完成后，`experiments/` 目录结构如下：

```
experiments/
├── GAME-1.0.3-medium-onnx/     # 音高/音符提取   ~172 MB
├── 1218_hfa_model_new_dict/    # 强制对齐        ~245 MB
├── Qwen3-ASR-1.7B-dml/         # 语音识别(int4)  ~2.7 GB
├── RMVPE/
│   └── rmvpe.onnx               # 音高曲线        ~362 MB
└── romajiASR/                   # 日语识别        ~168 MB
```

---

## 3. 启动 GUI

```bash
uv run vocal2midi
```

**首次使用：** 进入 **设置 → 模型配置**，确认 5 个模型的路径指向正确。

GUI 主界面操作：
1. 点击 **选择音频文件**（支持 `.wav` / `.mp3` / `.m4a`）
2. 选择 **输出格式**（MIDI / USTX / 文本 / CSV 等）
3. 选择 **语言**（中文 / 日语）
4. 点击 **开始转换**

---

## 4. CLI 批量处理

```bash
# 批量切片 + ASR 识别
uv run slice-asr ~/输入文件夹 ~/输出文件夹 \
  --asr-model experiments/Qwen3-ASR-1.7B-dml \
  --language zh \
  --device cpu

# 日语 + 整段识别（不切片）
uv run slice-asr ~/输入文件夹 ~/输出文件夹 \
  --asr-model experiments/Qwen3-ASR-1.7B-dml \
  --language ja \
  --device cpu \
  --no-slice

# 智能切片（使用 RMVPE 辅助）
uv run slice-asr ~/输入文件夹 ~/输出文件夹 \
  --asr-model experiments/Qwen3-ASR-1.7B-dml \
  --language zh \
  --device cpu \
  --slicing-method smart \
  --keep-model
```

### CLI 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `input_dir` | (必填) | 输入音频文件夹 |
| `output_dir` | (必填) | 输出文件夹 |
| `--asr-model` | (必填) | Qwen3-ASR 模型路径 |
| `--device` | `dml` | 运行设备：`cpu` / `dml`（Linux 上无 GPU 加速时用 cpu） |
| `--language` | `zh` | 语言：`zh`（中文）/ `ja`（日语） |
| `--slicing-method` | `default` | 切片方式：`default` / `smart` / `heuristic` / `grid` |
| `--no-slice` | — | 整段识别，不切片 |
| `--asr-batch-size` | `4` | ASR 批处理大小 |
| `--file-batch-size` | `1` | 每批处理文件数 |
| `--keep-model` | — | 保持模型常驻（批量处理加速） |
| `--save-json` | — | 输出 JSON 时间戳 |
| `--no-recursive` | — | 不递归扫描子目录 |
| `--no-skip-existing` | — | 强制重新处理已有文件 |

---

## 5. 运行测试

```bash
uv run pytest tests/
```

---

## 注意事项

- **DirectML 仅在 Windows 上可用**，Linux 上使用 `--device cpu` 运行 ONNX 和 llama.cpp
- **pyopenjtalk**（日语处理）在 Linux 上可能需要从源码编译，`uv sync` 会自动处理
- **llama.cpp** 编译需要 `cmake` 和 `gcc`/`clang`，脚本会自动编译
- **模型路径** 可在 GUI 设置中随时修改，不一定要放在 `experiments/`
- 所有模型均来自上游项目，请遵守各自的许可协议
