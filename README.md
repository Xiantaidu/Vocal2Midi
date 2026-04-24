# Vocal2Midi

Vocal2Midi 是一个端到端的歌声推理工具包，将人声音频转换为带有歌词对齐的 MIDI 文件。  
项目基于 GAME 音符提取模型构建，将 ASR 语音识别、强制对齐和歌词匹配集成到一个完整的工作流中。

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [安装指南](#安装指南)
- [模型配置](#模型配置)
- [处理流程](#处理流程)
- [输出格式](#输出格式)
- [语言支持](#语言支持)
- [故障排除](#故障排除)
- [项目信息](#项目信息)

---

## 项目简介

Vocal2Midi 提供了一套完整的歌声转 MIDI 工具链：

| 组件 | 说明 |
|------|------|
| **主 GUI** | `app_fluent.py` — 基于 PyQt5 + qfluentwidgets 的桌面界面 |
| **主管线** | `inference/pipeline/auto_lyric_hybrid.py` — 混合推理管线 |
| **应用层** | `application/pipeline.py` — GUI 与推理模块之间的编排边界 |

项目结构遵循分层架构：`gui → application → inference → modules/lib`

---

## 核心特性

- 歌声到 MIDI 的完整提取（音符时序 + 音高）
- 基于 Qwen3-ASR 的自动歌词转录与对齐
- 可选的参考歌词修正（通过 LyricFA 匹配）
- 中文（`zh`）和日语（`ja`）流程支持
- 多种输出格式：`.mid`、`.ustx`、`.txt`、`.csv`、TextGrid
- RMVPE 音高提取，支持带音高曲线（pitd）的 USTX 导出
- 批量处理
- 可中断的后台任务，实时日志输出

---

## 架构概览

```
gui/                界面层（PyQt5 + qfluentwidgets）
application/        应用编排层（Use Case 入口）
inference/          推理与算法实现层
  ├── pipeline/     流水线编排
  ├── API/          各模型 API（ASR / HFA / LFA / GAME / RMVPE / Slicer）
  ├── io/           输入输出与格式导出
  ├── slicer/       音频切片
  └── quant/        音符量化
experiments/        模型存放目录
```

详细架构说明见 [docs/architecture.md](docs/architecture.md)。

---

## 快速开始

### Windows 启动器（推荐）

双击运行：

```bat
start.bat
```

该脚本会执行 `python app_fluent.py`，默认使用主 Fluent GUI 和混合管线。

### 手动启动

主 GUI：

```bash
python app_fluent.py
```

---

## 安装指南

### 方式一：脚本安装（Windows）

```bat
安装环境.bat
```

### 方式二：手动安装

使用 Conda 创建环境：

```bash
conda env create -f environment.yml
conda activate vocal2midi_torch
```

或使用 pip：

```bash
conda create -n vocal2midi python=3.12
conda activate vocal2midi
pip install -r requirements.txt
```

> PyTorch 建议手动安装，版本需 ≥ 2.0。详见 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)。

---

## 模型配置

运行前需要准备以下模型路径：

| 模型 | 说明 |
|------|------|
| GAME 模型目录 | 音符提取核心模型 |
| HubertFA 模型目录 | 音素强制对齐 |
| ASR 模型路径 | Qwen3-ASR 本地路径 |

建议目录结构：

```text
experiments/
  GAME-1.0-medium/
  1218_hfa_model_new_dict/
  Qwen3-ASR-1.7B/
  ...
```

在 GUI 全局设置中配置好对应路径后再运行长时间任务。

---

## 处理流程

主路径（`app_fluent.py` → `auto_lyric_hybrid.py`）的处理流程如下：

1. 加载音频并切分为片段
2. 运行 ASR（Qwen3-ASR）进行语音识别
3. （可选）将 ASR 音素序列与参考歌词匹配（LyricFA）
4. 生成 `.lab` 音素标签文件
5. 运行 HubertFA 强制对齐
6. 运行 GAME 模型提取音符与音高
7. （可选）运行 RMVPE 提取帧级音高曲线
8. 将音符与歌词单元对齐
9. 导出目标格式（`.mid`、`.ustx` 等）

---

## 输出格式

根据选项设置，可输出以下格式：

| 格式 | 说明 |
|------|------|
| `.mid` | 标准 MIDI 文件 |
| `.ustx` | OpenUtau 项目（含音高曲线） |
| `.txt` | 音符/歌词文本导出 |
| `.csv` | 表格式音符/歌词导出 |
| TextGrid | 调试或分析用途的标注文件 |

ASR 与匹配行为的日志也会同步生成，便于追踪回退/命中的片段。

---

## 语言支持

- 当前 GUI 支持中文（`zh`）和日语（`ja`）模式
- 日语对齐依赖日语 G2P/分词处理及 LyricFA 匹配行为
- 当 ASR 结果存在替换或插入标记时，提供参考歌词可显著提升稳定性

---

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| 启动失败 | 检查 Python 环境和依赖是否安装完整 |
| 模型加载失败 | 确认所有模型路径为有效目录/文件 |
| 日语功能异常 | 安装缺失的 G2P 依赖包 |
| CUDA/显存不足 | 在 GUI 设置中减小 batch size |
| FFmpeg 缺失 | 项目已内置 `_ffmpeg/bin/` 目录 |

---

## 项目信息

- **许可证**: [MIT License](LICENSE)
- **致谢**: 见 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)
- **架构说明**: 见 [docs/architecture.md](docs/architecture.md)