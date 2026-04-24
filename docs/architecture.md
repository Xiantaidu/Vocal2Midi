# Vocal2Midi 架构说明

本文档描述 Vocal2Midi 项目的分层职责、依赖方向与处理流程，用于降低模块耦合、提升代码可维护性。

---

## 1. 分层总览

项目采用以下分层结构：

```
gui/                    界面层（PyQt5 + qfluentwidgets）
application/            应用编排层（Use Case 入口，连接 GUI 与推理能力）
inference/              推理与算法实现层
  ├── pipeline/         流水线编排
  ├── API/              各模型 API 封装
  ├── io/               输入输出与格式导出
  ├── slicer/           音频切片
  └── quant/            音符量化
modules/、lib/          模型/算法基础模块与通用底层工具（规划中）
configs/                配置文件（规划中）
deployment/             导出/部署相关能力（规划中）
tools/                  离线工具脚本（规划中）
```

---

## 2. 依赖方向（必须遵守）

推荐依赖方向：

```
gui → application → inference → modules/lib
```

约束说明：

1. **GUI 不直接 import** `inference.auto_lyric_hybrid` 等重流程模块
2. GUI 通过 `application` 层调用稳定入口
3. `inference` 不依赖 `gui`
4. `tools/` 不作为运行期核心依赖
5. 第三方源码应存放于 `third_party/`，与业务代码隔离

---

## 3. 各层职责

### 3.1 界面层 `gui/`

| 文件 | 职责 |
|------|------|
| `gui/fluent_main.py` | 应用主窗口，初始化界面和导航 |
| `gui/fluent_worker.py` | 后台工作线程，执行推理任务 |
| `gui/fluent_views.py` | 主视图组件，含参数面板和日志输出 |
| `gui/fluent_utils.py` | UI 工具函数 |
| `gui/auto_lyric_view.py` | 自动歌词提取界面 |
| `gui/global_settings_view.py` | 全局设置界面（模型路径等） |

### 3.2 应用编排层 `application/`

| 文件 | 职责 |
|------|------|
| `application/pipeline.py` | `run_auto_lyric_job()` 函数，作为 GUI 调用推理的统一入口 |

**设计原则**：GUI 通过 `run_auto_lyric_job()` 间接调用推理管线，不直接依赖 `inference` 内部实现。便于未来替换管线实现而不影响界面代码。

### 3.3 推理层 `inference/`

#### 流水线编排

| 文件 | 职责 |
|------|------|
| `inference/pipeline/auto_lyric_hybrid.py` | 主混合管线：串联 ASR → LyricFA → HubertFA → GAME → RMVPE → 导出 |

#### API 模块

| 文件 | 职责 |
|------|------|
| `inference/API/asr_api.py` | Qwen3-ASR 语音识别，通过子进程隔离运行 |
| `inference/API/lfa_api.py` | LyricFA 歌词强制对齐与音素匹配 |
| `inference/API/hfa_api.py` | HubertFA ONNX 模型音素级强制对齐 |
| `inference/API/game_api.py` | GAME PyTorch 模型音符提取与歌词对齐 |
| `inference/API/rmvpe_api.py` | RMVPE 歌声音高估计 |
| `inference/API/slicer_api.py` | 音频切片策略（智能/启发式/网格搜索） |

#### 输入输出

| 文件 | 职责 |
|------|------|
| `inference/io/note_io.py` | `NoteInfo` dataclass 及 MIDI/TXT/CSV 格式导出 |

#### 音频切片

| 文件 | 职责 |
|------|------|
| `inference/slicer/slicer2.py` | 静音检测切片器 |

#### 音符量化

| 文件 | 职责 |
|------|------|
| `inference/quant/quantization.py` | 音符时序量化（简单/智能/动态规划模式） |

---

## 4. 主处理流程

```
用户音频
  │
  ▼
[1] 音频切片 (Slicer)
  │   - 基于静音检测将音频切分为片段
  │
  ▼
[2] ASR 语音识别 (Qwen3-ASR)
  │   - 对每个片段进行语音识别，生成文本与音素序列
  │
  ▼
[3] 歌词匹配 (LyricFA)  ← 可选
  │   - 将 ASR 音素序列与用户提供的参考歌词匹配
  │   - 修正 ASR 识别错误
  │
  ▼
[4] 生成 .lab 音素标签
  │
  ▼
[5] 强制对齐 (HubertFA)
  │   - 将音素对齐到音频帧级别
  │
  ▼
[6] 音符提取 (GAME)
  │   - 从对齐的音素序列中提取音符起止时间和音高
  │
  ▼
[7] 音高曲线提取 (RMVPE)  ← 可选
  │   - 提取帧级音高曲线用于 USTX 导出
  │
  ▼
[8] 音符对齐与量化
  │   - 将音符与歌词单元对齐
  │   - 量化音符时间到节拍网格
  │
  ▼
[9] 格式导出
      - .mid / .ustx / .txt / .csv / TextGrid
```

---

## 5. 数据模型

### NoteInfo

核心数据结构，定义在 `inference/io/note_io.py`：

```python
@dataclass
class NoteInfo:
    start_time: float   # 音符起始时间（秒）
    end_time: float     # 音符结束时间（秒）
    pitch: int          # MIDI 音高编号
    lyric: str          # 歌词文本
    phoneme: str        # 音素标注
```

---

## 6. 线程模型

```
GUI 主线程
  │
  ├── GlobalSettingsInterface    # 全局设置
  ├── AutoLyricInterface         # 自动歌词界面
  │     │
  │     ▼
  │   WorkerThread (QThread)     # 后台推理线程
  │     │
  │     ├── 实时日志通过信号输出 (StreamRedirector)
  │     ├── 支持 cancel_checker 安全中断
  │     └── 完成后发送 finished_signal
  │
  └── 跨界面信号连接（歌词匹配开关影响自动歌词视图）
```

---

## 7. 已落地的架构调整

### 7.1 新增 application 层

- `application/__init__.py`
- `application/pipeline.py`

其中 `run_auto_lyric_job(...)` 作为 GUI 调用主流程的统一入口，内部转发到 `inference.pipeline.auto_lyric_hybrid.auto_lyric_hybrid_pipeline(...)`。

### 7.2 GUI 调用迁移

- 文件：`gui/fluent_worker.py`
- 变更：从直接调用 `inference.auto_lyric_hybrid_pipeline` 改为调用 `application.run_auto_lyric_job`
- 结果：行为保持一致，但分层边界更清晰。

---

## 8. 后续演进建议

1. 将 `inference/pipeline/auto_lyric_hybrid.py` 按 stage 拆分：
   - ASR Stage
   - FA Stage
   - Pitch/Note Stage
   - Export Stage
2. 统一取消机制、日志、异常封装
3. 为各 stage 增加最小单元测试与 smoke test
4. 考虑引入配置对象（dataclass）封装 30+ 个 pipeline 参数
5. 创建 `third_party/` 目录存放第三方源码
6. 添加 CI/CD 流水线