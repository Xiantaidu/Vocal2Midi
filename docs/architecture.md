# Vocal2Midi 架构说明（阶段 A）

本文档描述当前仓库的分层职责与依赖方向约束，用于降低模块耦合、提升后续重构可维护性。

## 1. 分层总览

当前建议采用以下分层：

- `gui/`：界面层（PyQt5 + qfluentwidgets）
- `application/`：应用编排层（Use Case 入口，连接 GUI 与推理能力）
- `inference/`：推理与算法实现层（ASR/HFA/GAME/RMVPE/USTX 等）
- `third_party/`：第三方源码镜像（如 HubertFA/LyricFA），与业务代码隔离
- `modules/`、`lib/`：模型/算法基础模块与通用底层工具
- `configs/`：配置文件
- `deployment/`：导出/部署相关能力
- `tools/`：离线工具脚本

## 2. 依赖方向（必须遵守）

推荐依赖方向：

`gui -> application -> inference -> modules/lib`

约束说明：

1. GUI 不直接 import `inference.auto_lyric_hybrid` 等重流程模块。
2. GUI 通过 `application` 层调用稳定入口。
3. `inference` 不依赖 `gui`。
4. `tools/` 不作为运行期核心依赖。
5. 第三方源码仅允许存放在 `third_party/`，不再放入 `inference/vendor/`。

## 3. 当前已落地的边界调整

### 3.1 新增 application 层

- `application/__init__.py`
- `application/pipeline.py`

其中 `run_auto_lyric_job(...)` 作为 GUI 调用主流程的统一入口，内部转发到 `inference.auto_lyric_hybrid.auto_lyric_hybrid_pipeline(...)`。

### 3.2 GUI 调用迁移

- 文件：`gui/fluent_worker.py`
- 变更：从直接调用 `inference.auto_lyric_hybrid_pipeline` 改为调用 `application.run_auto_lyric_job`
- 结果：行为保持一致，但分层边界更清晰。

### 3.3 vendor 外置（激进拆分）

- 变更前：`inference/vendor/HubertFA`、`inference/vendor/LyricFA`
- 变更后：`third_party/inference_vendor/HubertFA`、`third_party/inference_vendor/LyricFA`
- 代码调整：
  - `inference/hfa_api.py` 改为从 `third_party/inference_vendor/HubertFA` 注入路径
  - `inference/lfa_api.py` 改为直接从 `third_party.inference_vendor.LyricFA...` 导入
  - `inference/auto_lyric_hybrid.py` 删除旧的 `inference/vendor` 路径注入

## 4. 后续阶段 B 建议（不在本次实施范围）

1. 将 `inference/auto_lyric_hybrid.py` 按 stage 拆分：
   - ASR Stage
   - FA Stage
   - Pitch/Note Stage
   - Export Stage
2. 统一取消机制、日志、异常封装。
3. 为 stage 增加最小单元测试与 smoke test。
