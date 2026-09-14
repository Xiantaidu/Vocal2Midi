"""Lightweight UI translations for the Fluent GUI.

`tr(key)` returns the text for the active language; widgets bind a zero-arg
refresh callable via `_bind_tr`-style helpers so a language switch can
retranslate the whole UI live without recreating widgets.
"""
from __future__ import annotations

current_language = "zh"

_STRINGS: dict[str, dict[str, str]] = {
    # ── main window ────────────────────────────────────────────────
    "nav_auto": {"zh": "自动提取与灌注", "en": "Auto Extract & Lyrics"},
    "nav_model": {"zh": "模型配置", "en": "Model Config"},
    "nav_settings": {"zh": "全局设置", "en": "Settings"},
    "close_running_title": {"zh": "任务仍在运行", "en": "Task Still Running"},
    "close_running_body": {
        "zh": "提取任务尚未结束，关闭窗口将强制停止任务。确定关闭吗？",
        "en": "Extraction is still running. Closing the window will force-stop it. Close anyway?",
    },
    # ── auto lyric page ────────────────────────────────────────────
    "app_title": {"zh": "自动提取与歌词灌注", "en": "Auto Extract & Lyrics"},
    "upload_audio": {"zh": "上传音频文件", "en": "Upload Audio Files"},
    "pick_files": {"zh": "选择文件", "en": "Add Files"},
    "pick_files_dialog": {"zh": "选择音频文件", "en": "Select Audio Files"},
    "clear_files": {"zh": "清空选择", "en": "Clear"},
    "ref_lyrics": {"zh": "参考歌词 (可选)", "en": "Reference Lyrics (Optional)"},
    "ref_lyrics_hint": {
        "zh": "如果有确切的歌词，请在此输入（纯文本）以提高对齐准确率...",
        "en": "Enter exact lyrics here (plain text) to improve alignment...",
    },
    "slicing_method": {"zh": "音频切片方法", "en": "Slicing"},
    "slice_smart": {"zh": "智能切片", "en": "Smart"},
    "slice_heuristic": {"zh": "启发式切片", "en": "Heuristic"},
    "slice_default": {"zh": "默认切片", "en": "Default"},
    "slice_grid": {"zh": "网格搜索切片", "en": "Grid Search"},
    "target_lang": {"zh": "目标语言", "en": "Language"},
    "lang_name_zh": {"zh": "zh", "en": "zh"},
    "lang_name_zh_pinyin": {"zh": "中文-拼音", "en": "Chinese (Pinyin)"},
    "lang_name_ja": {"zh": "ja", "en": "ja"},
    "lang_name_en": {"zh": "en", "en": "en"},
    "lyric_output_format": {"zh": "歌词输出格式", "en": "Lyric Format"},
    "opt_pinyin": {"zh": "拼音", "en": "Pinyin"},
    "opt_hanzi": {"zh": "汉字", "en": "Hanzi"},
    "opt_romaji": {"zh": "罗马音", "en": "Romaji"},
    "opt_kana": {"zh": "假名", "en": "Kana"},
    "opt_word": {"zh": "单词", "en": "Word"},
    "device": {"zh": "计算设备", "en": "Device"},
    "match_lyrics": {"zh": "启用歌词匹配", "en": "Lyric Match"},
    "output_lyrics": {"zh": "输出歌词", "en": "Output Lyrics"},
    "export_format": {"zh": "导出格式", "en": "Export"},
    "pitch_curve": {"zh": "输出音高曲线", "en": "Pitch Curve"},
    "output_settings": {"zh": "输出设置", "en": "Output Settings"},
    "tempo_bpm": {"zh": "Tempo BPM:", "en": "Tempo BPM:"},
    "quant_step": {"zh": "MIDI 量化精度:", "en": "Quantization:"},
    "quant_off": {"zh": "不量化", "en": "Off"},
    "quant_1_4": {"zh": "1/4 音符 (1拍)", "en": "1/4 note (1 beat)"},
    "quant_1_8": {"zh": "1/8 音符 (1/2拍)", "en": "1/8 note (1/2 beat)"},
    "quant_1_16": {"zh": "1/16 音符 (1/4拍)", "en": "1/16 note (1/4 beat)"},
    "quant_1_32": {"zh": "1/32 音符 (1/8拍)", "en": "1/32 note (1/8 beat)"},
    "quant_1_64": {"zh": "1/64 音符 (1/16拍)", "en": "1/64 note (1/16 beat)"},
    "quant_mode": {"zh": "量化算法:", "en": "Algorithm:"},
    "quant_repair": {"zh": "节奏修复", "en": "Rhythm Repair"},
    "quant_bayes": {"zh": "贝叶斯", "en": "Bayesian"},
    "quant_dp": {"zh": "DP", "en": "DP"},
    "quant_simple": {"zh": "简单", "en": "Simple"},
    "save_dir": {"zh": "保存目录:", "en": "Save Directory:"},
    "browse": {"zh": "浏览", "en": "Browse"},
    "choose_folder_dialog": {"zh": "选择文件夹", "en": "Select Folder"},
    "run": {"zh": "开始全自动提取", "en": "Start Extraction"},
    "stop": {"zh": "强制停止", "en": "Force Stop"},
    "files_added": {"zh": "已添加 {n} 个音频文件", "en": "{n} audio file(s) added"},
    "err_hybrid": {
        "zh": "错误: 混合管线未能正确加载，请检查环境。",
        "en": "Error: hybrid pipeline failed to load, please check the environment.",
    },
    "err_cannot_start": {"zh": "无法开始", "en": "Cannot Start"},
    "err_no_audio": {"zh": "请至少上传一个音频文件", "en": "Please add at least one audio file"},
    "err_no_dir": {"zh": "请选择或输入一个保存目录", "en": "Please select or enter a save directory"},
    "info_dir_created": {
        "zh": "提示: 保存目录不存在，已自动创建 {dir}",
        "en": "Note: save directory did not exist; created {dir}",
    },
    "err_dir_create": {"zh": "无法创建保存目录", "en": "Failed to Create Save Directory"},
    "err_dir_not_dir": {
        "zh": "您输入的保存路径已存在但不是一个目录",
        "en": "The save path exists but is not a directory",
    },
    "preparing": {"zh": "准备处理 {n} 个文件...", "en": "Preparing to process {n} file(s)..."},
    "processing": {"zh": "处理中 {i}/{n}: {f}", "en": "Processing {i}/{n}: {f}"},
    "done_title": {"zh": "提取完成", "en": "Extraction Complete"},
    "stopping": {"zh": "正在停止...", "en": "Stopping..."},
    "stop_requested_title": {"zh": "已请求强制停止", "en": "Force Stop Requested"},
    "stop_requested_body": {"zh": "正在尽快中断当前流程", "en": "Interrupting the current pipeline as soon as possible"},
    "stop_hint": {
        "zh": "提示: 若正在执行底层模型调用，需等待当前批次返回后停止。",
        "en": "Note: if a low-level model call is in progress, stopping waits for the current batch to return.",
    },
    "fail_title": {"zh": "任务失败", "en": "Task Failed"},
    "fail_body": {"zh": "发生错误，详情请查看运行日志", "en": "An error occurred; see the run log for details"},
    "error_prefix": {"zh": "错误", "en": "Error"},
    # ── worker thread ──────────────────────────────────────────────
    "worker_processing": {"zh": "========== 正在处理: {f} ==========", "en": "========== Processing: {f} =========="},
    "worker_success": {"zh": "提取成功！文件已保存至: {d}", "en": "Extraction succeeded! Files saved to: {d}"},
    "worker_cancelled": {"zh": "任务已被取消。", "en": "Task was cancelled."},
    "worker_stopped": {"zh": "任务已被强制停止。", "en": "Task was force-stopped."},
    "worker_error": {"zh": "发生错误:\n{tb}", "en": "An error occurred:\n{tb}"},
    # ── global settings page ───────────────────────────────────────
    "settings_title": {"zh": "全局设置", "en": "Global Settings"},
    "reset_defaults": {"zh": "恢复默认", "en": "Restore Defaults"},
    "appearance": {"zh": "外观", "en": "Appearance"},
    "theme": {"zh": "界面主题:", "en": "Theme:"},
    "theme_light": {"zh": "浅色", "en": "Light"},
    "theme_dark": {"zh": "深色", "en": "Dark"},
    "theme_auto": {"zh": "跟随系统", "en": "Follow System"},
    "language": {"zh": "语言:", "en": "Language:"},
    "language_zh": {"zh": "中文", "en": "中文"},
    "language_en": {"zh": "English", "en": "English"},
    "adv_params": {"zh": "高级处理参数", "en": "Advanced Processing Parameters"},
    "seg_thresh": {"zh": "边界解码阈值:", "en": "Decode Threshold:"},
    "seg_rad": {"zh": "边界解码半径/秒:", "en": "Decode Radius (s):"},
    "est_thresh": {"zh": "音符存在阈值:", "en": "Note Threshold:"},
    "d3pm_t0": {"zh": "D3PM 起始 T 值:", "en": "D3PM Start T:"},
    "d3pm_nsteps": {"zh": "D3PM 采样步数:", "en": "D3PM Sampling Steps:"},
    "game_batch": {"zh": "GAME Batch:", "en": "GAME Batch:"},
    "asr_batch": {"zh": "ASR Batch:", "en": "ASR Batch:"},
    "slice_min": {"zh": "Slice Min (s):", "en": "Slice Min (s):"},
    "slice_max": {"zh": "Slice Max (s):", "en": "Slice Max (s):"},
    "debug": {"zh": "Debug", "en": "Debug"},
    "export_txt": {"zh": "导出 Text (.txt):", "en": "Export Text (.txt):"},
    "export_csv": {"zh": "导出 CSV (.csv):", "en": "Export CSV (.csv):"},
    "export_chunks": {"zh": "导出切片:", "en": "Export Chunks:"},
    "pitch_format": {"zh": "音高格式:", "en": "Pitch Format:"},
    "round_pitch": {"zh": "音高取整:", "en": "Round Pitch:"},
    # ── model config page ──────────────────────────────────────────
    "model_title": {"zh": "模型配置", "en": "Model Configuration"},
    "game_path": {"zh": "GAME 模型路径:", "en": "GAME Model Path:"},
    "hfa_path": {"zh": "HubertFA模型路径:", "en": "HubertFA Model Path:"},
    "asr_path": {"zh": "Qwen3-ASR模型路径:", "en": "Qwen3-ASR Model Path:"},
    "phoneme_path": {"zh": "音素ASR模型路径:", "en": "Phoneme ASR Model Path:"},
    "pinyin_path": {"zh": "拼音ASR模型路径:", "en": "Pinyin ASR Model Path:"},
    "rmvpe_path": {"zh": "RMVPE模型路径:", "en": "RMVPE Model Path:"},
}


def set_language(language: str):
    global current_language
    current_language = "en" if str(language or "").strip().lower() in {"en", "english"} else "zh"


def tr(key: str, **kwargs) -> str:
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(current_language) or entry["zh"]
    if kwargs:
        text = text.format(**kwargs)
    return text
