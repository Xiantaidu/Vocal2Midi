import pathlib

from inference.HubertFA.onnx_infer import InferenceOnnx


_HFA_REPAIR_IGNORE_TOKENS = {"SP", "AP", "EP", "br", "sil", "pau"}
_HFA_REPAIR_RATIO = 0.6
_HFA_REPAIR_MIN_GLOBAL_K = 3
_HFA_REPAIR_GLOBAL_PORTION = 0.15


def _is_lexical_word(word) -> bool:
    return getattr(word, "text", None) not in _HFA_REPAIR_IGNORE_TOKENS


def _repair_short_word_boundaries(words) -> list[str]:
    """
    Repair abnormally short lexical words produced by HFA.

    Rule of thumb:
    1. the current lexical word must be among the globally shortest few;
    2. it must also be locally much shorter than both surrounding lexical words;
    3. there must be a later lexical word, so the current word is not the last note.

    When triggered, the current word end is stretched to the next lexical word start,
    crossing over silence tags such as SP if necessary.
    """
    lexical_indices = [idx for idx, word in enumerate(words) if _is_lexical_word(word)]
    if len(lexical_indices) < 3:
        return []

    lexical_durations = [(idx, max(0.0, words[idx].end - words[idx].start)) for idx in lexical_indices]
    sorted_by_duration = sorted(lexical_durations, key=lambda item: item[1])
    shortest_k = max(_HFA_REPAIR_MIN_GLOBAL_K, int(len(lexical_indices) * _HFA_REPAIR_GLOBAL_PORTION + 0.999999))
    shortest_candidates = {idx for idx, _ in sorted_by_duration[:shortest_k]}

    repair_logs = []
    remove_indices = set()
    for pos in range(1, len(lexical_indices) - 1):
        cur_idx = lexical_indices[pos]
        if cur_idx in remove_indices or cur_idx not in shortest_candidates:
            continue

        prev_idx = lexical_indices[pos - 1]
        next_idx = lexical_indices[pos + 1]
        cur_word = words[cur_idx]
        prev_word = words[prev_idx]
        next_word = words[next_idx]
        middle_words = words[cur_idx + 1:next_idx]

        cur_dur = max(0.0, cur_word.end - cur_word.start)
        prev_dur = max(0.0, prev_word.end - prev_word.start)
        next_dur = max(0.0, next_word.end - next_word.start)
        neighbor_min = min(prev_dur, next_dur)

        if neighbor_min <= 0:
            continue
        if cur_dur >= neighbor_min * _HFA_REPAIR_RATIO:
            continue
        if next_word.start <= cur_word.end:
            continue
        if any(_is_lexical_word(word) for word in middle_words):
            continue

        old_end = cur_word.end
        cur_word.move_end(next_word.start)
        removed_texts = [word.text for word in middle_words]
        remove_indices.update(range(cur_idx + 1, next_idx))
        repair_logs.append(
            f"[HFA Repair] '{cur_word.text}' {cur_word.start:.4f}-{old_end:.4f} -> {cur_word.start:.4f}-{cur_word.end:.4f}"
            + (f"; removed fillers={removed_texts}" if removed_texts else "")
        )

    if remove_indices:
        kept_words = [word for idx, word in enumerate(words) if idx not in remove_indices]
        words.clear()
        words.extend(kept_words)

    return repair_logs


def _repair_pred_dict_short_words(pred_dict) -> None:
    total_repaired = 0
    for stem, pred in pred_dict.items():
        if len(pred) < 3:
            continue
        words = pred[2]
        repair_logs = _repair_short_word_boundaries(words)
        if repair_logs:
            print(f"[HFA Repair] {stem}: repaired {len(repair_logs)} short word(s)")
            for log in repair_logs:
                print(log)
            total_repaired += len(repair_logs)

    if total_repaired > 0:
        print(f"[HFA Repair] Total repaired short words: {total_repaired}")

def load_hfa_model(model_dir, device="cuda"):
    """
    Loads the HubertFA ONNX model ensuring it runs on the specified device (CUDA).
    """
    print("Loading HubertFA ONNX model for GPU...")
    model = InferenceOnnx(onnx_path=pathlib.Path(model_dir) / 'model.onnx')
    model.load_config()
    model.init_decoder()
    import onnxruntime as ort
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model.model = ort.InferenceSession(str(model.model_folder / 'model.onnx'), options, providers=providers)
    print(f"HubertFA ONNX session created with providers: {model.model.get_providers()}")
    return model

def run_hubert_fa(hfa_model, temp_dir, language="zh", cancel_checker=None, use_phoneme_g2p=False):
    print("[Hybrid Pipeline] Running HubertFA forced alignment on GPU...")
    if cancel_checker and cancel_checker():
        raise InterruptedError("HFA 任务已取消")
    hfa_model.dataset = []
    hfa_model.predictions = []
    
    dict_file = "ds-zh-pinyin-lite.txt" if language == "zh" else "japanese_dict_full.txt"
    dict_path = hfa_model.vocab_folder / dict_file

    if use_phoneme_g2p:
        # 输入 .lab 已经是音素 token。
        # 对日语额外做一层 "音素 -> mora词边界" 解析，这样 HFA 仍然按音素对齐，
        # 但 word 层级不再是单个 phoneme，而是更接近日语拍(mora)。
        g2p_mode = "ja_mora_phoneme" if language == "ja" else "phoneme"
        hfa_model.get_dataset(wav_folder=temp_dir, language=language, g2p=g2p_mode, dictionary_path=None)
    else:
        hfa_model.get_dataset(wav_folder=temp_dir, language=language, g2p="dictionary", dictionary_path=dict_path)
    if cancel_checker and cancel_checker():
        raise InterruptedError("HFA 任务已取消")
    if len(hfa_model.dataset) > 0:
        nl_phonemes = "AP" if language == "zh" else ""
        hfa_model.infer(non_lexical_phonemes=nl_phonemes, pad_times=1, pad_length=5)

    pred_dict = {p[0].stem: p for p in hfa_model.predictions}
    _repair_pred_dict_short_words(pred_dict)
    return pred_dict

def export_hfa_artifacts(chunks, temp_dir_path, hfa_model, output_key, output_dir, output_formats, cancel_checker=None):
    import shutil

    output_formats = set(output_formats or [])
    export_chunks = "chunks" in output_formats
                                           
    export_textgrid = ("textgrid" in output_formats) or export_chunks

    tg_subfolder = None
    if export_textgrid:
        if cancel_checker and cancel_checker():
            raise InterruptedError("HFA 导出任务已取消")
        temp_tg_dir = temp_dir_path / "temp_tg"
        hfa_model.export(temp_tg_dir, output_format=['textgrid'])
        tg_subfolder = temp_tg_dir / "TextGrid"
    
    for chunk_idx, chunk in enumerate(chunks):
        if cancel_checker and cancel_checker():
            raise InterruptedError("HFA 导出任务已取消")
        stem = f"chunk_{chunk_idx}"
        new_stem = f"{output_key}_{chunk_idx:03d}"
        
        if export_chunks:
            chunk_wav_path = temp_dir_path / f"{stem}.wav"
            try:
                if chunk_wav_path.exists():
                    shutil.copy2(chunk_wav_path, output_dir / f"{new_stem}.wav")
                else:
                    print(f"[Warning] Chunk WAV file not found, skipping: {chunk_wav_path}")
            except Exception as e:
                print(f"[Error] Failed to copy chunk {chunk_wav_path}: {e}")

        if tg_subfolder is not None and tg_subfolder.exists():
            tg_path = tg_subfolder / f"{stem}.TextGrid"
            try:
                if tg_path.exists():
                    shutil.copy2(tg_path, output_dir / f"{new_stem}.TextGrid")
            except Exception as e:
                print(f"[Error] Failed to copy TextGrid {tg_path}: {e}")
