import pathlib
import sys

# Add vendor paths
VENDOR_DIR = pathlib.Path(__file__).parent / "vendor"
if str(VENDOR_DIR / "HubertFA") not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR / "HubertFA"))

def load_hfa_model(model_dir, device="cuda"):
    """
    Loads the HubertFA ONNX model ensuring it runs on the specified device (CUDA).
    """
    print("Loading HubertFA ONNX model for GPU...")
    from onnx_infer import InferenceOnnx
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

def run_hubert_fa(hfa_model, temp_dir, language="zh", cancel_checker=None):
    print("[Hybrid Pipeline] Running HubertFA forced alignment on GPU...")
    if cancel_checker and cancel_checker():
        raise InterruptedError("HFA 任务已取消")
    hfa_model.dataset = []
    hfa_model.predictions = []
    
    dict_file = "ds-zh-pinyin-lite.txt" if language == "zh" else "japanese_dict_full.txt"
    dict_path = hfa_model.vocab_folder / dict_file
    
    hfa_model.get_dataset(wav_folder=temp_dir, language=language, g2p="dictionary", dictionary_path=dict_path)
    if cancel_checker and cancel_checker():
        raise InterruptedError("HFA 任务已取消")
    if len(hfa_model.dataset) > 0:
        nl_phonemes = "AP" if language == "zh" else ""
        hfa_model.infer(non_lexical_phonemes=nl_phonemes, pad_times=1, pad_length=5)

    pred_dict = {p[0].stem: p for p in hfa_model.predictions}
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
