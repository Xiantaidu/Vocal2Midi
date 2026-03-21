import gradio as gr
import pathlib
import os
import glob
import tempfile
import zipfile
import shutil
import sys

# Ensure current directory is in sys.path for portable environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Optional PyTorch imports
try:
    from lib.config.schema import ValidationConfig
    from inference.api import load_inference_model, infer_model
    from inference.data import SlicedAudioFileIterableDataset, DiffSingerTranscriptionsDataset
    from inference.callbacks import (
        SaveCombinedMidiFileCallback, 
        SaveCombinedTextFileCallback,
        UpdateDiffSingerTranscriptionsCallback
    )
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("WARNING: PyTorch is not available. Only ONNX engine will work.")

from inference.onnx_api import load_onnx_model, infer_from_files, align_with_transcriptions
from inference.slicer2 import Slicer
from inference.auto_lyric import auto_lyric_pipeline

def _get_language_id(language: str, lang_map: dict[str, int]) -> int:
    if language and lang_map:
        if language not in lang_map:
            raise ValueError(
                f"分割模型不支持语言 '{language}'。 "
                f"支持的语言: {', '.join(lang_map.keys())}"
            )
        language_id = lang_map[language]
    else:
        language_id = 0
    return language_id

def _t0_nstep_to_ts(t0: float, nsteps: int) -> list[float]:
    step = (1 - t0) / nsteps
    return [
        t0 + i * step
        for i in range(nsteps)
    ]

def load_model(model_path_str: str, engine: str, onnx_device: str):
    model_path = pathlib.Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
    if engine == "PyTorch":
        if not PYTORCH_AVAILABLE:
            raise ValueError("未安装 PyTorch。请使用 ONNX 引擎。")
        model, lang_map = load_inference_model(model_path)
    elif engine == "ONNX":
        model = load_onnx_model(model_path, device=onnx_device)
        lang_map = model.languages
    else:
        raise ValueError(f"不支持的引擎: {engine}")
        
    return model, lang_map

def release_memory():
    import gc
    gc.collect()
    if PYTORCH_AVAILABLE:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

def extract_midi(
    audio_files,
    model_path_str,
    engine,
    onnx_device,
    language,
    batch_size,
    seg_threshold,
    seg_radius,
    t0,
    nsteps,
    est_threshold,
    output_mid,
    output_txt,
    output_csv,
    tempo,
    quantize_option,
    pitch_format,
    round_pitch
):
    model = None
    try:
        if not audio_files:
            return None, "请至少上传一个音频文件。"
        
        if not model_path_str:
            return None, "请指定模型检查点路径。"

        # Parse outputs
        output_formats = set()
        if output_mid: output_formats.add("mid")
        if output_txt: output_formats.add("txt")
        if output_csv: output_formats.add("csv")
        
        if not output_formats:
            return None, "请至少选择一种输出格式。"

        # Create temporary directory for output
        output_dir = pathlib.Path(tempfile.mkdtemp(prefix="game_gradio_extract_"))
        
        # Prepare input files map
        filemap = {}
        for temp_file in audio_files:
            original_path = pathlib.Path(temp_file.name)
            filename = original_path.name
            if hasattr(temp_file, 'orig_name') and temp_file.orig_name:
                filename = temp_file.orig_name
            filemap[filename] = original_path

        # Load model
        model, lang_map = load_model(model_path_str, engine, onnx_device)
        language_id = _get_language_id(language, lang_map)

        ts = _t0_nstep_to_ts(t0, int(nsteps))

        if engine == "PyTorch":
            if not PYTORCH_AVAILABLE:
                return None, "未安装 PyTorch，请在推理引擎中选择 ONNX。"
            sr = model.inference_config.features.audio_sample_rate
            dataset = SlicedAudioFileIterableDataset(
                filemap=filemap,
                samplerate=sr,
                slicer=Slicer(
                    sr=sr,
                    threshold=-40.,
                    min_length=1000,
                    min_interval=200,
                    max_sil_kept=100,
                ),
                language=language_id,
            )

            # Parse quantization
            quantization_step = 0
            if "1/4 音符" in quantize_option:
                quantization_step = 480
            elif "1/8 音符" in quantize_option:
                quantization_step = 240
            elif "1/16 音符" in quantize_option:
                quantization_step = 120
            elif "1/32 音符" in quantize_option:
                quantization_step = 60
            elif "1/64 音符" in quantize_option:
                quantization_step = 30

            callbacks = []
            if "mid" in output_formats:
                callbacks.append(SaveCombinedMidiFileCallback(
                    output_dir=output_dir,
                    tempo=tempo,
                    quantization_step=quantization_step,
                ))
            if "txt" in output_formats:
                callbacks.append(SaveCombinedTextFileCallback(
                    output_dir=output_dir,
                    file_format="txt",
                    pitch_format=pitch_format,
                    round_pitch=round_pitch,
                ))
            if "csv" in output_formats:
                callbacks.append(SaveCombinedTextFileCallback(
                    output_dir=output_dir,
                    file_format="csv",
                    pitch_format=pitch_format,
                    round_pitch=round_pitch,
                ))

            # Run inference
            infer_model(
                model=model,
                dataset=dataset,
                config=ValidationConfig(
                    d3pm_sample_ts=ts,
                    boundary_decoding_threshold=seg_threshold,
                    boundary_decoding_radius=round(seg_radius / model.timestep),
                    note_presence_threshold=est_threshold,
                ),
                batch_size=int(batch_size),
                num_workers=0,
                callbacks=callbacks,
            )
        elif engine == "ONNX":
            infer_from_files(
                model=model,
                filemap=filemap,
                output_dir=output_dir,
                output_formats=output_formats,
                language_id=language_id,
                seg_threshold=seg_threshold,
                seg_radius=seg_radius,
                ts=ts,
                est_threshold=est_threshold,
                pitch_format=pitch_format,
                round_pitch=round_pitch,
                tempo=tempo,
                quantization_step=quantization_step,
                batch_size=int(batch_size),
            )

        # Collect output files
        generated_files = list(output_dir.glob("*"))
        if not generated_files:
            return None, "推理完成，但未生成任何输出文件。"
            
        if len(generated_files) == 1:
            return str(generated_files[0]), "提取成功！"
        else:
            zip_path = output_dir.parent / "extracted_midi.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in generated_files:
                    zipf.write(file, file.name)
            return str(zip_path), "提取成功！请下载 ZIP 压缩包。"

    except Exception as e:
        return None, f"发生错误: {str(e)}"
    finally:
        if model is not None:
            if hasattr(model, 'release'):
                model.release()
            del model
        release_memory()

def auto_lyric(
    audio_files,
    game_model_path_str,
    hfa_model_path_str,
    engine,
    onnx_device,
    language,
    original_lyrics,
    batch_size,
    seg_threshold,
    seg_radius,
    t0,
    nsteps,
    est_threshold,
    output_mid,
    output_txt,
    output_csv,
    output_chunks,
    tempo,
    quantize_option,
    pitch_format,
    round_pitch
):
    model = None
    try:
        if not audio_files:
            return None, "请至少上传一个音频文件。"
        if not game_model_path_str:
            return None, "请指定 GAME 模型路径。"
        if not hfa_model_path_str:
            return None, "请指定 HubertFA ONNX 模型路径。"
        if engine != "ONNX":
            return None, "自动歌词功能当前仅支持 ONNX 引擎。"

        output_formats = []
        if output_mid: output_formats.append("mid")
        if output_txt: output_formats.append("txt")
        if output_csv: output_formats.append("csv")
        if output_chunks: output_formats.append("chunks")
        
        if not output_formats:
            return None, "请至少选择一种输出格式。"

        output_dir = pathlib.Path(tempfile.mkdtemp(prefix="game_gradio_autolyric_"))

        # Load GAME model
        model = load_onnx_model(pathlib.Path(game_model_path_str), device=onnx_device)

        quantization_step = 0
        if "1/4 音符" in quantize_option: quantization_step = 480
        elif "1/8 音符" in quantize_option: quantization_step = 240
        elif "1/16 音符" in quantize_option: quantization_step = 120
        elif "1/32 音符" in quantize_option: quantization_step = 60
        elif "1/64 音符" in quantize_option: quantization_step = 30
        
        ts = _t0_nstep_to_ts(t0, int(nsteps))

        for temp_file in audio_files:
            original_path = pathlib.Path(temp_file.name)
            filename = original_path.name
            if hasattr(temp_file, 'orig_name') and temp_file.orig_name:
                filename = temp_file.orig_name
                
            print(f"Auto Lyric Processing: {filename}")
            
            auto_lyric_pipeline(
                audio_path=str(original_path),
                output_filename=filename,
                game_model=model,
                hfa_onnx_path=hfa_model_path_str,
                language=language,
                original_lyrics=original_lyrics,
                output_dir=output_dir,
                output_formats=output_formats,
                tempo=tempo,
                quantization_step=quantization_step,
                pitch_format=pitch_format,
                round_pitch=round_pitch,
                seg_threshold=seg_threshold,
                seg_radius=seg_radius,
                est_threshold=est_threshold,
                ts=ts,
                batch_size=int(batch_size)
            )

        generated_files = list(output_dir.glob("*"))
        if not generated_files:
            return None, "推理完成，但未生成任何输出文件。"
            
        if len(generated_files) == 1:
            return str(generated_files[0]), "自动灌词成功！"
        else:
            zip_path = output_dir.parent / "autolyric_results.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in generated_files:
                    zipf.write(file, file.name)
            return str(zip_path), "自动灌词成功！请下载 ZIP 压缩包。"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"发生错误: {str(e)}"
    finally:
        if model is not None:
            if hasattr(model, 'release'):
                model.release()
            del model
        release_memory()


def align_transcriptions(
    csv_files,
    model_path_str,
    engine,
    onnx_device,
    language,
    batch_size,
    seg_threshold,
    seg_radius,
    t0,
    nsteps,
    est_threshold,
    no_wb,
    uv_vocab_str,
    uv_word_cond,
    uv_note_cond
):
    model = None
    try:
        if not csv_files:
            return None, "请上传至少一个 DiffSinger 转录 CSV 文件。"
            
        if not model_path_str:
            return None, "请指定模型检查点路径。"

        # Create temporary directory for output
        output_dir = pathlib.Path(tempfile.mkdtemp(prefix="game_gradio_align_"))
        
        # Copy input files to temp dir to avoid modifying original uploads directly in gradio's temp
        paths = []
        for temp_file in csv_files:
            original_path = pathlib.Path(temp_file.name)
            filename = original_path.name
            if hasattr(temp_file, 'orig_name') and temp_file.orig_name:
                filename = temp_file.orig_name
            
            dest_path = output_dir / filename
            shutil.copy2(original_path, dest_path)
            paths.append(dest_path)

        # Parse UV options
        use_wb = not no_wb
        if uv_vocab_str and uv_vocab_str.strip():
            uv_vocab = {v.strip() for v in uv_vocab_str.split(",") if v.strip()}
        else:
            uv_vocab = None
        
        resolved_uv_vocab = uv_vocab if use_wb else None

        # Load model
        model, lang_map = load_model(model_path_str, engine, onnx_device)
        language_id = _get_language_id(language, lang_map)

        ts = _t0_nstep_to_ts(t0, int(nsteps))
        
        if engine == "PyTorch":
            if not PYTORCH_AVAILABLE:
                return None, "未安装 PyTorch，请在推理引擎中选择 ONNX。"
            sr = model.inference_config.features.audio_sample_rate
            dataset = DiffSingerTranscriptionsDataset(
                filelist=paths,
                samplerate=sr,
                language=language_id,
                use_wb=use_wb,
                uv_vocab=resolved_uv_vocab,
                uv_word_cond=uv_word_cond,
            )
            
            callbacks = [
                UpdateDiffSingerTranscriptionsCallback(
                    filelist=paths,
                    overwrite=True, # Overwrite the copied files in our temp dir
                    save_dir=None,
                    save_filename=None,
                    use_wb=use_wb,
                    uv_vocab=resolved_uv_vocab,
                    uv_word_cond=uv_word_cond,
                    uv_note_cond=uv_note_cond,
                )
            ]

            # Run inference
            infer_model(
                model=model,
                dataset=dataset,
                config=ValidationConfig(
                    d3pm_sample_ts=ts,
                    boundary_decoding_threshold=seg_threshold,
                    boundary_decoding_radius=round(seg_radius / model.timestep),
                    note_presence_threshold=est_threshold,
                ),
                batch_size=int(batch_size),
                num_workers=0,
                callbacks=callbacks,
            )
        elif engine == "ONNX":
            align_with_transcriptions(
                model=model,
                transcription_paths=paths,
                language_id=language_id,
                seg_threshold=seg_threshold,
                seg_radius=seg_radius,
                ts=ts,
                est_threshold=est_threshold,
                use_wb=use_wb,
                inplace=True,
                save_dir=None,
                save_name=None,
                batch_size=int(batch_size),
                uv_vocab=resolved_uv_vocab,
                uv_word_cond=uv_word_cond,
                uv_note_cond=uv_note_cond,
            )

        # The files in output_dir are now updated
        updated_files = list(output_dir.glob("*.csv"))
        if not updated_files:
            return None, "对齐处理完成，但未找到输出文件。"
            
        if len(updated_files) == 1:
            return str(updated_files[0]), "对齐处理成功！"
        else:
            zip_path = output_dir.parent / "aligned_transcriptions.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in updated_files:
                    zipf.write(file, file.name)
            return str(zip_path), "对齐处理成功！请下载 ZIP 压缩包。"

    except Exception as e:
        return None, f"发生错误: {str(e)}"
    finally:
        if model is not None:
            if hasattr(model, 'release'):
                model.release()
            del model
        release_memory()

# Custom CSS
css = """
.container { max-width: 1200px; margin: auto; }
"""

with gr.Blocks(title="GAME: 生成式自适应 MIDI 提取器") as demo:
    gr.Markdown("# 🎵 GAME: 生成式自适应 MIDI 提取器 (推理界面)")
    gr.Markdown("将歌声转换为乐谱（MIDI）。基于 D3PM 模型。支持提取原始音频和对齐 DiffSinger 数据集。")
    
    with gr.Row():
        model_path_input = gr.Textbox(label="模型路径 (Model Checkpoint / ONNX Dir)", placeholder="/path/to/model.pt 或 /path/to/onnx_dir", value="experiments/model.pt", scale=3)
        language_input = gr.Textbox(label="语言代码 (选填, 例如: zh)", placeholder="zh", scale=1)
        
    with gr.Row():
        engine_radio = gr.Radio(choices=["PyTorch", "ONNX"], value="PyTorch", label="推理引擎 (Inference Engine)")
        onnx_device_radio = gr.Radio(choices=["dml", "cpu"], value="dml", label="ONNX 设备 (Device, 仅 ONNX 引擎有效)", visible=True)

    with gr.Accordion("⚙️ 高级模型参数 (Advanced Parameters)", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 分割模型 (Segmentation Model)")
                seg_threshold_slider = gr.Slider(minimum=0.01, maximum=0.99, value=0.2, step=0.01, label="边界解码阈值 (Boundary Decoding Threshold)")
                seg_radius_slider = gr.Slider(minimum=0.01, maximum=0.1, value=0.02, step=0.005, label="边界解码半径/秒 (Boundary Decoding Radius)")
                t0_slider = gr.Slider(minimum=0.0, maximum=0.99, value=0.0, step=0.01, label="D3PM 起始 T 值 (Starting t0)")
                nsteps_slider = gr.Slider(minimum=1, maximum=20, value=8, step=1, label="D3PM 采样步数 (Sampling Steps)")
            with gr.Column():
                gr.Markdown("### 估计与通用 (Estimation & General)")
                est_threshold_slider = gr.Slider(minimum=0.01, maximum=0.99, value=0.2, step=0.01, label="音符存在阈值 (Note Presence Threshold)")
                batch_size_slider = gr.Slider(minimum=1, maximum=32, value=4, step=1, label="批处理大小 (Batch Size)")

    with gr.Tabs():
        with gr.TabItem("🎙️ 提取原始音频 (Extract Audio)"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.File(label="上传音频文件 (wav, flac, mp3 等)", file_count="multiple", type="filepath")
                    
                    with gr.Accordion("输出设置 (Output Options)", open=True):
                        with gr.Row():
                            out_mid_cb = gr.Checkbox(label="MIDI (.mid)", value=True)
                            out_txt_cb = gr.Checkbox(label="Text (.txt)", value=False)
                            out_csv_cb = gr.Checkbox(label="CSV (.csv)", value=False)
                        
                        tempo_number = gr.Number(label="曲速 (Tempo BPM)", value=120)
                        quantize_dropdown = gr.Dropdown(
                            choices=["不量化", "1/4 音符 (1拍)", "1/8 音符 (1/2拍)", "1/16 音符 (1/4拍) [推荐]", "1/32 音符 (1/8拍) [推荐]", "1/64 音符 (1/16拍)"],
                            value="1/32 音符 (1/8拍) [推荐]",
                            label="MIDI 量化 (Quantization)",
                            info="Vocaloid/SV/UTAU 建议选 1/32 或 1/16 音符。过大会吞字，过小会很碎。"
                        )
                        pitch_format_radio = gr.Radio(choices=["name", "number"], value="name", label="音高格式 (用于 Text/CSV)")
                        round_pitch_cb = gr.Checkbox(label="音高取整 (Round Pitch)", value=False)
                        
                    with gr.Row():
                        extract_btn = gr.Button("🚀 提取 MIDI", variant="primary")
                        extract_stop_btn = gr.Button("🛑 强制停止", variant="stop")
                    
                with gr.Column(scale=1):
                    extract_output_file = gr.File(label="下载提取结果 (Download Result)")
                    extract_msg = gr.Textbox(label="状态信息 (Status)", interactive=False)

            extract_event = extract_btn.click(
                fn=extract_midi,
                inputs=[
                    audio_input, model_path_input, engine_radio, onnx_device_radio, language_input, batch_size_slider,
                    seg_threshold_slider, seg_radius_slider, t0_slider, nsteps_slider, est_threshold_slider,
                    out_mid_cb, out_txt_cb, out_csv_cb, tempo_number, quantize_dropdown, pitch_format_radio, round_pitch_cb
                ],
                outputs=[extract_output_file, extract_msg]
            )
            extract_stop_btn.click(fn=None, cancels=[extract_event])

        with gr.TabItem("🎤 自动歌词与灌注 (Auto Lyric)"):
            gr.Markdown("结合 FunASR 和 HubertFA，自动识别歌词并基于**元音起始点**进行音符分割和歌词灌注。支持导出带歌词的 MIDI。")
            with gr.Row():
                with gr.Column(scale=1):
                    al_audio_input = gr.File(label="上传音频文件 (wav, flac 等)", file_count="multiple", type="filepath")
                    al_hfa_model_input = gr.Textbox(label="HubertFA ONNX 模型路径", placeholder="/path/to/hubertfa_model.onnx", value=R"E:\Vocal2Midi\experiments\1218_hfa_model_new_dict\model.onnx")
                    al_lyrics_input = gr.Textbox(label="参考歌词 (可选)", placeholder="如果有确切的歌词，请在此输入（纯文本），将使用 LyricFA 纠正 ASR 结果。", lines=4)
                    
                    with gr.Accordion("输出设置 (Output Options)", open=True):
                        with gr.Row():
                            al_out_mid_cb = gr.Checkbox(label="带歌词的 MIDI (.mid)", value=True)
                            al_out_txt_cb = gr.Checkbox(label="Text (.txt)", value=False)
                            al_out_csv_cb = gr.Checkbox(label="CSV (.csv)", value=False)
                            al_out_chunks_cb = gr.Checkbox(label="切片与 TextGrid (.wav, .TextGrid)", value=False)
                        
                        al_tempo_number = gr.Number(label="曲速 (Tempo BPM)", value=120)
                        al_quantize_dropdown = gr.Dropdown(
                            choices=["不量化", "1/4 音符 (1拍)", "1/8 音符 (1/2拍)", "1/16 音符 (1/4拍) [推荐]", "1/32 音符 (1/8拍) [推荐]", "1/64 音符 (1/16拍)"],
                            value="不量化",
                            label="MIDI 量化 (Quantization)"
                        )
                        al_pitch_format_radio = gr.Radio(choices=["name", "number"], value="name", label="音高格式 (用于 Text/CSV)")
                        al_round_pitch_cb = gr.Checkbox(label="音高取整 (Round Pitch)", value=False)
                        
                    with gr.Row():
                        al_btn = gr.Button("🎤 开始自动灌词", variant="primary")
                        al_stop_btn = gr.Button("🛑 强制停止", variant="stop")
                    
                with gr.Column(scale=1):
                    al_output_file = gr.File(label="下载提取结果 (Download Result)")
                    al_msg = gr.Textbox(label="状态信息 (Status)", interactive=False)

            al_event = al_btn.click(
                fn=auto_lyric,
                inputs=[
                    al_audio_input, model_path_input, al_hfa_model_input, engine_radio, onnx_device_radio, language_input, al_lyrics_input,
                    batch_size_slider, seg_threshold_slider, seg_radius_slider, t0_slider, nsteps_slider, est_threshold_slider,
                    al_out_mid_cb, al_out_txt_cb, al_out_csv_cb, al_out_chunks_cb, al_tempo_number, al_quantize_dropdown, al_pitch_format_radio, al_round_pitch_cb
                ],
                outputs=[al_output_file, al_msg]
            )
            al_stop_btn.click(fn=None, cancels=[al_event])

        with gr.TabItem("📝 对齐数据集 (Align Datasets)"):
            gr.Markdown("处理 DiffSinger 数据集格式。生成带有词边界的对齐音符标签。")
            with gr.Row():
                with gr.Column(scale=1):
                    csv_input = gr.File(label="上传 transcriptions.csv 文件", file_count="multiple", type="filepath")
                    
                    with gr.Accordion("对齐选项 (Alignment Options)", open=True):
                        uv_vocab_input = gr.Textbox(label="清音音素集 (Unvoiced Vocab)", value="AP,SP,br,sil", info="用逗号分隔的清音音素列表。")
                        with gr.Row():
                            uv_word_cond_radio = gr.Radio(choices=["lead", "all"], value="lead", label="清音词判断条件 (UV Word Cond)", info="lead: 首音素为清音; all: 所有音素为清音")
                            uv_note_cond_radio = gr.Radio(choices=["predict", "follow"], value="predict", label="清音音符判断条件 (UV Note Cond)", info="predict: 模型预测; follow: 跟随词的清浊状态")
                        no_wb_cb = gr.Checkbox(label="禁用词边界 (Disable Word Boundaries / no-wb)", value=False, info="不推荐勾选。如果勾选，将不检查和使用 'ph_num' 字段。")
                        
                    with gr.Row():
                        align_btn = gr.Button("⚡ 开始对齐", variant="primary")
                        align_stop_btn = gr.Button("🛑 强制停止", variant="stop")
                    
                with gr.Column(scale=1):
                    align_output_file = gr.File(label="下载更新后的 CSV")
                    align_msg = gr.Textbox(label="状态信息 (Status)", interactive=False)

            align_event = align_btn.click(
                fn=align_transcriptions,
                inputs=[
                    csv_input, model_path_input, engine_radio, onnx_device_radio, language_input, batch_size_slider,
                    seg_threshold_slider, seg_radius_slider, t0_slider, nsteps_slider, est_threshold_slider,
                    no_wb_cb, uv_vocab_input, uv_word_cond_radio, uv_note_cond_radio
                ],
                outputs=[align_output_file, align_msg]
            )
            align_stop_btn.click(fn=None, cancels=[align_event])

    def on_engine_change(engine):
        if engine == "ONNX":
            return gr.update(value="experiments/GAME-1.0.3-medium-onnx")
        else:
            return gr.update(value="experiments/model.pt")
            
    engine_radio.change(fn=on_engine_change, inputs=engine_radio, outputs=model_path_input)

    def on_device_change(device):
        if device == "cpu":
            return gr.update(value=1)
        else:
            return gr.update(value=4)
            
    onnx_device_radio.change(fn=on_device_change, inputs=onnx_device_radio, outputs=batch_size_slider)

if __name__ == "__main__":
    print("正在启动 GAME Gradio 界面...")
    
    port = 7860
    while port < 7960:
        try:
            demo.launch(server_name="0.0.0.0", server_port=port, share=False, inbrowser=True, css=css)
            break
        except OSError as e:
            if "Cannot find empty port" in str(e) or "already in use" in str(e).lower():
                print(f"端口 {port} 被占用，尝试端口 {port + 1}...")
                port += 1
            else:
                raise e
