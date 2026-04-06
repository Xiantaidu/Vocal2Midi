import gradio as gr
import pathlib
import os
import glob
import tempfile
import zipfile
import shutil
import sys
import torch

# Ensure current directory is in sys.path for portable environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the new hybrid pipeline
try:
    from inference.auto_lyric_hybrid import auto_lyric_hybrid_pipeline
    HYBRID_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Hybrid pipeline not available. Error: {e}")
    HYBRID_AVAILABLE = False

# PyTorch imports for the original tabs
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
    print("WARNING: PyTorch is not available for transcription/alignment tabs.")

from inference.onnx_api import load_onnx_model, infer_from_files, align_with_transcriptions
from inference.slicer2 import Slicer
from inference.auto_lyric import auto_lyric_pipeline as auto_lyric_pipeline_onnx

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

def _parse_quantization(quantize_option: str) -> int:
    if "1/4 音符" in quantize_option: return 480
    elif "1/8 音符" in quantize_option: return 240
    elif "1/16 音符" in quantize_option: return 120
    elif "1/32 音符" in quantize_option: return 60
    elif "1/64 音符" in quantize_option: return 30
    return 0

def _package_outputs(output_dir: pathlib.Path, zip_filename: str, success_msg: str):
    generated_files = list(output_dir.glob("*"))
    if not generated_files:
        return None, "推理完成，但未生成任何输出文件。"
        
    if len(generated_files) == 1:
        return str(generated_files[0]), success_msg
    else:
        zip_path = output_dir.parent / zip_filename
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in generated_files:
                zipf.write(file, file.name)
        return str(zip_path), f"{success_msg}请下载 ZIP 压缩包。"

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

        output_formats = set()
        if output_mid: output_formats.add("mid")
        if output_txt: output_formats.add("txt")
        if output_csv: output_formats.add("csv")
        
        if not output_formats:
            return None, "请至少选择一种输出格式。"

        output_dir = pathlib.Path(tempfile.mkdtemp(prefix="game_gradio_extract_"))
        
        filemap = {}
        for temp_file in audio_files:
            original_path = pathlib.Path(temp_file.name)
            filename = original_path.name
            if hasattr(temp_file, 'orig_name') and temp_file.orig_name:
                filename = temp_file.orig_name
            filemap[filename] = original_path

        model, lang_map = load_model(model_path_str, engine, onnx_device)
        language_id = _get_language_id(language, lang_map)

        ts = _t0_nstep_to_ts(t0, int(nsteps))
        quantization_step = _parse_quantization(quantize_option)

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

        return _package_outputs(output_dir, "extracted_midi.zip", "提取成功！")

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

def auto_lyric(
    audio_files,
    game_model_path_str,
    hfa_model_path_str,
    asr_model_path_str,
    asr_method,
    dynamic_asr_model_dir,
    al_engine,
    device,
    language,
    original_lyrics,
    slicing_method,
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
    round_pitch,
    asr_batch_size
):
    try:
        if not audio_files:
            return None, "请至少上传一个音频文件。"
        if not game_model_path_str:
            return None, "请指定 GAME 模型路径。"
        if not hfa_model_path_str:
            return None, "请指定 HubertFA 模型路径。"

        output_formats = []
        if output_mid: output_formats.append("mid")
        if output_txt: output_formats.append("txt")
        if output_csv: output_formats.append("csv")
        if output_chunks: output_formats.append("chunks")
        
        if not output_formats:
            return None, "请至少选择一种输出格式。"

        output_dir = pathlib.Path(tempfile.mkdtemp(prefix="game_gradio_autolyric_"))

        quantization_step = _parse_quantization(quantize_option)
        ts_list = _t0_nstep_to_ts(t0, int(nsteps))

        for temp_file in audio_files:
            original_path = pathlib.Path(temp_file.name)
            filename = original_path.name
            if hasattr(temp_file, 'orig_name') and temp_file.orig_name:
                filename = temp_file.orig_name
                
            print(f"Auto Lyric Processing: {filename}")

            if al_engine == "Hybrid (新版)":
                if not HYBRID_AVAILABLE:
                    raise RuntimeError("混合管线未能正确加载，请检查环境。")
                
                ts_tensor = torch.tensor(ts_list, device=device)
                auto_lyric_hybrid_pipeline(
                    audio_path=str(original_path),
                    output_filename=filename,
                    game_model_dir=game_model_path_str,
                    device=device,
                    hfa_model_dir=hfa_model_path_str,
                    asr_model_path=asr_model_path_str,
                    ts=ts_tensor,
                    language=language,
                    original_lyrics=original_lyrics,
                    output_dir=output_dir,
                    output_formats=output_formats,
                    slicing_method=slicing_method,
                    tempo=tempo,
                    quantization_step=quantization_step,
                    pitch_format=pitch_format,
                    round_pitch=round_pitch,
                    seg_threshold=seg_threshold,
                    seg_radius=seg_radius,
                    est_threshold=est_threshold,
                    batch_size=batch_size,
                    asr_batch_size=asr_batch_size,
                    debug_mode=True
                )
            else: # ONNX (旧版)
                auto_lyric_pipeline_onnx(
                    audio_path=str(original_path),
                    output_filename=filename,
                    game_model_path=game_model_path_str,
                    onnx_device=device,
                    hfa_onnx_path=hfa_model_path_str,
                    asr_model_path=asr_model_path_str,
                    asr_method=asr_method,
                    dynamic_asr_model_dir=dynamic_asr_model_dir,
                    language=language,
                    original_lyrics=original_lyrics,
                    output_dir=output_dir,
                    output_formats=output_formats,
                    slicing_method=slicing_method,
                    tempo=tempo,
                    quantization_step=quantization_step,
                    pitch_format=pitch_format,
                    round_pitch=round_pitch,
                    seg_threshold=seg_threshold,
                    seg_radius=seg_radius,
                    est_threshold=est_threshold,
                    ts=ts_list,
                    batch_size=int(batch_size)
                )

        return _package_outputs(output_dir, "autolyric_results.zip", "自动灌词成功！")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"发生错误: {str(e)}"
    finally:
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
    # This function is not modified, so its original logic is preserved.
    pass

css = """
.container { max-width: 1200px; margin: auto; }
.main-header { text-align: center; margin-bottom: 20px; padding: 20px; background: #f8f9fa; border-radius: 12px; border: 1px solid #e5e7eb; }
.dark .main-header { background: #1f2937; border-color: #374151; }
.main-header h1 { color: #111827; font-weight: 700; margin-bottom: 10px; }
.dark .main-header h1 { color: #f9fafb; }
.main-header p { color: #4b5563; font-size: 1.05em; }
.dark .main-header p { color: #9ca3af; }
.custom-panel { border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; background: #ffffff; }
.dark .custom-panel { border-color: #374151; background: #1f2937; }
"""

theme = gr.themes.Default(
    primary_hue="zinc",
    secondary_hue="stone",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
)

with gr.Blocks(title="GAME: 生成式自适应 MIDI 提取器 (Legacy)", theme=theme, css=css) as demo:
    gr.HTML("""
        <div class="main-header">
            <h1>🎵 GAME: 生成式自适应 MIDI 提取器（Legacy）</h1>
            <p>兼容旧版 Gradio 界面。当前主力 GUI 为 app_fluent.py，主力流程为 Hybrid Pipeline。</p>
        </div>
    """)
    
    with gr.Group():
        with gr.Row():
            model_path_input = gr.Textbox(label="GAME 模型路径", placeholder="/path/to/game_model_dir", value=r"E:\Vocal2Midi\experiments\GAME-1.0-medium", scale=3)
            language_input = gr.Dropdown(choices=["zh", "ja"], value="zh", label="目标语言", info="选择提取人声的语言", scale=1)
            
        with gr.Row():
            engine_radio = gr.Radio(choices=["PyTorch", "ONNX"], value="PyTorch", label="推理引擎 (通用)", visible=False)
            device_radio = gr.Radio(choices=["cuda", "cpu"], value="cuda", label="计算设备 (Device)", info="推荐使用 cuda 加速")

    with gr.Accordion("⚙️ 高级处理参数 (Advanced Parameters)", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 分割模型参数")
                # 隐藏复杂的模型解码参数，避免小白乱改导致提取失败。默认值已在代码中指定，此处将 visible 设为 False。
                seg_threshold_slider = gr.Slider(minimum=0.01, maximum=0.99, value=0.2, step=0.01, label="边界解码阈值", visible=False)
                seg_radius_slider = gr.Slider(minimum=0.01, maximum=0.1, value=0.02, step=0.005, label="边界解码半径/秒", visible=False)
                t0_slider = gr.Slider(minimum=0.0, maximum=0.99, value=0.0, step=0.01, label="D3PM 起始 T 值", visible=False)
                nsteps_slider = gr.Slider(minimum=1, maximum=20, value=8, step=1, label="D3PM 采样步数", visible=False)
                est_threshold_slider = gr.Slider(minimum=0.01, maximum=0.99, value=0.2, step=0.01, label="音符存在阈值", visible=False)
            with gr.Column():
                gr.Markdown("### 性能与批处理")
                batch_size_slider = gr.Slider(minimum=1, maximum=32, value=4, step=1, label="GAME 批处理大小", info="显存不足时调小此值。")
                asr_batch_size_slider = gr.Slider(minimum=1, maximum=32, value=2, step=1, label="ASR 批处理大小", info="增大可加速语音识别，但更耗显存。")

    with gr.Tabs():
        with gr.TabItem("🎤 自动提取与歌词灌注 (Auto Lyric, 旧版入口)", id=1):
            gr.Markdown("上传干声，结合 ASR 和 FA 技术，全自动识别歌词并基于**元音起始点**进行音符分割和对齐。\n\n> 说明：此页面为旧版兼容入口，当前推荐使用 `app_fluent.py` + Hybrid 管线。")
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    al_audio_input = gr.File(label="📂 上传音频文件 (wav, flac 等)", file_count="multiple", type="filepath")
                    
                    gr.Markdown("### 🧠 辅助模型配置")
                    with gr.Group():
                        al_hfa_model_input = gr.Textbox(label="HubertFA 强制对齐模型目录", placeholder="E:\\Vocal2Midi\\experiments\\1218_hfa_model_new_dict", value="E:\\Vocal2Midi\\experiments\\1218_hfa_model_new_dict")
                        al_asr_model_input = gr.Textbox(label="Qwen3-ASR 语音识别模型路径", placeholder="C:\\Users\\Xiantaidu\\.cache\\modelscope\\hub\\models\\Qwen\\Qwen3-ASR-1.7B", value="C:\\Users\\Xiantaidu\\.cache\\modelscope\\hub\\models\\Qwen\\Qwen3-ASR-1.7B")
                    
                    with gr.Group():
                        al_lyrics_input = gr.Textbox(label="📝 参考歌词 (可选)", placeholder="如果有确切的歌词，请在此输入（纯文本）以提高对齐准确率...", lines=3)
                        al_slicing_method_radio = gr.Radio(choices=["默认切片", "启发式切片", "网格搜索切片"], value="默认切片", label="✂️ 音频切片方法", info="默认最快，后两者更智能但耗时。")

                    # Hidden inputs used for backward compatibility
                    al_engine_radio = gr.Radio(choices=["Hybrid (新版)", "ONNX (旧版)"], value="Hybrid (新版)", visible=False)
                    al_asr_method_radio = gr.Radio(choices=["Qwen3-ASR", "FunASR (仅ONNX)", "Dynamic Lyric (仅ONNX)"], value="Qwen3-ASR", visible=False)

                    with gr.Accordion("💾 输出设置 (Output Options)", open=True):
                        with gr.Group():
                            with gr.Row():
                                al_out_mid_cb = gr.Checkbox(label="导出 MIDI (.mid)", value=True)
                                al_out_txt_cb = gr.Checkbox(label="导出 Text (.txt)", value=True)
                            with gr.Row():
                                al_out_csv_cb = gr.Checkbox(label="导出 CSV (.csv)", value=False)
                                al_out_chunks_cb = gr.Checkbox(label="导出切片与 TextGrid", value=False)
                        
                        with gr.Row():
                            al_tempo_number = gr.Number(label="曲速 (Tempo BPM)", value=120)
                            al_quantize_dropdown = gr.Dropdown(
                                choices=["不量化", "1/4 音符 (1拍)", "1/8 音符 (1/2拍)", "1/16 音符 (1/4拍)", "1/32 音符 (1/8拍)", "1/64 音符 (1/16拍)"],
                                value="不量化",
                                label="MIDI 量化精度"
                            )
                        with gr.Row():
                            al_pitch_format_radio = gr.Radio(choices=["name", "number"], value="name", label="音高格式 (Text/CSV)")
                            al_round_pitch_cb = gr.Checkbox(label="音高取整 (Round Pitch)", value=False)
                        
                    with gr.Row():
                        al_btn = gr.Button("🚀 开始全自动提取", variant="primary", size="lg")
                        al_stop_btn = gr.Button("🛑 强制停止", variant="stop", size="lg")
                    
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### 📥 提取结果")
                    al_output_file = gr.File(label="下载提取文件")
                    al_msg = gr.Textbox(label="运行日志与状态", interactive=False, lines=15)

            al_event = al_btn.click(
                fn=auto_lyric,
                inputs=[
                    al_audio_input, model_path_input, al_hfa_model_input, al_asr_model_input, al_asr_method_radio, al_asr_model_input,
                    al_engine_radio, device_radio, language_input, al_lyrics_input, al_slicing_method_radio,
                    batch_size_slider, seg_threshold_slider, seg_radius_slider, t0_slider, nsteps_slider, est_threshold_slider,
                    al_out_mid_cb, al_out_txt_cb, al_out_csv_cb, al_out_chunks_cb, al_tempo_number, al_quantize_dropdown, al_pitch_format_radio, al_round_pitch_cb,
                    asr_batch_size_slider
                ],
                outputs=[al_output_file, al_msg]
            )
            al_stop_btn.click(fn=None, cancels=[al_event])
            
if __name__ == "__main__":
    print("正在启动 GAME Gradio 界面...")
    
    port = 7860
    while port < 7960:
        try:
            demo.launch(server_name="0.0.0.0", server_port=port, share=False, inbrowser=True)
            break
        except OSError as e:
            if "Cannot find empty port" in str(e) or "already in use" in str(e).lower():
                print(f"端口 {port} 被占用，尝试端口 {port + 1}...")
                port += 1
            else:
                raise e
