#!/usr/bin/env python3
#
# Copyright (c)  2025
# Dynamic Lyric Tracking ASR Inference using Sliding Window & VAD

import argparse
import logging
import re
import time
import os
import difflib
from typing import Tuple, List, Optional

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import AutoTokenizer

from .voice_activity_detector import SileroVAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from .utils import (
    build_source_ids,
    compute_feat,
    sample_token,
    select_device,
    EncoderAdaptorOnnxModel,
    EmbeddingOnnxIOB,
    UnifiedKvDeltaLLMOnnxIOB,
    device_from_str
)

def _run_vad_segments_1p1(
    vad_model: SileroVAD,
    samples: np.ndarray,
    sr: int,
    pad_sec: float = 0.0,
    merge_gap_sec: float = 0.0,
    min_seg_sec: float = 0.0,
    max_seg_sec: float = 20.0,
) -> List[Tuple[int, int]]:
    if sr != 16000:
        raise ValueError(f"Expected sr=16000 for VAD. Given: {sr}")

    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    max_val = np.abs(samples).max()
    if max_val > 1.0:
        samples = samples / max_val

    vad_model.reset()
    window_size_samples = vad_model.get_window_size()
    window_shift_samples = vad_model.get_window_shift()
    n_total = int(len(samples))
    
    max_seg_samples = int(round(float(max_seg_sec) * sr))
    original_threshold = vad_model.threshold
    original_min_silence_samples = vad_model.min_silence_samples
    original_min_silence_duration = vad_model.min_silence_duration
    
    new_threshold = 0.9
    new_min_silence_duration = 0.1
    new_min_silence_samples = int(round(new_min_silence_duration * sr))

    speech_segments = []
    speech_start = None
    buffer_tail = 0
    chunk_size = window_shift_samples * 10
    buffer_head = 0
    buffer_size = 0
    
    i = 0
    while i < n_total - window_size_samples + 1:
        remaining = n_total - window_size_samples - i
        windows_in_chunk = min(chunk_size // window_shift_samples, 
                              (remaining // window_shift_samples) + 1)
        
        if buffer_size > max_seg_samples:
            vad_model.threshold = new_threshold
            vad_model.min_silence_duration = new_min_silence_duration
            vad_model.min_silence_samples = new_min_silence_samples
        else:
            vad_model.threshold = original_threshold
            vad_model.min_silence_duration = original_min_silence_duration
            vad_model.min_silence_samples = original_min_silence_samples
        
        is_speech = False
        chunk_end = i
        
        for w in range(windows_in_chunk):
            window_start = i + w * window_shift_samples
            if window_start + window_size_samples > n_total:
                break
            
            chunk = samples[window_start:window_start + window_size_samples]
            this_window_is_speech = vad_model.is_speech(chunk)
            is_speech = is_speech or this_window_is_speech
            chunk_end = window_start + window_shift_samples
        
        buffer_tail = chunk_end
        buffer_size = buffer_tail - buffer_head
        
        if is_speech:
            if speech_start is None:
                min_speech_samples = vad_model.min_speech_samples
                lookback = 2 * window_size_samples + min_speech_samples
                speech_start = max(buffer_head, buffer_tail - lookback)
        else:
            if speech_start is not None:
                min_silence_samples = vad_model.min_silence_samples
                segment_end = buffer_tail - min_silence_samples
                if segment_end > speech_start:
                    speech_segments.append((speech_start, segment_end))
                buffer_head = segment_end
                buffer_size = buffer_tail - buffer_head
                speech_start = None
            
            if speech_start is None:
                end = buffer_tail - 2 * window_size_samples - vad_model.min_speech_samples
                samples_to_pop = max(0, end - buffer_head)
                if samples_to_pop > 0:
                    buffer_head += samples_to_pop
                    buffer_size = buffer_tail - buffer_head
        i = chunk_end

    if speech_start is not None:
        segment_end = min(buffer_tail, n_total)
        if segment_end > speech_start:
            speech_segments.append((speech_start, segment_end))

    if len(speech_segments) == 0:
        return []

    speech_segments.sort(key=lambda x: x[0])

    pad = int(round(float(pad_sec) * sr))
    if pad > 0:
        padded = []
        for s, e in speech_segments:
            ss = max(0, s - pad)
            ee = min(n_total, e + pad)
            padded.append((ss, ee))
        speech_segments = padded

    merge_gap = int(round(float(merge_gap_sec) * sr))
    if merge_gap > 0:
        merged = []
        cur_s, cur_e = speech_segments[0]
        for s, e in speech_segments[1:]:
            if s <= cur_e + merge_gap:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        speech_segments = merged

    min_seg = int(round(float(min_seg_sec) * sr))
    if min_seg > 0:
        speech_segments = [(s, e) for (s, e) in speech_segments if (e - s) >= min_seg]

    return speech_segments

def _split_segment_with_overlap(ss: int, ee: int, sr: int, max_len_sec: float, overlap_sec: float):
    max_len = int(round(max_len_sec * sr))
    overlap = int(round(overlap_sec * sr))
    if max_len <= 0:
        return [(ss, ee)]
    if overlap < 0:
        overlap = 0
    if overlap >= max_len:
        overlap = max_len // 2

    if ee - ss <= max_len:
        return [(ss, ee)]

    out = []
    cur = ss
    while cur < ee:
        nxt = min(cur + max_len, ee)
        out.append((cur, nxt))
        if nxt >= ee:
            break
        cur = nxt - overlap
    return out


def get_prompt(hotwords: List[str], language: Optional[str] = None, itn: bool = True) -> str:
    if len(hotwords) > 0:
        hotwords_s = ", ".join(hotwords)
        prompt = (
            "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n"
            "**上下文信息：**\n\n\n"
        )
        prompt += f"热词列表：[{hotwords_s}]\n"
    else:
        prompt = ""
    if language is None:
        prompt += "语音转写"
    else:
        prompt += f"语音转写成{language}"
    if not itn:
        prompt += "，不进行文本规整"
    return prompt + "："


def run_llm_inference(
    llm: UnifiedKvDeltaLLMOnnxIOB,
    embedding: EmbeddingOnnxIOB,
    tokenizer,
    inputs_embeds: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    im_end_token_id: Optional[int]
) -> List[int]:
    prompt_len = int(inputs_embeds.shape[1])
    
    B = 1
    caches_k, caches_v = llm.alloc_caches(batch=B, device=device)
    cache_position = torch.arange(0, prompt_len, device=device, dtype=torch.int64)
    attention_mask = torch.ones((B, prompt_len), device=device, dtype=torch.int64)

    logits_out = llm.alloc_logits(batch=B, device=device)
    kd_out, vd_out = llm.alloc_kv_deltas(batch=B, seq=prompt_len, device=device)

    logits, k_deltas, v_deltas = llm.run_iobinding(
        inputs_embeds, attention_mask, cache_position,
        caches_k, caches_v,
        logits_out, kd_out, vd_out
    )

    for i in range(llm.num_layers):
        caches_k[i][:, 0:prompt_len, :, :].copy_(k_deltas[i])
        caches_v[i][:, 0:prompt_len, :, :].copy_(v_deltas[i])

    next_logits = logits[0, 0, :]
    past_len = prompt_len
    generated = []

    kd_out_1, vd_out_1 = llm.alloc_kv_deltas(batch=B, seq=1, device=device)
    cache_pos_1 = torch.empty((1,), device=device, dtype=torch.int64)
    
    for step in range(max_new_tokens):
        if llm.max_total_len > 0 and past_len >= llm.max_total_len:
            break
            
        next_logits_np = next_logits.detach().cpu().numpy().astype(np.float32)
        tok = sample_token(next_logits_np, temperature=temperature, top_p=top_p, eos_token_id=eos_token_id, im_end_token_id=im_end_token_id, step=step)
        generated.append(tok)

        if step > 0:
            if eos_token_id is not None and tok == eos_token_id:
                break
            if im_end_token_id is not None and tok == im_end_token_id:
                break

        tok_t = torch.tensor([[tok]], device=device, dtype=torch.int64)
        tok_embeds = embedding.forward_ids(tok_t).to(dtype=llm.input_torch_dtype).contiguous()

        cache_pos_1[0] = past_len
        att_mask = torch.ones((B, past_len + 1), device=device, dtype=torch.int64)

        _, k_d_1, v_d_1 = llm.run_iobinding(
            tok_embeds, att_mask, cache_pos_1, caches_k, caches_v, logits_out, kd_out_1, vd_out_1
        )

        for i in range(llm.num_layers):
            caches_k[i][:, past_len:past_len + 1, :, :].copy_(k_d_1[i])
            caches_v[i][:, past_len:past_len + 1, :, :].copy_(v_d_1[i])

        past_len += 1
        next_logits = logits_out[0, 0, :]

    return generated

def find_best_match(recognized: str, candidate_lines: List[str]) -> Tuple[int, float]:
    best_idx = 0
    best_ratio = 0.0
    rec_clean = re.sub(r'[\W_]+', '', recognized)
    
    for i, line in enumerate(candidate_lines):
        line_clean = re.sub(r'[\W_]+', '', line)
        if not line_clean or not rec_clean:
            continue
        
        # 针对高语速/多句被合并到一个音频片段的情况：
        # 如果 ASR 结果很长，我们采用滑动窗口在 ASR 结果中寻找与单行歌词最像的局部
        L = len(line_clean)
        local_best = 0.0
        
        # 为了容错，窗口比原句稍微放宽几个字
        window_size = L + 3 
        
        if len(rec_clean) <= window_size:
            local_best = difflib.SequenceMatcher(None, line_clean, rec_clean).ratio()
        else:
            for j in range(len(rec_clean) - L + 1):
                sub_rec = rec_clean[j:min(len(rec_clean), j + window_size)]
                r = difflib.SequenceMatcher(None, line_clean, sub_rec).ratio()
                if r > local_best:
                    local_best = r
                    
        ratio = local_best
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
            
    return best_idx, min(1.0, best_ratio)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-adaptor-model", type=str, required=True)
    p.add_argument("--embedding-model", type=str, required=True)
    p.add_argument("--llm-model", type=str, required=True)
    p.add_argument("--llm-tokenizer", type=str, required=True)
    p.add_argument("--wave", type=str, required=True)
    p.add_argument("--lyrics-file", type=str, required=True, help="Text file with lyrics separated by newline")
    
    p.add_argument("--vad-model", type=str, required=True, help="Path to silero_vad.onnx")
    p.add_argument("--vad-pad-sec", type=float, default=0.30)
    p.add_argument("--vad-merge-gap-sec", type=float, default=0.20)
    p.add_argument("--vad-min-seg-sec", type=float, default=0.20)
    p.add_argument("--vad-max-seg-sec", type=float, default=20.0)
    p.add_argument("--vad-split-overlap-sec", type=float, default=0.40)
    
    p.add_argument("--window-size", type=int, default=8, help="How many lyric lines to feed at once")
    p.add_argument("--match-threshold", type=float, default=0.25, help="Minimum sequence match ratio to advance lyric pointer")
    
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    
    p.add_argument("--encoder-device", type=str, choices=["cpu", "cuda", "dml", "auto"], default="auto")
    p.add_argument("--llm-device", type=str, choices=["cpu", "cuda", "dml", "auto"], default="auto")
    p.add_argument("--embedding-device", type=str, choices=["cpu", "cuda", "dml", "auto"], default="auto")
    return p.parse_args()


def main():
    args = get_args()
    
    # Load lyrics (split long lines optionally, but we keep original)
    with open(args.lyrics_file, 'r', encoding='utf-8') as f:
        lyrics = [line.strip() for line in f if line.strip()]
        
    logging.info(f"Loaded {len(lyrics)} lines of lyrics.")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_tokenizer, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else None
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_token_id = im_end_ids[0] if len(im_end_ids) > 0 else None

    enc_dev = select_device(args.encoder_device)
    emb_dev = select_device(args.embedding_device)
    llm_dev = select_device(args.llm_device, model_path=args.llm_model)

    device = device_from_str("cuda" if llm_dev == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    encoder = EncoderAdaptorOnnxModel(args.encoder_adaptor_model, device=enc_dev)
    embedding = EmbeddingOnnxIOB(args.embedding_model, device=emb_dev)
    llm = UnifiedKvDeltaLLMOnnxIOB(args.llm_model, device=llm_dev)
    
    if not os.path.exists(args.vad_model):
        raise FileNotFoundError(f"--vad-model not found: {args.vad_model}")
    vad_model = SileroVAD(
        model_path=args.vad_model,
        threshold=0.5,
        min_silence_duration=0.5,
        min_speech_duration=0.25,
        window_size=512,
        max_speech_duration=20,
        sample_rate=16000,
        num_threads=1,
    )

    logging.info(f"[DEV] encoder={enc_dev}, llm={llm_dev}, device={device.type}")

    # Read and slice audio
    data, sr = sf.read(args.wave, always_2d=True, dtype="float32")
    waveform = data[:, 0]
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    segments = _run_vad_segments_1p1(
        vad_model=vad_model,
        samples=waveform,
        sr=sr,
        pad_sec=args.vad_pad_sec,
        merge_gap_sec=args.vad_merge_gap_sec,
        min_seg_sec=args.vad_min_seg_sec,
        max_seg_sec=args.vad_max_seg_sec,
    )
    if not segments:
        segments = [(0, len(waveform))]

    final_chunks = []
    for (ss, ee) in segments:
        sub_segs = _split_segment_with_overlap(ss, ee, sr, max_len_sec=args.vad_max_seg_sec, overlap_sec=args.vad_split_overlap_sec)
        for (ss2, ee2) in sub_segs:
            final_chunks.append({
                'offset': ss2 / sr,
                'waveform': waveform[ss2:ee2]
            })

    logging.info(f"Audio sliced into {len(final_chunks)} chunks using VAD.")

    def run_chunk_with_hotwords(wf, sr, hotwords):
        prompt_text = get_prompt(hotwords, language=None, itn=True)
        system_prompt = "You are a helpful assistant."
        user_prompt = f"{prompt_text}<|startofspeech|>!!<|endofspeech|>"
        
        feats = compute_feat(wf, sr, encoder.window_size, encoder.window_shift)[None, ...]
        enc_out = encoder(feats)
        enc_out = np.where(np.isfinite(enc_out), enc_out, 0.0)
        
        audio_token_len = int(enc_out.shape[1])
        
        source_ids_1d, fbank_beg_idx, fake_len = build_source_ids(tokenizer, system_prompt, user_prompt, audio_token_len)
        
        input_ids = torch.tensor(source_ids_1d.reshape(1, -1), device=device, dtype=torch.int64)
        text_embeds = embedding.forward_ids(input_ids).to(dtype=llm.input_torch_dtype)
        
        enc_t = torch.tensor(enc_out, device=device, dtype=llm.input_torch_dtype)
        fl = min(fake_len, enc_t.shape[1])
        text_embeds[0, fbank_beg_idx:fbank_beg_idx+fl, :] = enc_t[0, :fl, :]
        
        if llm.max_total_len > 0 and text_embeds.shape[1] >= llm.max_total_len:
            logging.warning(f"Chunk too long (prompt_len={text_embeds.shape[1]} >= {llm.max_total_len}). Skipping.")
            return None
            
        generated = run_llm_inference(
            llm=llm,
            embedding=embedding,
            tokenizer=tokenizer,
            inputs_embeds=text_embeds,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
            im_end_token_id=im_end_token_id
        )
        
        if generated:
            result_text = tokenizer.decode(generated, skip_special_tokens=True)
            result_text = result_text.replace("▁", " ").replace("<|im_end|>", "").replace("<|endoftext|>", "")
            return " ".join(result_text.split())
        return ""


    current_lyric_idx = 0
    is_aligned = False
    
    print("\n===== DYNAMIC LYRICS INFERENCE RESULTS =====\n")
    
    for idx, chunk in enumerate(final_chunks):
        offset = chunk['offset']
        wf = chunk['waveform']
        dur = len(wf) / sr
        
        # 1. Global Alignment (Blind Run) if not aligned yet
        if not is_aligned:
            raw_text = run_chunk_with_hotwords(wf, sr, [])
            if raw_text is None:
                continue
            
            best_idx, match_ratio = find_best_match(raw_text, lyrics)
            if match_ratio >= args.match_threshold:
                current_lyric_idx = best_idx
                is_aligned = True
                logging.info(f"[Global Aligned] Chunk '{raw_text}' matches L{best_idx}: '{lyrics[best_idx]}' (ratio {match_ratio:.2f})")
            else:
                logging.info(f"[Global Search] Chunk '{raw_text}' did not match any line significantly (best ratio {match_ratio:.2f}). Holding...")
                print(f"[{offset:.2f}s - {offset+dur:.2f}s] [UNALIGNED] {raw_text}")
                continue

        # 2. Local Tracking (Sliding Window) once aligned
        window_end = min(current_lyric_idx + args.window_size, len(lyrics))
        current_hotwords = lyrics[current_lyric_idx:window_end]
        
        result_text = run_chunk_with_hotwords(wf, sr, current_hotwords)
        if result_text is None:
            continue
            
        # 3. Alignment and Pointer Update
        best_idx, match_ratio = find_best_match(result_text, current_hotwords)
        
        if match_ratio >= args.match_threshold:
            matched_global_idx = current_lyric_idx + best_idx
            current_lyric_idx = matched_global_idx + 1
            align_status = f"[MATCH {match_ratio:.2f} -> L{matched_global_idx}]"
        else:
            align_status = f"[NO MATCH {match_ratio:.2f} -> HOLD]"
            
        print(f"[{offset:.2f}s - {offset+dur:.2f}s] {align_status} {result_text}")

if __name__ == "__main__":
    main()
