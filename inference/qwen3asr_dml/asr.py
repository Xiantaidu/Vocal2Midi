# coding=utf-8
import codecs
import dataclasses
import multiprocessing as mp
from pathlib import Path
import re
import time
from collections import deque
from typing import List, Optional

import numpy as np

from . import llama
from .asr_worker import asr_helper_worker_proc
from .schema import ASREngineConfig, DecodeResult, MsgType, StreamingMessage, TranscribeResult
from .utils import normalize_language_name, validate_language


@dataclasses.dataclass
class ASRSegment:
    idx: int
    audio_start: float
    audio_end: float
    text: str = ""


class QwenASREngine:
    def __init__(self, config: ASREngineConfig):
        self.config = config
        self.verbose = config.verbose
        if self.verbose:
            print(f"--- [QwenASR] Initializing engine (DML: {config.use_dml}) ---")

        llm_gguf = str(Path(config.model_dir) / config.llm_fn)
        self.to_worker_q = mp.Queue()
        self.from_enc_q = mp.Queue()
        self.helper_proc = mp.Process(
            target=asr_helper_worker_proc,
            args=(self.to_worker_q, self.from_enc_q, config),
            daemon=True,
        )
        self.helper_proc.start()

        self.model = llama.LlamaModel(llm_gguf, backend=config.llama_backend, quiet=not self.verbose)
        self.embedding_table = llama.get_token_embeddings_gguf(llm_gguf, quiet=not self.verbose)
        self.ctx = llama.LlamaContext(self.model, n_ctx=config.n_ctx, n_batch=4096, embeddings=False)

        msg = self.from_enc_q.get()
        if msg.msg_type == MsgType.MSG_ERROR:
            raise RuntimeError(f"worker failed to start:\n\n{msg.data}")
        self.encoder_runtime = msg.data or {}
        self.decoder_backend = getattr(self.model, "backend", "unknown")
        if msg.msg_type == MsgType.MSG_READY and self.verbose:
            print("--- [QwenASR] Worker is ready ---")

        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = self.model.token_to_id("<asr_text>")

    def shutdown(self) -> None:
        if self.helper_proc:
            self.to_worker_q.put(StreamingMessage(MsgType.CMD_STOP))
            self.helper_proc.join()
            self.helper_proc = None
        if self.verbose:
            print("--- [QwenASR] Engine closed ---")

    def _build_prompt_embd(
        self,
        audio_embd: np.ndarray,
        context: Optional[str],
        language: Optional[str],
    ) -> np.ndarray:
        def tk(text: str) -> List[int]:
            return self.model.tokenize(text)

        prefix_tokens = []
        if context:
            prefix_tokens.extend([self.ID_IM_START] + tk(f"system\n{context}") + [self.ID_IM_END])
            prefix_tokens.extend(tk("\n"))

        prefix_tokens.extend([self.ID_IM_START] + tk("user\n") + [self.ID_AUDIO_START])

        suffix_tokens = [self.ID_AUDIO_END, self.ID_IM_END]
        suffix_tokens.extend(tk("\n"))
        suffix_tokens.extend([self.ID_IM_START] + tk("assistant\n"))
        if language:
            suffix_tokens.extend(tk(f"language {language}"))
            suffix_tokens.append(self.ID_ASR_TEXT)

        n_pre = len(prefix_tokens)
        n_aud = audio_embd.shape[0]
        n_suf = len(suffix_tokens)
        total_embd = np.zeros((n_pre + n_aud + n_suf, self.model.n_embd), dtype=np.float32)
        total_embd[:n_pre] = self.embedding_table[prefix_tokens]
        total_embd[n_pre:n_pre + n_aud] = audio_embd
        total_embd[n_pre + n_aud:] = self.embedding_table[suffix_tokens]
        return total_embd

    def _decode(
        self,
        full_embd: np.ndarray,
        rollback_num: int,
        temperature: float = 0.0,
    ) -> DecodeResult:
        result = DecodeResult()
        total_len = full_embd.shape[0]
        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)])
        batch = self.llama_mod.LlamaBatch(max(total_len * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)

        self.ctx.clear_kv_cache()
        t_pre_start = time.time()
        self.ctx.decode(batch)
        result.t_prefill = time.time() - t_pre_start

        t_gen_start = time.time()
        display_queue = deque()
        stable_tokens = []
        stable_text = ""
        text_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        rollback_num = max(int(rollback_num), 0)
        seed = int(np.random.randint(0, 2**31 - 1))
        sampler = self.llama_mod.LlamaSampler(temperature=temperature, seed=seed)
        last_sampled_token = sampler.sample(self.ctx.ptr)
        for _ in range(self.config.max_decode_tokens):
            if last_sampled_token in [self.model.eos_token, self.ID_IM_END]:
                break
            if self.ctx.decode_token(last_sampled_token) != 0:
                break

            display_queue.append(last_sampled_token)
            if len(display_queue) > rollback_num:
                ready_token = display_queue.popleft()
                stable_tokens.append(ready_token)
                piece = text_decoder.decode(self.model.token_to_bytes(ready_token))
                if piece:
                    print(re.sub(r"([，。？！：,\.])", r"\1\n", piece), end="", flush=True)
                    stable_text += piece

            if len(stable_tokens) > 15 and len(set(stable_tokens[-15:])) <= 3:
                result.is_aborted = True
                break

            last_sampled_token = sampler.sample(self.ctx.ptr)
            result.n_generate += 1

        result.t_generate = time.time() - t_gen_start
        del sampler
        del batch

        if not result.is_aborted:
            while display_queue:
                token_id = display_queue.popleft()
                stable_tokens.append(token_id)
                piece = text_decoder.decode(self.model.token_to_bytes(token_id))
                if piece:
                    print(re.sub(r"([，。？！：,\.])", r"\1\n", piece), end="", flush=True)
                    stable_text += piece
            final_piece = text_decoder.decode(b"", final=True)
            if final_piece:
                print(final_piece, end="", flush=True)
                stable_text += final_piece

        result.text = stable_text
        result.stable_tokens = stable_tokens
        result.n_prefill = total_len
        return result

    @property
    def llama_mod(self):
        return llama

    def _safe_decode(
        self,
        full_embd: np.ndarray,
        rollback_num: int,
        temperature: float,
    ) -> DecodeResult:
        result = self._decode(full_embd, rollback_num, temperature)
        if result.is_aborted and self.verbose:
            print("\n\n[!] decode stopped by repetition safeguard.\n")
        return result

    def _print_stats(self, stats: dict, audio_duration: float, total_time: float) -> None:
        rtf = total_time / audio_duration if audio_duration > 0 else 0.0
        pre_speed = stats["prefill_tokens"] / stats["prefill_time"] if stats["prefill_time"] > 0 else 0.0
        gen_speed = stats["decode_tokens"] / stats["decode_time"] if stats["decode_time"] > 0 else 0.0
        print("\n\nPerformance:")
        print(f"  RTF            : {rtf:.3f}")
        print(f"  Audio duration : {audio_duration:.2f} s")
        print(f"  Total time     : {total_time:.2f} s")
        print(f"  Encode wait    : {stats['wait_time']:.2f} s")
        print(
            f"  LLM prefill    : {stats['prefill_time']:.3f} s "
            f"({stats['prefill_tokens']} tokens, {pre_speed:.1f} tok/s)"
        )
        print(
            f"  LLM decode     : {stats['decode_time']:.3f} s "
            f"({stats['decode_tokens']} tokens, {gen_speed:.1f} tok/s)"
        )

    def transcribe(
        self,
        audio_file: str,
        language: Optional[str] = None,
        context: Optional[str] = None,
        start_second: float = 0.0,
        duration: float = 0.0,
        temperature: float = 0.0,
        rollback_num: int = 5,
    ) -> TranscribeResult:
        from .utils import load_audio

        audio = load_audio(audio_file, start_second=start_second, duration=duration)
        return self.asr(
            audio=audio,
            context=context or "",
            language=language,
            chunk_size_sec=self.config.chunk_size,
            memory_chunks=self.config.memory_num,
            temperature=temperature,
            rollback_num=rollback_num,
        )

    def transcribe_batch(
        self,
        audio_files: List[str],
        language: Optional[str] = None,
        context: Optional[str] = None,
        start_second: float = 0.0,
        duration: float = 0.0,
        temperature: float = 0.0,
        rollback_num: int = 5,
    ) -> List[TranscribeResult]:
        from .utils import load_audio

        audios = [
            load_audio(audio_file, start_second=start_second, duration=duration)
            for audio_file in audio_files
        ]
        return self.asr_batch(
            audios=audios,
            context=context or "",
            language=language,
            chunk_size_sec=self.config.chunk_size,
            memory_chunks=self.config.memory_num,
            temperature=temperature,
            rollback_num=rollback_num,
        )

    def asr(
        self,
        audio: np.ndarray,
        context: Optional[str],
        language: Optional[str],
        chunk_size_sec: float = 40.0,
        memory_chunks: int = 1,
        temperature: float = 0.0,
        rollback_num: int = 5,
    ) -> TranscribeResult:
        if language:
            language = normalize_language_name(language)
            validate_language(language)
        _ = memory_chunks  # Kept for API compatibility; offline decoding follows official per-chunk inference.

        sr = 16000
        samples_per_chunk = int(chunk_size_sec * sr)
        total_len = len(audio)
        num_chunks = int(np.ceil(total_len / samples_per_chunk))
        total_duration = total_len / sr
        total_full_text = ""
        stats = {
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "wait_time": 0.0,
            "encode_time": 0.0,
        }
        t_main_start = time.time()

        def send_enc(idx: int) -> None:
            if idx >= num_chunks:
                return
            start = idx * samples_per_chunk
            end = min((idx + 1) * samples_per_chunk, total_len)
            data = audio[start:end]
            if len(data) < samples_per_chunk:
                data = np.pad(data, (0, samples_per_chunk - len(data)))
            self.to_worker_q.put(
                StreamingMessage(
                    MsgType.CMD_ENCODE,
                    data=data,
                    is_last=(idx == num_chunks - 1),
                )
            )

        if num_chunks > 0:
            send_enc(0)

        for idx in range(num_chunks):
            t_wait_start = time.time()
            msg = self.from_enc_q.get()
            stats["wait_time"] += time.time() - t_wait_start
            stats["encode_time"] += msg.encode_time
            audio_feature = msg.data
            was_last = msg.is_last

            if not was_last:
                send_enc(idx + 1)

            full_embd = self._build_prompt_embd(audio_feature, context, language)
            res = self._safe_decode(full_embd, rollback_num, temperature)

            total_full_text += res.text
            stats["prefill_tokens"] += res.n_prefill
            stats["prefill_time"] += res.t_prefill
            stats["decode_tokens"] += res.n_generate
            stats["decode_time"] += res.t_generate

        total_time = time.time() - t_main_start
        if self.verbose:
            self._print_stats(stats, total_duration, total_time)
        return TranscribeResult(text=total_full_text, performance=stats)

    def asr_batch(
        self,
        audios: List[np.ndarray],
        context: Optional[str],
        language: Optional[str],
        chunk_size_sec: float = 40.0,
        memory_chunks: int = 1,
        temperature: float = 0.0,
        rollback_num: int = 5,
    ) -> List[TranscribeResult]:
        if language:
            language = normalize_language_name(language)
            validate_language(language)
        _ = memory_chunks  # Kept for API compatibility; offline decoding follows official per-chunk inference.

        if not audios:
            return []

        sr = 16000
        samples_per_chunk = int(chunk_size_sec * sr)
        states = []
        for audio in audios:
            total_len = len(audio)
            num_chunks = int(np.ceil(total_len / samples_per_chunk)) if total_len > 0 else 0
            states.append(
                {
                    "audio": audio,
                    "total_len": total_len,
                    "total_duration": total_len / sr,
                    "num_chunks": num_chunks,
                    "text": "",
                    "stats": {
                        "prefill_time": 0.0,
                        "decode_time": 0.0,
                        "prefill_tokens": 0,
                        "decode_tokens": 0,
                        "wait_time": 0.0,
                        "encode_time": 0.0,
                    },
                }
            )

        max_rounds = max((state["num_chunks"] for state in states), default=0)
        for round_idx in range(max_rounds):
            active_indices = []
            batch_audio = []
            for state_idx, state in enumerate(states):
                if round_idx >= state["num_chunks"]:
                    continue
                start = round_idx * samples_per_chunk
                end = min((round_idx + 1) * samples_per_chunk, state["total_len"])
                data = state["audio"][start:end]
                if len(data) < samples_per_chunk:
                    data = np.pad(data, (0, samples_per_chunk - len(data)))
                batch_audio.append(data)
                active_indices.append(state_idx)

            if not batch_audio:
                continue

            t_wait_start = time.time()
            self.to_worker_q.put(StreamingMessage(MsgType.CMD_ENCODE, data=batch_audio))
            msg = self.from_enc_q.get()
            wait_time = time.time() - t_wait_start
            if msg.msg_type == MsgType.MSG_ERROR:
                raise RuntimeError(msg.data)

            audio_features = list(msg.data)
            encode_time = float(msg.encode_time)
            shared_weight = 1.0 / len(active_indices)

            for batch_idx, state_idx in enumerate(active_indices):
                state = states[state_idx]
                stats = state["stats"]
                stats["wait_time"] += wait_time * shared_weight
                stats["encode_time"] += encode_time * shared_weight

                audio_feature = audio_features[batch_idx]
                full_embd = self._build_prompt_embd(audio_feature, context, language)
                res = self._safe_decode(full_embd, rollback_num, temperature)

                state["text"] += res.text
                stats["prefill_tokens"] += res.n_prefill
                stats["prefill_time"] += res.t_prefill
                stats["decode_tokens"] += res.n_generate
                stats["decode_time"] += res.t_generate

        results = []
        for state in states:
            if self.verbose and state["num_chunks"] > 0:
                total_time = (
                    state["stats"]["wait_time"]
                    + state["stats"]["prefill_time"]
                    + state["stats"]["decode_time"]
                )
                self._print_stats(state["stats"], state["total_duration"], total_time)
            results.append(TranscribeResult(text=state["text"], performance=state["stats"]))
        return results
