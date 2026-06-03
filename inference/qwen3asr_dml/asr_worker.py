# coding=utf-8
import os

from . import logger
from .encoder import QwenAudioEncoder
from .schema import ASREngineConfig, MsgType, StreamingMessage


def do_encode_task(msg: StreamingMessage, encoder: QwenAudioEncoder, from_enc_q) -> None:
    if isinstance(msg.data, list):
        audio_embd, encode_time = encoder.encode_batch(msg.data)
    else:
        audio_embd, encode_time = encoder.encode(msg.data)
    from_enc_q.put(
        StreamingMessage(
            msg_type=MsgType.MSG_EMBD,
            data=audio_embd,
            is_last=msg.is_last,
            encode_time=encode_time,
        )
    )


def asr_helper_worker_proc(to_worker_q, from_enc_q, config: ASREngineConfig) -> None:
    try:
        frontend_path = os.path.join(config.model_dir, config.encoder_frontend_fn)
        backend_path = os.path.join(config.model_dir, config.encoder_backend_fn)
        encoder = QwenAudioEncoder(
            frontend_path=frontend_path,
            backend_path=backend_path,
            use_dml=config.use_dml,
            warmup_sec=min(float(config.chunk_size), 5.0),
            verbose=False,
        )
        from_enc_q.put(
            StreamingMessage(
                MsgType.MSG_READY,
                data={
                    "encoder_provider": encoder.provider_name,
                    "frontend_providers": list(encoder.frontend_providers),
                    "backend_providers": list(encoder.backend_providers),
                },
            )
        )
    except Exception as exc:
        logger.error(f"[ASRWorker] failed to initialize encoder:\n{exc}")
        from_enc_q.put(StreamingMessage(MsgType.MSG_ERROR, data=exc))
        return

    while True:
        msg: StreamingMessage = to_worker_q.get()
        if msg.msg_type == MsgType.CMD_STOP:
            from_enc_q.put(StreamingMessage(MsgType.MSG_DONE))
            break
        if msg.msg_type == MsgType.CMD_ENCODE:
            do_encode_task(msg, encoder, from_enc_q)
